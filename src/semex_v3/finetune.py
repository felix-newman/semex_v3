# Adapted from https://github.com/zhangfaen/finetune-Qwen2.5-VL/blob/main/finetune_distributed.py
import json
from pathlib import Path
from typing import Optional, Dict
import tempfile

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from torch.utils.data import DataLoader
from torch.optim import AdamW
from functools import partial
import wandb
from loguru import logger
from dotenv import load_dotenv
from accelerate import Accelerator, DeepSpeedPlugin
from vllm import LLM
import torch
import random

from semex_v3.config import Config
from semex_v3.data import QwenDataset, collate_fn

load_dotenv()


def setup_accelerator(config: Config) -> Accelerator:
    """Initialize and return the Accelerator with DeepSpeed configuration."""
    print("Init deepspeed plugin...")
    deepspeed_plugin = DeepSpeedPlugin(
        zero_stage=config.deepspeed.zero_stage,
        gradient_accumulation_steps=config.deepspeed.gradient_accumulation_steps,
        zero3_save_16bit_model=config.deepspeed.zero3_save_16bit_model,
        offload_optimizer_device=config.deepspeed.offload_optimizer_device,
        offload_param_device=config.deepspeed.offload_param_device,
    )
    print("Init deepspeed plugin done")

    print("Init accelerator...")
    accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin)
    print("Init accelerator done")
    return accelerator


def write_chat_template(processor, output_dir: Path) -> None:
    """Save chat template to a JSON file."""
    output_chat_template_file = output_dir / "chat_template.json"
    chat_template_json_string = (
        json.dumps({"chat_template": processor.chat_template}, indent=2, sort_keys=True)
        + "\n"
    )
    output_chat_template_file.write_text(chat_template_json_string, encoding="utf-8")
    logger.info(f"Chat template saved in {output_chat_template_file}")


def evaluate_with_vllm(
    model_path: Path,
    eval_dataset: QwenDataset,
    num_samples: int = 100,
    tensor_parallel_size: int = 1,
) -> Dict[str, float]:
    """Run evaluation using VLLM.

    Args:
        model_path: Path to the saved model checkpoint
        eval_dataset: Dataset to evaluate on
        num_samples: Number of samples to evaluate
        tensor_parallel_size: Number of GPUs to use for tensor parallelism

    Returns:
        Dictionary containing evaluation metrics
    """
    # Sample random indices for evaluation
    eval_indices = random.sample(
        range(len(eval_dataset)), min(num_samples, len(eval_dataset))
    )
    eval_subset = [eval_dataset[i] for i in eval_indices]

    # Initialize VLLM
    llm = LLM(
        model=str(model_path),
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
        dtype="bfloat16",
    )

    # Prepare prompts
    prompts = []
    for item in eval_subset:
        messages = item["messages"]
        # We only take the system and user messages for prompting
        prompt_messages = [msg for msg in messages if msg["role"] != "assistant"]
        prompts.append(prompt_messages)

    # Generate responses
    outputs = llm.generate(prompts, temperature=0.7, max_tokens=512)

    # Calculate metrics (for now, just average length)
    avg_length = sum(len(output.text.split()) for output in outputs) / len(outputs)

    return {
        "eval/avg_response_length": avg_length,
        # Add more metrics as needed
    }


def train(config: Optional[Config] = None) -> None:
    """Main training function."""
    if config is None:
        config = Config()

    accelerator = setup_accelerator(config)
    config.training.output_dir.mkdir(parents=True, exist_ok=True)

    if accelerator.is_local_main_process and config.wandb.enabled:
        wandb.init(
            project=config.wandb.project,
            config=config.to_wandb_config(),
        )

    # Load model and processor
    print("Loading model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config.model.name, torch_dtype=config.model.torch_dtype
    )

    processor = AutoProcessor.from_pretrained(
        config.model.name,
        min_pixels=config.model.min_pixels,
        max_pixels=config.model.max_pixels,
        padding_side=config.model.padding_side,
    )

    # Load datasets
    train_dataset = QwenDataset("data/conversations.json")
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        collate_fn=partial(collate_fn, processor=processor, device=accelerator.device),
    )

    model.train()
    optimizer = AdamW(model.parameters(), lr=config.training.learning_rate)
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

    # Calculate total number of update steps
    num_update_steps_per_epoch = (
        len(train_loader) // config.deepspeed.gradient_accumulation_steps
    )
    num_total_update_steps = num_update_steps_per_epoch * config.training.epochs

    for epoch in range(config.training.epochs):
        steps = 0
        epoch_loss = 0.0
        for batch in train_loader:
            steps += 1
            with accelerator.accumulate(model):
                inputs, labels = batch
                outputs = model(**inputs, labels=labels)
                loss = outputs.loss
                epoch_loss += loss.item()
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

                if accelerator.is_local_main_process and config.wandb.enabled:
                    wandb.log(
                        {
                            "train/loss": loss.item(),
                            "train/epoch": epoch + 1,
                            "train/learning_rate": optimizer.param_groups[0]["lr"],
                        },
                        step=steps,
                    )

                    logger.info(
                        f"Step {steps}/{num_total_update_steps} "
                        f"(epoch {epoch + 1}/{config.training.epochs}), "
                        f"training loss: {loss.item():.10f}"
                    )

        # Run evaluation at the end of each epoch
        if accelerator.is_local_main_process:
            # Create a temporary directory for the checkpoint
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir)

                # Save current model state
                logger.info("Saving temporary checkpoint for evaluation...")
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(
                    tmp_path,
                    is_main_process=True,
                    save_function=accelerator.save,
                )
                processor.save_pretrained(tmp_path)

                # Run evaluation
                logger.info("Running VLLM evaluation...")
                eval_metrics = evaluate_with_vllm(
                    tmp_path,
                    train_dataset,
                    num_samples=100,
                    tensor_parallel_size=torch.cuda.device_count(),
                )

                if config.wandb.enabled:
                    wandb.log(
                        {
                            **eval_metrics,
                            "train/epoch": epoch + 1,
                        },
                        step=steps,
                    )

    # Cleanup wandb
    if accelerator.is_local_main_process and config.wandb.enabled:
        wandb.finish()

    # Save final model and processor
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(
        config.training.output_dir,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
        max_shard_size="20GB",
        state_dict=accelerator.get_state_dict(model),
    )

    if accelerator.is_local_main_process:
        processor.save_pretrained(config.training.output_dir)
        write_chat_template(processor, config.training.output_dir)


if __name__ == "__main__":
    train()
