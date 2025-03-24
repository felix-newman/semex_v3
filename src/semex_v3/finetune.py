# Adapted from https://github.com/zhangfaen/finetune-Qwen2.5-VL/blob/main/finetune_distributed.py
import json
from pathlib import Path
from typing import Optional

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from torch.utils.data import DataLoader
from torch.optim import AdamW
from functools import partial
import wandb
from loguru import logger
from dotenv import load_dotenv
from accelerate import Accelerator, DeepSpeedPlugin

from semex_v3.config import Config
from semex_v3.data import ToyDataSet, collate_fn

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

    train_loader = DataLoader(
        ToyDataSet("data/conversations.json"),
        batch_size=config.training.batch_size,
        collate_fn=partial(collate_fn, processor=processor, device=accelerator.device),
    )

    model.train()
    optimizer = AdamW(model.parameters(), lr=config.training.learning_rate)
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

    # Calculate total number of update steps
    total_train_batch_size = (
        config.training.batch_size 
        * accelerator.num_processes 
        * config.deepspeed.gradient_accumulation_steps
    )
    num_update_steps_per_epoch = len(train_loader) // config.deepspeed.gradient_accumulation_steps
    num_total_update_steps = num_update_steps_per_epoch * config.training.epochs

    global_step = 0
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

                # Only increment global_step after gradient accumulation steps
                if steps % config.deepspeed.gradient_accumulation_steps == 0:
                    global_step += 1

                    if accelerator.is_local_main_process and config.wandb.enabled:
                        wandb.log(
                            {
                                "train/loss": loss.item(),
                                "train/epoch": epoch + 1,
                                "train/global_step": global_step,
                                "train/learning_rate": optimizer.param_groups[0]["lr"],
                            },
                            step=global_step,
                        )

                        logger.info(
                            f"Step {global_step}/{num_total_update_steps} "
                            f"(epoch {epoch + 1}/{config.training.epochs}), "
                            f"training loss: {loss.item():.10f}"
                        )

    # Cleanup wandb
    if accelerator.is_local_main_process and config.wandb.enabled:
        wandb.finish()

    # Save model and processor
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
