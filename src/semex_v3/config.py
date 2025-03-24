from pathlib import Path
from datetime import datetime
from typing import Literal
from pydantic import BaseModel, Field


class DeepSpeedConfig(BaseModel):
    """DeepSpeed-specific configuration parameters."""

    zero_stage: Literal[0, 1, 2, 3] = Field(
        3, description="ZeRO optimization stage (0-3)"
    )
    gradient_accumulation_steps: int = Field(
        2, description="Number of steps to accumulate gradients over", gt=0
    )
    zero3_save_16bit_model: bool = Field(
        True, description="Save model in 16-bit precision when using ZeRO-3"
    )
    offload_optimizer_device: Literal["cpu", "nvme"] = Field(
        "cpu", description="Device to offload optimizer states to"
    )
    offload_param_device: Literal["cpu", "nvme"] = Field(
        "cpu", description="Device to offload parameters to"
    )


class ModelConfig(BaseModel):
    """Model-specific configuration parameters."""

    name: str = Field(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        description="Name or path of the pretrained model",
    )
    torch_dtype: str = Field("bfloat16", description="PyTorch dtype for model weights")
    min_pixels: int = Field(
        128 * 28 * 28, description="Minimum number of pixels for image processing"
    )
    max_pixels: int = Field(
        256 * 28 * 28, description="Maximum number of pixels for image processing"
    )
    padding_side: Literal["left", "right"] = Field(
        "right", description="Side to pad inputs on"
    )


class TrainingConfig(BaseModel):
    """Training-specific configuration parameters."""

    epochs: int = Field(4, description="Number of training epochs", gt=0)
    batch_size: int = Field(2, description="Training batch size per device", gt=0)
    learning_rate: float = Field(1e-5, description="Learning rate", gt=0)
    output_dir: Path = Field(
        Path("train_output") / datetime.now().strftime("%Y%m%d%H%M%S"),
        description="Directory to save model outputs",
    )


class WandbConfig(BaseModel):
    """Weights & Biases configuration parameters."""

    project: str = Field("semex-v3", description="W&B project name")
    enabled: bool = Field(True, description="Whether to enable W&B logging")


class Config(BaseModel):
    """Main configuration class combining all sub-configurations."""

    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    deepspeed: DeepSpeedConfig = Field(default_factory=DeepSpeedConfig)
    wandb: WandbConfig = Field(default_factory=WandbConfig)

    def to_wandb_config(self) -> dict:
        """Convert relevant config parameters to W&B config format."""
        return {
            "epochs": self.training.epochs,
            "batch_size": self.training.batch_size,
            "learning_rate": self.training.learning_rate,
            "gradient_accumulation_steps": self.deepspeed.gradient_accumulation_steps,
            "model": self.model.name,
            "zero_stage": self.deepspeed.zero_stage,
        }
