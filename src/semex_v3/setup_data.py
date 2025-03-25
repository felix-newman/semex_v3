from typing import Tuple
from pathlib import Path
from PIL import Image
from loguru import logger
from concurrent.futures import ThreadPoolExecutor
import json
from datasets import load_dataset


def _save_image(args: Tuple[Image.Image, Path, str]) -> None:
    """Helper function to save an image in parallel.

    Args:
        args: Tuple of (image, output_dir, image_key)
    """
    try:
        image, output_dir, image_key = args
        output_path = output_dir / f"{image_key}.jpg"
        if output_path.exists():
            return
        image.save(output_path, format="JPEG", quality=95)
    except Exception as e:
        logger.error(f"Failed to save image {image_key}: {str(e)}")
        raise  # Re-raise to be caught by the executor


def setup_hf_dataset(
    output_dir: Path | str = "data",
    max_workers: int = 8,
    chunk_size: int = 1000,
) -> None:
    """Setup HuggingFace dataset by converting it to conversation format.

    Args:
        output_dir: Directory to save the processed dataset
        max_workers: Maximum number of parallel workers for image processing
        chunk_size: Number of examples to process in each chunk
    """
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    logger.info("Loading dataset from HuggingFace...")
    ds = load_dataset("SemanticExtraction/proper_dataset_v3_10k", split="train")

    # Process dataset in chunks to manage memory
    conversations = []
    total_chunks = (len(ds) + chunk_size - 1) // chunk_size
    total_processed = 0
    total_failed = 0

    for chunk_idx in range(total_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, len(ds))
        chunk = ds[start_idx:end_idx]

        # Process images in parallel
        logger.info(f"Processing images for chunk {chunk_idx + 1}/{total_chunks}")
        image_tasks = list(
            zip(chunk["image"], [images_dir] * (end_idx - start_idx), chunk["key"])
        )

        print(len(image_tasks))
        print(image_tasks[:10])

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            executor.map(_save_image, image_tasks)

        for key, ground_truth in zip(chunk["key"], chunk["ground_truth"]):
            # Skip failed images
            image_path = images_dir / f"{key}.jpg"
            if not image_path.exists():
                continue

            conversation = {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": str(image_path.relative_to(output_dir)),
                            },
                            {
                                "type": "text",
                                "text": "Extract the semantic information from this image.",
                            },
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": ground_truth}],
                    },
                ]
            }
            conversations.append(conversation)

        # Free up memory
        del chunk

    # Save conversations
    conversations_file = output_dir / "hf_conversations.json"
    logger.info(f"Saving {len(conversations)} conversations to {conversations_file}")
    conversations_file.write_text(
        json.dumps(conversations, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    logger.info(
        f"Dataset setup completed successfully! "
        f"Processed {total_processed}/{len(ds)} examples "
        f"({total_failed} failed)"
    )


if __name__ == "__main__":
    setup_hf_dataset()
