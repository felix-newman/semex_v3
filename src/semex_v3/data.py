from pathlib import Path
from typing import List, Tuple, Dict, Any
import json
import torch
from torch.utils.data import Dataset
from qwen_vl_utils import process_vision_info


class ToyDataSet(Dataset):
    """Dataset for fine-tuning Qwen-VL model."""

    def __init__(self, data_path: Path | str) -> None:
        """Initialize dataset from JSON file.

        Args:
            data_path: Path to the JSON data file
        """
        super().__init__()
        data_path = Path(data_path)
        self.data = json.loads(data_path.read_text())

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, List[Dict[str, Any]]]:
        return self.data[idx]


def find_assistant_content_sublist_indexes(
    input_ids: List[int],
) -> List[Tuple[int, int]]:
    """Find start and end indexes of assistant content in tokenized input.

    The function looks for special tokens that mark the beginning and end of assistant responses
    in the tokenized input. This is used to create labels for training, where we want the model
    to predict only the assistant's responses.

    Args:
        input_ids: List of token IDs from the tokenizer

    Returns:
        List of tuples containing (start_idx, end_idx) for each assistant response
    """
    start_indexes = []
    end_indexes = []

    # Special token IDs for assistant content
    ASSISTANT_START = [151644, 77091, 198]  # <|im_start|>assistant\n
    ASSISTANT_END = [151645, 198]  # <|im_end|>\n

    # Find all assistant response spans
    for i in range(len(input_ids) - 2):
        # Check for assistant start sequence
        if input_ids[i : i + 3] == ASSISTANT_START:
            start_indexes.append(i + 3)
            # Find corresponding end sequence
            for j in range(i + 3, len(input_ids) - 1):
                if input_ids[j : j + 2] == ASSISTANT_END:
                    # Include end tokens in the label
                    end_indexes.append(j + 2)
                    break

    return list(zip(start_indexes, end_indexes))


def collate_fn(
    batch: List[Dict], processor: Any, device: torch.device
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """Collate function for DataLoader.

    Processes a batch of examples into model inputs and labels. This includes:
    1. Extracting messages from the batch
    2. Applying chat template
    3. Processing vision information
    4. Creating attention masks and labels

    Args:
        batch: List of examples from the dataset
        processor: Qwen processor for tokenization and image processing
        device: Device to place tensors on

    Returns:
        Tuple of (model_inputs, labels)
    """
    messages = [m["messages"] for m in batch]
    texts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
        for msg in messages
    ]

    # Process images and videos if present
    image_inputs, video_inputs = process_vision_info(messages)

    # Create model inputs
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    # Create labels for assistant responses
    input_ids_lists = inputs["input_ids"].tolist()
    assert len(messages) == len(input_ids_lists)

    labels_list = []
    for ids_list in input_ids_lists:
        # Initialize with ignore index
        label_ids = [-100] * len(ids_list)
        # Fill in assistant response spans
        for start_idx, end_idx in find_assistant_content_sublist_indexes(ids_list):
            label_ids[start_idx:end_idx] = ids_list[start_idx:end_idx]
        labels_list.append(label_ids)

    labels = torch.tensor(labels_list, dtype=torch.int64)
    return inputs, labels
