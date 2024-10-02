import torch  # type: ignore
import re


def gpu_stats(device_id=0):
    # @title Show current memory stats
    gpu_stats = torch.cuda.get_device_properties(device_id)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    return {
        "gpu": gpu_stats.name,
        "max_memory": max_memory,
        "start_gpu_memory": start_gpu_memory,
    }


def parse_output(text):
    # Extract the content inside <output></output> tags
    output_match = re.search(r"<output>(.*?)</output>", text, re.DOTALL)
    if not output_match:
        return None

    output_content = output_match.group(1).strip()

    # Split the content into lines and convert each line to a list of single-digit integers
    try:
        grid = []
        for line in output_content.split("\n"):
            row = [int(char) for char in line.strip() if char.isdigit()]
            if row:
                grid.append(row)

        # Ensure all rows have the same length
        if grid and all(len(row) == len(grid[0]) for row in grid):
            return grid
        else:
            return None
    except ValueError:
        return None


def tensor_to_int(value):
    if isinstance(value, torch.Tensor):
        return tensor_to_int(value.item())
    elif isinstance(value, list):
        return [tensor_to_int(item) for item in value]
    else:
        return value


def calculate_partial_match(pred, label):
    if not isinstance(pred, list) or not isinstance(label, list):
        return 0  # No match if either is not a list

    if len(pred) != len(label):
        return 0  # No match if outer dimensions differ

    total_elements = 0
    correct_elements = 0

    for p_row, l_row in zip(pred, label):
        if not isinstance(p_row, list) or not isinstance(l_row, list) or len(p_row) != len(l_row):
            return 0  # No match if any row is not a list or dimensions differ

        total_elements += len(l_row)
        correct_elements += sum(p == l for p, l in zip(p_row, l_row))

    return correct_elements / total_elements if total_elements > 0 else 0


def calculate_metrics(preds, labels):
    total_samples = len(labels)

    correct = sum(1 for p, l in zip(preds, labels) if p == l)
    accuracy = correct / total_samples

    partial_match_scores = [calculate_partial_match(p, l) if p is not None else 0 for p, l in zip(preds, labels)]

    avg_partial_match = sum(partial_match_scores) / total_samples

    return accuracy, avg_partial_match


def collate(mode, tokenizer):
    def collate_fn(batch):
        # Separate the different components of the batch
        ids = [item["id"] for item in batch]
        challenges = [item["challenge"] for item in batch]

        # For 'test' mode, remove the last assistant message from each entry
        if mode == "test":
            messages = [
                item["messages"][:-1] for item in batch
            ]  # last message is always assistant message - solution, we don't need it for evaluation
        else:
            messages = [item["messages"] for item in batch]

        # Tokenize the texts
        encodings = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            padding=True,
            # truncation=True
        )

        # If 'solution' is present (for training/validation data)
        if "solution" in batch[0]:
            solutions = [item["solution"] for item in batch]
            return {
                "id": ids,
                "challenge": challenges,
                "solution": solutions,
                "input_ids": encodings["input_ids"].to("cuda"),
                "attention_mask": encodings["attention_mask"].to("cuda"),
            }
        else:
            return {
                "id": ids,
                "challenge": challenges,
                "input_ids": encodings["input_ids"].to("cuda"),
                "attention_mask": encodings["attention_mask"].to("cuda"),
            }

    return collate_fn

