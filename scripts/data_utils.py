import json

from datasets import Dataset, DatasetDict  # type: ignore
from datasets import concatenate_datasets  # type: ignore


def load_data(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


# Function to calculate the number of tokens in a text
def count_tokens(tokenizer, text):
    """
    Calculate the number of tokens in a given text using the tokenizer.

    Parameters:
    text (str): The input text to be tokenized.

    Returns:
    int: The number of tokens in the input text.
    """
    return len(tokenizer.encode(text))


def split_train_examples(train_examples, max_size=4096 - 32):
    total_size = sum(
        len(example["input"]) * len(example["input"][0]) + len(example["output"]) * len(example["output"][0]) for example in train_examples
    )
    if total_size <= max_size:
        return [train_examples]

    split_size = max(1, max_size // total_size)
    return [train_examples[i : i + split_size] for i in range(0, len(train_examples), split_size)]


def to_dataset(data, solutions=None, fit_dataset=False):
    restructured_data = {
        "id": [],
        "challenge": [],
    }
    if solutions is not None:
        restructured_data["solution"] = []

    for challenge_id, challenge_data in data.items():  # for all challenges
        for test_id, task in enumerate(
            challenge_data["test"]
        ):  # for all test tasks in this challenge we want to expand dataset so that each test task is separate dataset record
            if fit_dataset:
                for split_id, split_train in enumerate(
                    split_train_examples(challenge_data["train"])
                ):  # if fit_dataset is true, we split each training example into multiple records so that each record has less than MAX_SEQ_LENGTH tokens
                    restructured_data["id"].append(challenge_id)
                    restructured_data["challenge"].append({"train": split_train, "test": task, "order": test_id})
                    if solutions is not None:
                        restructured_data["solution"].append(solutions[challenge_id][test_id])
            else:
                restructured_data["id"].append(challenge_id)
                restructured_data["challenge"].append({"train": challenge_data["train"], "test": task, "order": test_id})
                if solutions is not None:
                    restructured_data["solution"].append(solutions[challenge_id][test_id])

    return Dataset.from_dict(restructured_data)


def prepare_inputs(dct):
    input_str = "\n".join("".join(map(str, row)) for row in dct["input"])
    output_str = "\n".join("".join(map(str, row)) for row in dct["output"]) if "output" in dct else ""
    text = f"<input>\n{input_str}\n</input>\n\n<output>\n{output_str}\n</output>"
    return text


def prepare_dataset(tokenizer, use_system_prompt=False, fit_dataset=False, base_path=None, final_training=False):
    # The system_prompt defines the initial instructions for the model, setting the context for solving ARC tasks.
    system_prompt = (
        """You are a puzzle solving wizard. You are given a puzzle from the abstraction and reasoning corpus developed by Francois Chollet."""
    )

    # User message template is a template for creating user prompts. It includes placeholders for training data and test input data, guiding the model to learn the rule and apply it to solve the given puzzle.
    user_message_template = """Here are the example input and output pairs from which you should learn the underlying rule to later predict the output for the given test input:
-----------------
{training_data}
-----------------
Now, solve the following puzzle based on its input grid by applying the rules you have learned from the training data.:
-----------------
{input_test_data}
-----------------
What is the output grid? Only provide the output grid in the form as in the example input and output pairs. Do not provide any additional information:"""

    # Load all datasets
    training_challenges = load_data(f"{base_path}/arc-prize-2024/arc-agi_training_challenges.json")
    training_solutions = load_data(f"{base_path}/arc-prize-2024/arc-agi_training_solutions.json")
    evaluation_challenges = load_data(f"{base_path}/arc-prize-2024/arc-agi_evaluation_challenges.json")
    evaluation_solutions = load_data(f"{base_path}/arc-prize-2024/arc-agi_evaluation_solutions.json")
    test_challenges = load_data(f"{base_path}/arc-prize-2024/arc-agi_test_challenges.json")

    train_dataset = to_dataset(training_challenges, training_solutions, fit_dataset=fit_dataset)
    eval_dataset = to_dataset(evaluation_challenges, evaluation_solutions, fit_dataset=fit_dataset)
    pred_dataset = to_dataset(test_challenges, fit_dataset=fit_dataset)

    def create_chat(challenge, solution=None):
        user_content = user_message_template.format(
            training_data="\n\n".join([prepare_inputs(ex) for ex in challenge["train"]]),
            input_test_data=prepare_inputs(challenge["test"]),
        )

        if use_system_prompt:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]
        else:
            messages = [{"role": "user", "content": f"{system_prompt}\n\n{user_content}"}]

        if solution:
            messages.append(
                {
                    "role": "assistant",
                    "content": "<output>\n" + "\n".join("".join(map(str, row)) for row in solution) + "\n</output>",
                }
            )

        return messages

    def process_dataset(examples, solutions=None):
        # Create messages for each challenge-solution pair
        chats = []
        for challenge, solution in zip(examples["challenge"], solutions or [None] * len(examples["challenge"])):
            chat = create_chat(challenge, solution)
            chats.append(chat)

        # Apply chat template to each message
        texts = [tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False) for chat in chats]

        return {"texts": texts, "messages": chats}

    pred_dataset = pred_dataset.map(lambda x: process_dataset(x), batched=True)
    train_dataset = train_dataset.map(lambda x: process_dataset(x, train_dataset["solution"]), batched=True)
    eval_dataset = eval_dataset.map(lambda x: process_dataset(x, eval_dataset["solution"]), batched=True)

    if final_training:  # if final training, we need to add the validation dataset to the training dataset
        train_dataset = concatenate_datasets([train_dataset, eval_dataset])

        return DatasetDict(
            {
                "train": train_dataset,
                "predict": pred_dataset,
            }
        )

    train_dataset = train_dataset.map(lambda x: process_dataset(x, train_dataset["solution"]), batched=True)

    eval_dataset = eval_dataset.map(lambda x: process_dataset(x, eval_dataset["solution"]), batched=True)
    test_dataset = eval_dataset.train_test_split(test_size=0.3)

    dataset = DatasetDict(
        {
            "train": train_dataset,
            "test": test_dataset["train"],
            "val": test_dataset["test"],
            "predict": pred_dataset,
        }
    )

    return dataset
