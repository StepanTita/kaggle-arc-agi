import json

from datasets import Dataset, DatasetDict  # type: ignore
from datasets import concatenate_datasets  # type: ignore

from . import data_utils

MODEL_REPLY = """assistant<output>
{output}
</output>
"""

USER_CORRECTION = """user<message>
{message}
</message>

Now, solve the following puzzle based on its input grid by applying the rules you have learned from the training data and the error corrections you have received.
Only provide the output grid in the form as in the example input and output pairs. Do not provide any additional information:
-----------------
{input_test_data}
"""


def create_chat(challenge, solution=None):
    user_content = data_utils.BASIC_PROMPT.format(
        training_data="\n\n".join([data_utils.prepare_inputs(ex) for ex in challenge["train"]]),
        input_test_data=data_utils.prepare_inputs(challenge["test"]),
    )

    messages = [
        {"role": "system", "content": data_utils.SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    if solution:
        messages.append(
            {
                "role": "assistant",
                "content": data_utils.prepare_inputs(solution, prepare_solution=True),
            }
        )

    return messages


def to_augmentation_dataset(data, solutions=None):
    restructured_data = {
        "id": [],
        "challenge": [],
        "solution": [],
    }

    for challenge_id, challenge_data in data.items():  # for all challenges
        for train_id, train_task in enumerate(challenge_data["train"]):
            restructured_data["id"].append(challenge_id)
            restructured_data["challenge"].append(
                {
                    "train": challenge_data["train"][:train_id] + challenge_data["train"][train_id + 1 :],
                    "test": {"input": train_task["input"]},
                    "order": train_id,
                }
            )
            restructured_data["solution"].append(train_task["output"])

    return Dataset.from_dict(restructured_data)


def prepare_dataset(base_path=None, final_training=False):
    # Load all datasets
    training_challenges = data_utils.load_data(f"{base_path}/arc-prize-2024/arc-agi_training_challenges.json")
    training_solutions = data_utils.load_data(f"{base_path}/arc-prize-2024/arc-agi_training_solutions.json")
    evaluation_challenges = data_utils.load_data(f"{base_path}/arc-prize-2024/arc-agi_evaluation_challenges.json")
    evaluation_solutions = data_utils.load_data(f"{base_path}/arc-prize-2024/arc-agi_evaluation_solutions.json")
    test_challenges = data_utils.load_data(f"{base_path}/arc-prize-2024/arc-agi_test_challenges.json")

    train_dataset = to_augmentation_dataset(training_challenges)
    eval_dataset = to_augmentation_dataset(evaluation_challenges)
    pred_dataset = to_augmentation_dataset(test_challenges)

    if final_training:  # if final training, we need to add the validation dataset to the training dataset
        train_dataset = concatenate_datasets([train_dataset, eval_dataset]).shuffle(seed=42)
        return DatasetDict(
            {
                "train": train_dataset,
                "predict": pred_dataset,
            }
        )

    dataset = DatasetDict(
        {
            "train": train_dataset,
            "test": eval_dataset,
            "predict": pred_dataset,
        }
    )

    return dataset


def build_augmented_dataset(dataset, train_corrections, test_corrections):
    train_augmented_dataset = []
    test_augmented_dataset = []
    for train_instance in dataset["train"]:
        challenge = train_instance["challenge"].copy()
        challenge_id = train_instance["id"]
        order = challenge["order"]

        correction = train_corrections[challenge_id][str(order)]

        challenge["correction"] = {}
        challenge["correction"]["output"] = correction[0]
        challenge["correction"]["message"] = correction[1]

        train_augmented_dataset.append({"id": challenge_id, "challenge": challenge, "solution": train_instance["solution"]})

    for test_instance in dataset["test"]:
        challenge = test_instance["challenge"].copy()
        challenge_id = test_instance["id"]
        order = challenge["order"]

        correction = test_corrections[challenge_id][str(order)]

        challenge["correction"] = {}
        challenge["correction"]["output"] = correction[0]
        challenge["correction"]["message"] = correction[1]

        test_augmented_dataset.append({"id": challenge_id, "challenge": challenge, "solution": test_instance["solution"]})

    return DatasetDict({"train": Dataset.from_list(train_augmented_dataset), "test": Dataset.from_list(test_augmented_dataset)})


def prepare_corrections(correction):
    if correction is None:
        return "None"
    padded_rows = [" ".join(map(str, row)) for row in correction]
    return "\n".join(padded_rows)


def create_chat_with_correction(challenge, solution):
    initial_input = data_utils.BASIC_PROMPT.format(
        training_data="\n\n".join([data_utils.prepare_inputs(ex) for ex in challenge["train"]]),
        input_test_data=data_utils.prepare_inputs({"input": challenge["test"]["input"]}),
    )

    assistant_reply = MODEL_REPLY.format(output=prepare_corrections(challenge["correction"]["output"]))

    user_correction = USER_CORRECTION.format(
        message=challenge["correction"]["message"],
        input_test_data=data_utils.prepare_inputs(challenge["test"]),
    )

    assistant_final_reply = MODEL_REPLY.format(output=prepare_corrections(solution))

    messages = [
        {"role": "system", "content": data_utils.SYSTEM_PROMPT},
        {"role": "user", "content": initial_input},  # intial input
        {"role": "assistant", "content": assistant_reply},  # assistant response
        {"role": "user", "content": user_correction},  # user feedback + correction
    ]

    if solution is not None:
        messages.append({"role": "assistant", "content": assistant_final_reply})

    return messages


def prepare_augmented_dataset(tokenizer, dataset, base_path=None):
    # Load all datasets
    train_corrections = data_utils.load_data(f"{base_path}/data/train_corrections.json")
    test_corrections = data_utils.load_data(f"{base_path}/data/test_corrections.json")

    augmented_dataset = build_augmented_dataset(dataset, train_corrections, test_corrections)

    def process_dataset(examples, solutions=None):
        # Create messages for each challenge-solution pair
        chats = []
        for challenge, solution in zip(examples["challenge"], solutions or [None] * len(examples["challenge"])):
            chat = create_chat_with_correction(challenge, solution)
            chats.append(chat)

        # Apply chat template to each message
        texts = [tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False) for chat in chats]

        return {"texts": texts, "messages": chats}

    train_dataset = augmented_dataset["train"].map(lambda x: process_dataset(x, solutions=x["solution"]), batched=True)
    test_dataset = augmented_dataset["test"].map(lambda x: process_dataset(x, solutions=x["solution"]), batched=True)

    return DatasetDict(
        {
            "train": train_dataset,
            "test": test_dataset,
        }
    )
