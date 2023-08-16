from datasets import load_dataset

dataset = load_dataset("snli")

ids = list(range(0, 10000))
dataset["test"] = dataset["test"].add_column("id", ids)
