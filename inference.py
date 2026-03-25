import argparse
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from sparsify import Sae
from pathlib import Path
import pickle
from torch.utils.data import DataLoader

def apply_chat_template(prompt, response, tokenizer):
    templated_example = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]
    templated_example = tokenizer.apply_chat_template(templated_example, tokenize=False)
    templated_example = templated_example[len(tokenizer.bos_token):]

    return templated_example

def run_inference(examples, model, tokenizer):
    dataloader = DataLoader(
        examples,
        batch_size=16,
        collate_fn=lambda batch: tokenizer(batch, return_tensors="pt", padding=True),
    )

    cached_activations = {}
    def score_hook(module, inputs, output):
        cached_activations["score_inputs"] = inputs[0]
    score_hook_handle = model.score.register_forward_hook(score_hook)

    activations_batches = []

    with torch.no_grad():
        for batch in dataloader:

            batch = {key: value.to("cuda:0") for key, value in batch.items()}
            model(**batch, return_dict=True)

            score_inputs = cached_activations["score_inputs"]
            example_indices = torch.arange(score_inputs.size(0), device="cuda:0")
            final_token_indices = batch["attention_mask"].sum(dim=1) - 1
            activations_batch = score_inputs[example_indices, final_token_indices, :].cpu()
            activations_batches.append(activations_batch)
            cached_activations["score_inputs"] = None
    
    score_hook_handle.remove()

    return torch.cat(activations_batches, dim=0)

def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--sae_directory", required=True, type=str)
    argument_parser.add_argument("--output_directory", required=True, type=str)
    arguments = argument_parser.parse_args()

    dataset = load_dataset("allenai/reward-bench", split="filtered")
    model = AutoModelForSequenceClassification.from_pretrained(
        "Skywork/Skywork-Reward-V2-Llama-3.1-8B",
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        num_labels=1,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("Skywork/Skywork-Reward-V2-Llama-3.1-8B")

    templated_chosen_examples = []
    templated_rejected_examples = []
    for example in dataset:
        templated_chosen_examples.append(apply_chat_template(example["prompt"], example["chosen"], tokenizer))
        templated_rejected_examples.append(apply_chat_template(example["prompt"], example["rejected"], tokenizer))

    chosen_activations = run_inference(templated_chosen_examples, model, tokenizer)
    rejected_activations = run_inference(templated_rejected_examples, model, tokenizer)

    sae = Sae.load_from_disk(arguments.sae_directory, device="cuda:0")

    chosen_features = sae.encode(chosen_activations.to("cuda:0"))
    chosen_feature_activations = chosen_features.top_acts.detach().cpu().tolist()
    chosen_feature_indices = chosen_features.top_indices.cpu().tolist()

    rejected_features = sae.encode(rejected_activations.to("cuda:0"))
    rejected_feature_activations = rejected_features.top_acts.detach().cpu().tolist()
    rejected_feature_indices = rejected_features.top_indices.cpu().tolist()

    data = {
        "chosen_feature_activations": chosen_feature_activations,
        "chosen_feature_indices": chosen_feature_indices,
        "rejected_feature_activations": rejected_feature_activations,
        "rejected_feature_indices": rejected_feature_indices,
    }
    sae_directory = Path(arguments.sae_directory)
    output_directory = Path(arguments.output_directory) / sae_directory.parts[-6] / sae_directory.parts[-5]
    output_directory.mkdir(parents=True, exist_ok=True)
    with open(output_directory / f"{sae_directory.parts[-4]}.pkl", "wb") as f:
        pickle.dump(data, f)

if __name__ == "__main__":
    main()