import argparse
import torch
import torch.nn as nn
from sparsify import Sae
from transformers import AutoModelForSequenceClassification

def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--sae_directory", required=True, type=str)
    argument_parser.add_argument("--device", required=True, choices=["cpu", "cuda"])
    arguments = argument_parser.parse_args()

    sae = Sae.load_from_disk(arguments.sae_directory, device=arguments.device, decoder=True)
    w_dec = sae.W_dec

    model = AutoModelForSequenceClassification.from_pretrained(
        "Skywork/Skywork-Reward-V2-Llama-3.1-8B",
        torch_dtype=torch.bfloat16,
        device_map=arguments.device,
        num_labels=1,
    )
    score = model.score.weight.T.to(dtype=sae.W_dec.dtype)

    feature_influence = (w_dec @ score).squeeze()
    print(feature_influence)

    sorted_feature_ids = torch.argsort(feature_influence)
    negative_feature_ids = [feature_id for feature_id in sorted_feature_ids if feature_influence[feature_id] < 0]
    positive_feature_ids = [feature_id for feature_id in sorted_feature_ids if feature_influence[feature_id] > 0]

    print("\nNegative features:")
    for feature_id in negative_feature_ids:
        print(f"Index {feature_id.item()}: {feature_influence[feature_id].item()}")

    print("\nPositive features:")
    for feature_id in reversed(positive_feature_ids):
        print(f"Index {feature_id.item()}: {feature_influence[feature_id].item()}")


if __name__ == "__main__":
    main()