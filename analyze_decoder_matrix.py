import argparse
from sparsify import Sae
import torch
import torch.nn.functional as F

def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--sae_directory", required=True, type=str)
    argument_parser.add_argument("--device", required=True, choices=["cpu", "cuda"])
    arguments = argument_parser.parse_args()

    sae = Sae.load_from_disk(arguments.sae_directory, device=arguments.device, decoder=True)
    w_dec = sae.W_dec.float()

    rank = torch.linalg.matrix_rank(w_dec).item()
    print(f"Decoder matrix rank: {rank}")

    w_dec_normalized = F.normalize(w_dec, p=2, dim=1)
    w_dec_cosine_similarity = w_dec_normalized @ w_dec_normalized.T
    n = w_dec_cosine_similarity.shape[0]
    rows, columns = torch.triu_indices(n, n, offset=1)
    w_dec_cosine_similarity_unique = w_dec_cosine_similarity[rows, columns]
    w_dec_cosine_similarity_unique_magnitudes = w_dec_cosine_similarity_unique.abs()
    topk_w_dec_cosine_similarity_unique_magnitudes = torch.topk(w_dec_cosine_similarity_unique_magnitudes, k=10, largest=True)
    for i in topk_w_dec_cosine_similarity_unique_magnitudes.indices:
        print(f"({rows[i].item()}, {columns[i].item()})  value={w_dec_cosine_similarity_unique[i].item():.6f}")

if __name__ == "__main__":
    main()