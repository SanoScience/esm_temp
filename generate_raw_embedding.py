import pandas as pd
import esm
import torch
from typing import List
import argparse


def generate_embedding_as_csv(dataset: List[tuple],
                              filename: str,
                              size_embedding: int = 1280):
    df_data = []
    n = len(dataset)
    results = torch.tensor((n, size_embedding + 1))

    with torch.no_grad():
        for i, data in enumerate(dataset):
            batch_labels, batch_strs, batch_tokens_t = batch_converter([data])
            row = {}
            results_t = model(batch_tokens_t, repr_layers=[layer], return_contacts=True)
            token_representations_t = results_t["representations"][layer]
            embedding = token_representations_t.mean(0)
            results[i, 0] = data[0]
            results[i, 1:] = embedding
    
    labels_t = results[:, 0]
    embeddings_t = results[:, 1:]

    labels_np = labels_t.numpy()
    embeddings_np = embeddings_t.numpy()

    data = {'embedding': embeddings_np, 'labels': labels_np}

    df_train = pd.DataFrame(data)
    df_train.to_csv(filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script with a parameter to choose model size.")

    parser.add_argument("--model_size", type=str, default="big", help="Specify the size of model.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Specify the path to the CSV file.")
    parser.add_argument("--path_embedding_file", type=str, required=True, help="Specify the name for the embedding file.")

    args = parser.parse_args()
    
    model_size = args.model_size

    if model_size == "big":
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        batch_converter = alphabet.get_batch_converter()
        layer = 33
    elif model_size == "small":
        model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
        batch_converter = alphabet.get_batch_converter()
        layer = 30

    model.eval()
    
    df = pd.read_csv(args.dataset_path)
    df_data = [(row["tm"], row["protein_sequence"]) for _, row in df.iterrows()]

    generate_embedding_as_csv(dataset=df_data,  
                              filename=args.path_embedding_file)

