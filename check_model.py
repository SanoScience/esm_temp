import torch
import torch.nn as nn
import logging
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import esm
from sklearn.metrics import r2_score
import argparse
from models.transfer_learning import TransferLearningModel
from data.batch_converter import BatchConverterProteinDataset
from sklearn.preprocessing import MinMaxScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process files with a specified index.')
    parser.add_argument('--test_dataset_path', type=str, help='Path to csv test file', required=True)
    parser.add_argument('--train_df_path', type=str, help='Path to csv test file', required=True)
    parser.add_argument("--size", type=str, choices=["big", "small"], default="big", 
                        help="Specify the size parameter (default: big)")
    parser.add_argument('--checkpoint_path', type=str, help='Path to model checkpoint file', required=True)
    parser.add_argument('--result_file_name', type=str, help='Path of output csv file', required=True)
    
    args = parser.parse_args()
    
    fold_index = args.file_index
    df_test = pd.read_csv(args.test_dataset_path)
    
    test_data = [(row["tm"], row["protein_sequence"]) for _, row in df_test.iterrows()]
    
    size = args.size
    num_unfrozen_layers = str(args.num_unfrozen_layers)
    model = TransferLearningModel(size=size)
    learning_rate = 0.001
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate)

    checkpoint = torch.load(args.checkpoint_path, map_location=device)  
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    mse_loss = torch.nn.MSELoss()
    model.to(device)
    
    predictions = []
    temperatures = []
    sequences = []

    model.eval()
    with torch.no_grad():
        for data in test_data:
            _, string_seq = data
            batch_labels, batch_tokens = model.get_alphabet(data=[data])
            batch_converter_dataset = BatchConverterProteinDataset(batch_labels, batch_tokens)
            val_dataset = [batch_converter_dataset[0]]
            val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True)
            for sequence, temperature in val_dataloader:
                sequence = sequence.to(device)
                temperature = temperature.to(device)
                output = model(sequence)
                output = output.squeeze()
                prediction = output.detach().cpu().numpy()
                predictions.append(prediction)
                temperature = temperature.squeeze()
                label = temperature.detach().cpu().numpy()
                temperatures.append(label)
                sequences.append(string_seq)
                

    df_result = pd.DataFrame({"pred": predictions,
                            "labels": temperatures,
                            "protein_sequence": sequences})
    
    scaler = MinMaxScaler()
    
    train_df = pd.read_csv(args.train_df_path)
    tm_values = train_df["tm"].values.reshape(-1, 1)
    scaler.fit_transform(tm_values)

    scaled_values = scaler.inverse_transform(df_result['pred'].values.reshape(-1, 1))
    df_result['scaled_column'] = scaled_values
    
    df_result.to_csv(args.result_file_name)