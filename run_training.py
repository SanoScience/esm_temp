import torch
import logging
import pandas as pd
import os
import random
from models.transfer_learning import TransferLearningModel
from data.batch_converter import BatchConverterProteinDataset
import wandb
import argparse

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser(description="Check a machine learning model with a specific "
                                                 "dataset and model type.")
    
    parser.add_argument(
        "--train_csv_file",
        type=str,
        required=True,
        help="Choose path to train file"
    )
    
    parser.add_argument(
        "--val_csv_file",
        type=str,
        required=True,
        help="Choose path to val file"
    )
    
    parser.add_argument(
        "--size",
        type=str,
        choices=["big", "small"],
        default="big",
        help="Specify the size parameter (default: big)"
    )
    
    parser.add_argument(
        "--num_unfrozen_layers",
        type=int,
        choices=[0, 1, 2, 3],
        default=0,
        help="Specify the number of unfrozen layers (default: 0)"
    )
    
    parser.add_argument(
        "--num_of_epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)"
    )
    
    parser.add_argument(
        "--random_subset_size",
        type=int,
        default=0,
        help="Size of the random subset used for each iteration "
             "(default: 0, using the full dataset)"
    )
    
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        default=False,
        help="Log into WandB"
    )
    
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Choose the directory for saving checkpoints"
    )
    
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    train_df = pd.read_csv(args.train_csv_file)
    val_df = pd.read_csv(args.val_csv_file)

    train_data = [(row["scale_tm"], row["protein_sequence"]) for _, row in train_df.iterrows()]
    val_data = [(row["scale_tm"], row["protein_sequence"]) for _, row in val_df.iterrows()]

    size = args.size
    num_unfrozen_layers = str(args.num_unfrozen_layers)
    model = TransferLearningModel(size=size)
    learning_rate = 0.001
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate)

    for name, param in model.named_parameters():
        if num_unfrozen_layers == "0":
            if "regressor" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        elif num_unfrozen_layers == "1":
            if("regressor" in name
                    or "contact_head" in name
                    or "emb_layer_norm_after" in name
                    or "lm_head" in name):
                param.requires_grad = True
            else:
                param.requires_grad = False
        elif num_unfrozen_layers == "2":
            if (
                "layers.32" in name
                or "regressor" in name
                or "contact_head" in name
                or "emb_layer_norm_after" in name
                or "lm_head" in name
            ):
                param.requires_grad = True
            else:
                param.requires_grad = False
        elif num_unfrozen_layers == "3":
            if (
                "layers.31" in name
                or "layers.32" in name
                or "regressor" in name
                or "contact_head" in name
                or "emb_layer_norm_after" in name
                or "lm_head" in name
            ):
                param.requires_grad = True
            else:
                param.requires_grad = False
    
    model.to(device)
    mse_loss = torch.nn.MSELoss()
    
    if args.use_wandb:
        wandb.init(project="project_name", entity="user_name")

    lowest_loss = float('inf')
    for epoch in range(args.num_of_epochs):
        total_loss_train = 0.0
        total_samples_train = 0
        model.train()
        current_sample = 0
        real_batch_size = 16
            
        train_data_sample = random.sample(train_data, args.random_subset_size) if args.random_subset_size != 0 else train_data

        for data in train_data_sample:
            batch_labels, batch_tokens = model.get_alphabet(data=[data])
            batch_converter_dataset = BatchConverterProteinDataset(batch_labels, batch_tokens)
            train_dataset = [batch_converter_dataset[0]]
            train_dataloader = torch.utils.data.DataLoader(
                train_dataset, batch_size=1, shuffle=True)

            for sequence, temperature in train_dataloader:
                sequence = sequence.to(device)
                temperature = temperature.to(device)
                output = model(sequence)
                loss = mse_loss(output, temperature)
                loss.backward()

                total_loss_train += loss.item()
                total_samples_train += 1
                current_sample += 1
                if current_sample >= real_batch_size:
                    current_sample = 0
                    optimizer.step()
                    optimizer.zero_grad()

        total_loss_val = 0.0
        total_samples_val = 0
        model.eval()
        with torch.no_grad():
            for data in val_data:
                batch_labels, batch_tokens = model.get_alphabet(data=[data])
                batch_converter_dataset = BatchConverterProteinDataset(batch_labels, batch_tokens)
                val_dataset = [batch_converter_dataset[0]]
                val_dataloader = torch.utils.data.DataLoader(
                    val_dataset, batch_size=1, shuffle=True)
                for sequence, temperature in val_dataloader:
                    sequence = sequence.to(device)
                    temperature = temperature.to(device)
                    output = model(sequence)
                    loss = mse_loss(output, temperature)
                    total_loss_val += loss.item()
                    total_samples_val += 1

        mean_loss_train = total_loss_train / total_samples_train

        mean_loss_val = total_loss_val / total_samples_val

        if args.use_wandb:
            wandb.log({"epoch": epoch, "val_loss": mean_loss_val})
            wandb.log({"epoch": epoch, "train_loss": mean_loss_train})

        logging.info(
            f"epoch: {epoch}, train loss: {mean_loss_train} val_loss: {mean_loss_val}")

        if epoch > 1 and epoch % 10 == 0:
            lowest_loss = loss
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            checkpoint_name = os.path.join(args.checkpoint_dir, "checkpoint.pth")
            torch.save(checkpoint, checkpoint_name)
