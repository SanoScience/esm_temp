import torch


class BatchConverterProteinDataset(torch.utils.data.Dataset):
    def __init__(self, labels, tokens):
        self.labels = labels
        self.tokens = tokens

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        token = self.tokens[idx]
        label = self.labels[idx]
        label_float = float(label)
        return token, torch.tensor([label_float], dtype=torch.float)