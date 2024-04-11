import torch
import esm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TransferLearningModel(torch.nn.Module):
    def __init__(self, size: str = "big"):
        super(TransferLearningModel, self).__init__()
        self.size = size
        if self.size == "big":
            self.encoder, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            self.regressor = torch.nn.Linear(1280, 1).to(device)
        elif self.size == "small":
            self.encoder, self.alphabet = esm.pretrained.esm2_t30_150M_UR50D()
            self.regressor = torch.nn.Linear(640, 1).to(device)
        self.encoder = self.encoder.to(device)

    def forward(self, x):
        x = x.to(device)
        if self.size == "big":
            results = self.encoder(x, repr_layers=[33], return_contacts=True)
            token_repr = results["representations"][33].mean([1]) 
        elif self.size == "small":
            results = self.encoder(x, repr_layers=[30], return_contacts=True)
            token_repr = results["representations"][30].mean([1]) 
        x = self.regressor(token_repr)
        return x

    def get_embedding(self, x):
        return self.encoder(x)

    def get_alphabet(self, data):
        batch_converter = self.alphabet.get_batch_converter()
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)
        return (batch_labels, batch_tokens)
