import torch

class NNAutoencoder(torch.nn.Module):
    def __init__(self, in_features, latent_dim, drop = 0.3):
        super().__init__()
        self.in_features = in_features
        self.latent_dim = latent_dim
        self.drop = drop

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(in_features= self.in_features, out_features= 128), 
            torch.nn.ReLU(),
            torch.nn.Dropout(self.drop),
            torch.nn.Linear(in_features= 128, out_features= 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.drop),
            torch.nn.Linear(in_features= 64, out_features= 32),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.drop),
            torch.nn.Linear(in_features= 32, out_features= self.latent_dim)
        )

        #contenedor decoder
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(in_features= self.latent_dim, out_features= 32),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features= 32, out_features= 64),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features= 64, out_features= 128),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features= 128, out_features= self.in_features)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
