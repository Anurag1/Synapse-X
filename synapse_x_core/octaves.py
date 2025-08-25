import torch
import torch.nn as nn

class SpecialistOctave(nn.Module):
    """
    A generic, MLP-based specialist for demonstrating on-the-fly training.
    In a real system, different tasks might use different architectures (Transformers, CNNs).
    """
    def __init__(self, input_dim, output_dim, z_dim=32):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Simple encoder
        self.encoder_fc1 = nn.Linear(input_dim, 128)
        self.encoder_fc2 = nn.Linear(128, 64)
        self.fc_mu = nn.Linear(64, z_dim)
        self.fc_logvar = nn.Linear(64, z_dim)
        
        # Simple decoder
        self.decoder_fc1 = nn.Linear(z_dim, 64)
        self.decoder_fc2 = nn.Linear(64, 128)
        self.decoder_fc3 = nn.Linear(128, output_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def encode(self, x):
        h = F.relu(self.encoder_fc1(x))
        h = F.relu(self.encoder_fc2(h))
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z):
        h = F.relu(self.decoder_fc1(z))
        h = F.relu(self.decoder_fc2(h))
        return self.decoder_fc3(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    @torch.no_grad()
    def execute(self, x):
        """For inference, we pass the input through the encoder's mu value to the decoder."""
        mu, _ = self.encode(x)
        return self.decode(mu)