import torch
import torch.nn as nn

class VAE_model(nn.Module):
  def __init__(self, input_dim, hidden_dims, decode_dim=-1):
      '''
      input_dim: The dimensionality of the input data.
      hidden_dims: A list of hidden dimensions for the layers of the encoder and decoder.
      decode_dim: (Optional) Specifies the dimensions to decode, if different from input_dim. Keep them same for this HW.
      '''
      super().__init__()
      
      if decode_dim == -1:
            decode_dim = input_dim
      self.decode_dim = decode_dim
      # Latent dimension (z_size) = half of final hidden size
      self.z_size = hidden_dims[-1] // 2
    
      self.encoder = nn.Sequential(
          nn.Linear(input_dim, hidden_dims[0]),
          nn.LeakyReLU(0.2, inplace=True),
          nn.Linear(hidden_dims[0], hidden_dims[1]),
          nn.LeakyReLU(0.2, inplace=True),
          nn.Linear(hidden_dims[1], hidden_dims[2]),
          nn.LeakyReLU(0.2, inplace=True),
          nn.Linear(hidden_dims[2], 2 * self.z_size)
      
      )
      
      self.decoder = nn.Sequential(
 
          nn.Linear(self.z_size, hidden_dims[2]),
          nn.LeakyReLU(0.2, inplace=True),
          nn.Linear(hidden_dims[2], hidden_dims[1]),
          nn.LeakyReLU(0.2, inplace=True),
          nn.Linear(hidden_dims[1], hidden_dims[0]),
          nn.LeakyReLU(0.2, inplace=True),
          nn.Linear(hidden_dims[0], self.decode_dim),
          nn.Sigmoid()
      
      )

  def encode(self, x):
      mean, logvar = torch.split(self.encoder(x), split_size_or_sections=[self.z_size, self.z_size], dim=-1)
      return mean, logvar

  def reparameterize(self, mean, logvar):
      std = torch.exp(0.5 * logvar)
      eps = torch.randn_like(std)
      return mean + eps * std

  def decode(self, z):
      probs = self.decoder(z)
      return probs

  def forward(self, x):
      mean, logvar = self.encode(x)
      z = self.reparameterize(mean, logvar)
      x_probs = self.decode(z)
      return {"imgs": x_probs, "z": z, "mean": mean, "logvar": logvar}
