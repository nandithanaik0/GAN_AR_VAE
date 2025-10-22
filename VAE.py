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
      
      ############################
      # TODO: Implement the encoder
      # - Use three fully connected hidden layers
      # - Apply LeakyReLU activations after each hidden layer
      # - Final layer should output 2 * latent_dim units 
      #   (concatenated mean and log-variance, no activation after the final layer)
      ############################
      self.encoder = nn.Sequential(
          # TODO: fill in layers here
          nn.Linear(input_dim, hidden_dims[0]),
          nn.LeakyReLU(0.2, inplace=True),
          nn.Linear(hidden_dims[0], hidden_dims[1]),
          nn.LeakyReLU(0.2, inplace=True),
          nn.Linear(hidden_dims[1], hidden_dims[2]),
          nn.LeakyReLU(0.2, inplace=True),
          nn.Linear(hidden_dims[2], 2 * self.z_size)
        #   nn.Identity()  # placeholder
      )
      
      ############################
      # TODO: Implement the decoder
      # - Mirror the encoder architecture with three hidden layers
      # - Apply LeakyReLU activations after each hidden layer
      # - Final layer should output 'decode_dim' units
      # - Apply Sigmoid activation only on the last layer
      ############################
      self.decoder = nn.Sequential(
          # TODO: fill in layers here
          nn.Linear(self.z_size, hidden_dims[2]),
          nn.LeakyReLU(0.2, inplace=True),
          nn.Linear(hidden_dims[2], hidden_dims[1]),
          nn.LeakyReLU(0.2, inplace=True),
          nn.Linear(hidden_dims[1], hidden_dims[0]),
          nn.LeakyReLU(0.2, inplace=True),
          nn.Linear(hidden_dims[0], self.decode_dim),
          nn.Sigmoid()
        #   nn.Identity()  # placeholder
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