import torch
import torch.nn as nn
from torch.autograd import Variable

class Discriminator(nn.Module):
    def __init__(self, channels = [256, 128, 64]):
        super().__init__()
        

        self.model = nn.Sequential(
            nn.Linear(28*28, channels[0]),
            nn.LeakyReLU(0.25),
            nn.Dropout(0.3),

            nn.Linear(channels[0], channels[1]),
            nn.LeakyReLU(0.25),
            nn.Dropout(0.3),

            nn.Linear(channels[1], channels[2]),
            nn.LeakyReLU(0.25),
            nn.Dropout(0.3),

            nn.Linear(channels[2], 1),
            nn.Identity()  
        )

    def forward(self, x):
        x = x.view(x.size(0), 784)
        out = self.model(x)
        out = out.view(out.size(0), -1)
        return out

class Generator(nn.Module):
    def __init__(self, dim_z=100, channels = [64, 128, 256]):
        super().__init__()
        self.dim_z = dim_z
        self.model = nn.Sequential(
            # TODO: fill in layers here
            nn.Linear(self.dim_z, channels[0]),
            nn.LeakyReLU(0.25),

            nn.Linear(channels[0], channels[1]),
            nn.LeakyReLU(0.25),
        
            nn.Linear(channels[1], channels[2]),
            nn.LeakyReLU(0.25),

            nn.Linear(channels[2], 784),
            nn.Tanh(),

            nn.Identity()  
        )

    def forward(self, x):
        x = x.view(x.size(0), self.dim_z)
        out = self.model(x)
        return out


def train_discriminator(discriminator, d_optimizer, images, fake_images, criterion, device):
    
    discriminator.zero_grad()
    outputs_real = discriminator(images)

    real_labels = torch.ones(images.size(0), 1).to(device)
    fake_labels = torch.zeros(fake_images.size(0), 1).to(device)


    real_loss = criterion(outputs_real, real_labels)

    outputs_fake = discriminator(fake_images.detach())


    fake_loss = criterion(outputs_fake, fake_labels)

  
    d_loss = real_loss + fake_loss
    
    d_loss.backward()
    d_optimizer.step()

    return d_loss, outputs_real, outputs_fake, fake_loss

def train_generator(generator, g_optimizer, discriminator_outputs, criterion, device):
    
    generator.zero_grad()
    real_labels =  torch.ones(discriminator_outputs.size(0), 1).to(device)

    g_loss = criterion(discriminator_outputs, real_labels)
    g_loss_fake = criterion(discriminator_outputs, 1 - real_labels.view(-1, 1))
    g_loss.backward()
    g_optimizer.step()

    return g_loss, g_loss_fake
