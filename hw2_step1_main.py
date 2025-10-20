import torch
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.autograd import Variable
import numpy as np
import itertools
import math
from tqdm import tqdm

import torch.nn as nn

from models.GAN import Generator, Discriminator, train_generator, train_discriminator 

def monitor_images(generator, test_noise):
    num_test_samples = test_noise.shape[0]
    size_figure_grid = int(math.sqrt(num_test_samples))
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(6, 6))

    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    test_images = generator(test_noise).detach().cpu()

    for k in range(num_test_samples):
        i = k // size_figure_grid
        j = k % size_figure_grid
        ax[i, j].cla()
        ax[i, j].imshow(test_images[k, :].numpy().reshape(28, 28), cmap='gray')

    save_file = "/Users/nandithanaik/Downloads/HW2_codes/imgs/generated_images.png"

    plt.tight_layout()
    plt.savefig(save_file)
    plt.close(fig)

def monitor_losses(d_losses, d_losses_fake, g_losses, g_losses_fake, batch_size=128):
    def moving_average(x, win=100):
        return np.convolve(x, np.ones(win), 'same') / np.convolve(np.ones_like(x), np.ones(win), 'same')

    plt.figure(figsize=(10, 5))
    iters = np.arange(len(d_losses))
    epochs = iters * batch_size / 60000
    plt.plot(epochs, moving_average(d_losses), label='d_loss')
    plt.plot(epochs, moving_average(g_losses), label='g_loss')
    plt.plot(epochs, moving_average(d_losses_fake), label='d_loss_fake')
    plt.plot(epochs, moving_average(g_losses_fake), label='g_loss_fake')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig("/Users/nandithanaik/Downloads/HW2_codes/imgs/loss_monitoring.png")
    plt.close()

def train_main_loop(generator, discriminator, d_optimizer, g_optimizer, test_noise, num_epochs, d_freq, train_loader, dim_z, criterion, device):
    d_losses = []
    g_losses = []
    d_losses_fake = []
    g_losses_fake = []

    for epoch in tqdm(range(num_epochs)):
        for n, (images, cls_labels) in enumerate(train_loader):
            images = Variable(images.to(device))
            cls_labels = cls_labels.to(device)

            # Sample from generator
            noise = Variable(torch.randn(images.size(0), dim_z).to(device))
            fake_images = generator(noise)

            # Train the discriminator
            d_loss, real_score, fake_score, d_loss_fake = train_discriminator(discriminator, d_optimizer, images, fake_images, criterion, device)

            if n % d_freq == 0:
                # Sample again from the generator and get output from discriminator
                noise = Variable(torch.randn(images.size(0), dim_z).to(device))
                fake_images = generator(noise)
                outputs = discriminator(fake_images)
                # Train the generator
                g_loss, g_loss_fake = train_generator(generator, g_optimizer, outputs, criterion,device)

            d_losses.append(d_loss.data.detach().cpu().numpy())
            g_losses.append(g_loss.data.detach().cpu().numpy())
            d_losses_fake.append(d_loss_fake.data.detach().cpu().numpy())
            g_losses_fake.append(g_loss_fake.data.detach().cpu().numpy())

        monitor_images(generator, test_noise)
        monitor_losses(d_losses, d_losses_fake, g_losses, g_losses_fake)

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    batch_size = 128
    dim_z = 100
    num_test_samples = 64
    num_epochs = 100     # TODO: Set an appropriate number of epochs (e.g., 50–100+ for GANs). Try multiple values and compare results.
    d_learning_rate = 2e-4   # TODO: Set the discriminator learning rate. Experiment with different values and explain your choice in the report. 1e-5<LR<1e-3 should be a good start
    g_learning_rate = 2e-4   # TODO: Set the generator learning rate. Experiment with different values and explain your choice in the report. 1e-5<LR<1e-3 should be a good start
    criterion = nn.BCEWithLogitsLoss()        # TODO: Choose a suitable loss function. Justify your choice in the report.
    d_freq = 1               # TODO: Modify this in part (c), bullet point #4 as instructed.

    tensor_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5, ), std=(0.5, ))])
    train_dataset = datasets.MNIST(root = "./data",	train = True, download = True,	transform = tensor_transform)    
    train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)

    test_noise = Variable(torch.randn(num_test_samples, dim_z).to(device))
    discriminator = Discriminator().to(device)
    generator = Generator().to(device)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=d_learning_rate)
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=g_learning_rate)

    train_main_loop(generator, discriminator, d_optimizer, g_optimizer, test_noise, num_epochs, d_freq, train_loader, dim_z, criterion, device)

if __name__ == "__main__":
    main()