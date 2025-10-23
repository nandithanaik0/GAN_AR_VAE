import torch
from torchvision import transforms, datasets
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F

from models.VAE import VAE_model

def plot_latent_images(model, n, device, digit_size=28):
    grid_x = np.linspace(-2, 2, n)
    grid_y = np.linspace(-2, 2, n)

    image_width = digit_size * n
    image_height = digit_size * n
    image = np.zeros((image_height, image_width))

    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z = torch.tensor([[xi, yi]], dtype=torch.float32).to(device)
            with torch.no_grad():
                x_decoded = model.decode(z)
            digit = x_decoded.view(digit_size, digit_size).cpu().numpy()
            image[i * digit_size: (i + 1) * digit_size,
                  j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap='gray')
    plt.axis('Off')
    plt.tight_layout()
    plt.savefig("imgs/VAE_generated.png")
    plt.close()

def eval(model, train_dataset, device):
    original_imgs = torch.cat([train_dataset[i][0] for i in range(5)])
    with torch.no_grad():
      res = model(original_imgs.reshape(5, -1).to(device))
      reconstructed_imgs = res['imgs']
      reconstructed_imgs = reconstructed_imgs.cpu().reshape(*original_imgs.shape)

    fig, axes = plt.subplots(2, 5, figsize=(25, 10))

    for i in range(5):
        original_image = original_imgs[i].reshape(28, 28)
        axes[0, i].imshow(original_image, cmap='gray')
        axes[0, i].set_title(f'Original Image {i+1}')
        axes[0, i].axis('off')

        reconstructed_image = reconstructed_imgs[i].reshape(28, 28)
        axes[1, i].imshow(reconstructed_image, cmap='gray')
        axes[1, i].set_title(f'Reconstructed Image {i+1}')
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig("imgs/VAE_recon.png")
    plt.close()

def loss_func(model, x, coeff):
    output = model(x)

    ############################
    # TODO: Reconstruction loss (per sample, summed over pixels)
    # Think of a proper loss function to use for the reconstruction loss
    ############################
    #recon_loss = select_some_loss(output['imgs'], x, reduction='sum')
    recon_loss = F.binary_cross_entropy(output['imgs'], x, reduction='sum')


    ############################
    # TODO: KL divergence term (closed form)
    # Eq: 0.5 * sum_j (sigma_j^2 + mu_j^2 - 1 - log(sigma_j^2))
    # Hint: 'var' (variance) is already square of sigma (std)
    # Hint2: Sum over dim=1
    ############################
    mean = output['mean']
    logvar = output['logvar']
    var = torch.exp(logvar)
    kl_div = 0.5 * torch.sum(var + mean**2 - 1 - logvar, dim=1)
    kl_div = kl_div.mean()  # average over batch

    ############################
    # TODO: Combine reconstruction loss and KL term
    # Hint: divide recon_loss by batch size, then add coeff * kl_div
    ############################
    batch_size = x.size(0)
    loss = recon_loss / batch_size + coeff * kl_div

    return loss

def train(dataloader, model, train_dataset, optimizer, epochs, coeff, device):
    for epoch in tqdm(range(epochs), desc='Epochs'):
        running_loss = 0.0
        batch_progress = tqdm(dataloader, desc='Batches', leave=False)
        for _, (images, labels) in enumerate(batch_progress):
            batch_size = images.shape[0]
            images = images.reshape(batch_size, -1).to(device)
            loss = loss_func(model, images, coeff)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            avg_loss = running_loss / len(train_dataset) * batch_size

        tqdm.write(f'----\nEpoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.4f}\n')
        plot_latent_images(model, n=8, device=device)
        eval(model, train_dataset, device)

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    batch_size = 256
    input_dim = 256
    learning_rate = 1e-3      # TODO: Set the learning rate. Experiment with different values and explain your choice in the report. 1e-4<LR<1e-2 should be a good start  
    epochs = 30           # TODO: Set an appropriate number of epochs (e.g., 20–100+ for VAEs). Try multiple values and compare results.
    coeff = 1e-2              # TODO: Set an appropriate KL regularization weight. Try multiple values and compare results. 1e-3<coeff<1e-0 should be a good start  
    hidden_dims = [1024, 512, 256, 4]       # TODO: Set the channel size from the given set of values in the HW PDF e.g., [#, #, #, 4]
    assert hidden_dims[-1] == 4, "always use 4 as the latent dimension for generating a 2D image grid during evaluation"
    
    tensor_transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root = "./data",	train = True, download = True,	transform = tensor_transform)    
    train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)

    image_shape = train_dataset[0][0].shape
    input_dim = torch.prod(torch.tensor(image_shape)).item()
    print("input_dim: ", input_dim)

    vae_model = VAE_model(input_dim, hidden_dims).to(device)
    optimizer = torch.optim.Adam(vae_model.parameters(),  lr = learning_rate,  weight_decay = 1e-8)

    train(train_loader, vae_model, train_dataset, optimizer, epochs, coeff, device)

if __name__ == "__main__":
    main()