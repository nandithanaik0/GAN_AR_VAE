import torch
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

from models.AR import PixelCNN, ConditionalPixelCNN

def train(dataloader, model, optimizer, epochs, train_dataset, loss_fn, batch_size, conditioning_flag, device):

    for epoch in tqdm(range(epochs), desc='Epochs'):
        running_loss = 0.0
        batch_progress = tqdm(dataloader, desc='Batches', leave=False)

        for _, (images, labels) in enumerate(batch_progress):
            images = images.to(device)
            tgt = images.clone()

            if conditioning_flag == True:
                labels_onehot = F.one_hot(labels, num_classes=10).float().to(device)
                pred = model(images, labels_onehot)
            else:
                pred = model(images)
            
            loss = loss_fn(pred, tgt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            avg_loss = running_loss * batch_size / len(train_dataset)

        tqdm.write(f'----\nEpoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.4f}\n')

def binarize_image(tensor):
    return (tensor > 0.5).float()

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    batch_size = 128
    num_epochs = 50                      # TODO: Set an appropriate number of epochs (e.g., 20–50+ for ARs). Try multiple values and compare results.
    learning_rate =  1e-3                   # TODO: Set the learning rate. Experiment with different values and explain your choice in the report. 1e-4<LR<1e-2 should be a good 
    conditioning_flag = False               # TODO: Change this for part-(d) of Step-2 
    loss_fn = torch.nn.BCEWithLogitsLoss()                         # TODO: Set your loss function which is discussed in the HW PDF

    tensor_transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(binarize_image)])
    train_dataset = datasets.MNIST(root = "./data",	train = True, download = True,	transform = tensor_transform)    
    train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)

    if conditioning_flag == True:
        model = ConditionalPixelCNN(num_classes=10).to(device)
    else:
        model = PixelCNN().to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train(train_loader, model, optimizer, num_epochs, train_dataset, loss_fn, batch_size, conditioning_flag, device)

    model.eval()
    H, W = 28, 28
    num_samples = 64
    grid_size = 8

    samples = torch.zeros(size=(num_samples, 1, H, W), device=device)
    
    if conditioning_flag == True:
        selected_num = None            # TODO: Choose between 0-9 (which number do you want to generate)
        gen_labels = torch.full((num_samples,), selected_num, dtype=torch.long, device=device)  
        gen_labels_onehot = F.one_hot(gen_labels, num_classes=10).float()

    with torch.no_grad():
        for i in range(H):
            for j in range(W):
                if conditioning_flag == True:
                    logits = model(samples, gen_labels_onehot)
                else:
                    logits = model(samples)
                probs = torch.sigmoid(logits[:, :, i, j])
                samples[:, :, i, j] = torch.bernoulli(probs)

    np_samples = samples.cpu().numpy().transpose(0, 2, 3, 1)

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(8, 8))
    for idx in range(num_samples):
        r, c = divmod(idx, grid_size)
        axes[r, c].imshow(np_samples[idx].squeeze(), cmap="gray")
        axes[r, c].axis("off")

    plt.tight_layout()
    plt.savefig("imgs/pixelcnn_generated.png")
    plt.close(fig)

if __name__ == "__main__":
    main()