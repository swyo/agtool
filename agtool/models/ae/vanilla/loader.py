from typing import Tuple

import fire
import torch
import torchvision

type_loader = torch.utils.data.dataloader.DataLoader


def get_loader(batch_size=256) -> Tuple[type_loader, type_loader]:
    # Initializing the transform for the dataset
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5), (0.5))
    ])

    # Downloading the MNIST dataset
    train_dataset = torchvision.datasets.MNIST(
        root="./MNIST/train", train=True,
        transform=torchvision.transforms.ToTensor(),
        download=True)

    test_dataset = torchvision.datasets.MNIST(
        root="./MNIST/test", train=False,
        transform=torchvision.transforms.ToTensor(),
        download=True)

    # Creating Dataloaders from the
    # training and testing dataset
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size
    )
    return train_loader, test_loader


def __plot_25_images():
    import numpy as np
    import matplotlib.pyplot as plt
    train_loader, _ = get_loader()
    train_dataset = train_loader.dataset
    # Printing 25 random images from the training dataset
    random_samples = np.random.randint(1, len(train_dataset), (25))
    for idx in range(random_samples.shape[0]):
        plt.subplot(5, 5, idx + 1)
        # train_dataset[idx][0][0].shape = (28, 28)
        plt.imshow(train_dataset[idx][0][0].numpy(), cmap='gray')
        plt.title(train_dataset[idx][1])
        plt.axis('off')
    plt.tight_layout()
    plt.show()
    plt.savefig('./mnist_samples.png')


if __name__ == '__main__':
    fire.Fire(__plot_25_images)
