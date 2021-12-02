import fire
import torch
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt

from agtool.models.ae.vanilla.model import DeepAutoEncoder
from agtool.models.ae.vanilla.loader import get_loader


def train(epochs=100, lr=1e-3, batch_size=256):
    model = DeepAutoEncoder()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_loader, test_loader = get_loader(batch_size)
    train_loss = []
    # Dictionary that will store the different images and outputs for various epochs
    outputs = {}
    batch_size = len(train_loader)
    for epoch in tqdm(range(epochs), 'Train', ncols=100):
        model.train()
        running_loss = 0
        for (img, _) in train_loader:
            # reshaping it into a 1-d vector
            img = img.reshape(-1, 28 * 28)
            out = model(img)
            loss = criterion(out, img)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Incrementing loss
            running_loss += loss.item()
        # Averaging out loss over entire batch
        running_loss /= batch_size
        train_loss.append(running_loss)
        # Storing useful images and reconstructed outputs for the last batch
        outputs[epoch + 1] = {'img': img, 'out': out}
    model.save('./vanilla_ae.pt')
    with open('./outputs', 'wb') as fout:
        pickle.dump(outputs, fout)
    # Plotting the training loss
    plt.plot(range(1, epochs + 1), train_loss)
    plt.xlabel("Number of epochs")
    plt.ylabel("Training Loss")
    plt.show()
    plt.savefig('./train_graph.png')


if __name__ == '__main__':
    fire.Fire(train)
