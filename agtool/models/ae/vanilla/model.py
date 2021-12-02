from os import makedirs
from os.path import exists


import torch
from tqdm import tqdm

from agtool.misc import get_logger
from agtool.models import PytorchModelBase


class DeepAutoEncoder(PytorchModelBase):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(28 * 28, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 10)
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(10, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 28 * 28),
            torch.nn.Sigmoid()
        )
        self.clf = torch.nn.Sequential(
            torch.nn.Linear(28 * 28, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 10),
        )
        self.logger = get_logger('DeepAutoEncoder', 'INFO')

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def fit(self, epochs=100, lr=1e-3, batch_size=256):
        import pickle
        import matplotlib.pyplot as plt

        from agtool.models.ae.vanilla.loader import get_loader

        makedirs('./results', exist_ok=True)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        train_loader, _ = get_loader(batch_size)
        train_loss = []
        # Dictionary that will store the different images and outputs for various epochs
        outputs = {}
        batch_size = len(train_loader)
        for epoch in tqdm(range(epochs), 'Train', ncols=100):
            self.train()
            running_loss = 0
            for (img, _) in train_loader:
                # reshaping it into a 1-d vector
                img = img.reshape(-1, 28 * 28)
                out = self(img)
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
        self.save('./results/vanilla_ae.pt')
        with open('./results/outputs', 'wb') as fout:
            pickle.dump(outputs, fout)
        # Plotting the training loss
        plt.plot(range(1, epochs + 1), train_loss)
        plt.xlabel("Number of epochs")
        plt.ylabel("Training Loss")
        plt.show()
        plt.savefig('./results/train_graph.png')

    def analysis(self):
        import pickle
        from agtool.models.ae.vanilla import get_loader, analysis_train, analysis_test

        if not exists('./MNIST/test'):
            self.logger.info('Please download MNIST dataset before run analysis.')
            return
        elif not exists('./results/outputs'):
            self.logger.info('Please fit(train) model before run analysis.')
            return
        _, test_loader = get_loader()
        with open('./results/outputs', 'rb') as fin:
            outputs = pickle.load(fin)
        self.from_pretrained('./results/vanilla_ae.pt')
        analysis_train(outputs, './results/analysis_train.png')
        analysis_test(self, test_loader, './results/analysis_test.png')

    def fit_classifier(self, epochs=100, lr=1e-3, batch_size=256):
        from agtool.models.ae.vanilla.loader import get_loader

        if not exists('./results/vanilla_ae.pt'):
            self.logger.info('Please fit(train) model before run analysis.')
            return
        self.from_pretrained('./results/vanilla_ae.pt')
        optimizer = torch.optim.Adam(self.clf.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()
        train_loader, _ = get_loader(batch_size)
        self.encoder.eval()
        self.decoder.eval()
        self.clf.train()
        pbar = tqdm(range(epochs), 'Train Classifier', ncols=100)
        for epoch in pbar:
            hit = total = 0
            for (img, label) in train_loader:
                img = img.reshape(-1, 28 * 28)
                out = self(img)
                logit = self.clf(out)
                loss = criterion(logit, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                hit += (label == logit.argmax(-1)).sum()
                total += len(label)
            acc = hit / total * 100
            pbar.set_postfix({'Accurcay[%]': acc.item()})
        self.save('./results/vanilla_ae.pt')

    def test_classifier(self, batch_size=256):
        from agtool.models.ae.vanilla.loader import get_loader

        if not exists('./results/vanilla_ae.pt'):
            self.logger.info('Please fit(train) model before run analysis.')
            return
        self.from_pretrained('./results/vanilla_ae.pt')
        self.eval()
        _, test_loader = get_loader(batch_size)
        hit = total = 0
        for (img, label) in tqdm(test_loader, 'Test Classifier'):
            img = img.reshape(-1, 28 * 28)
            out = self(img)
            hit += (label == self.clf(out).argmax(-1)).sum()
            total += len(label)
        acc = hit / total * 100
        self.logger.info(f"Accuracy: {acc:.4f} [%]")


if __name__ == '__main__':
    model = DeepAutoEncoder()
    print(model)
    model.fit(epochs=10)
    model.analysis()
    model.fit_classifier(epochs=10)
    model.test_classifier()
