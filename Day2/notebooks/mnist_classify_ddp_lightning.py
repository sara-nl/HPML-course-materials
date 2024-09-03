import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import pytorch_lightning as L
from torch.utils.data import DataLoader, random_split
import os



class MNISTClassifier(L.LightningModule):
    def __init__(self, lr, gamma, data_dir, batch_size, test_batch_size):
        super().__init__()
        self.lr = lr
        self.gamma = gamma
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adadelta(self.parameters(), lr=self.lr)
        scheduler = StepLR(optimizer, step_size=1, gamma=self.gamma)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def training_step(self, batch, batch_idx):
        data, target = batch
        output = self.forward(data)
        loss = F.nll_loss(output, target)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        data, target = batch
        output = self.forward(data)
        loss = F.nll_loss(output, target)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        data, target = batch
        output = self.forward(data)
        # loss = F.nll_loss(output, target, reduction='sum').item()
        loss = F.nll_loss(output, target)
        pred = torch.argmax(output, dim=1)
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # correct = pred.eq(target.view_as(pred)).sum().item()
        self.log('test_acc', self.test_accuracy, prog_bar=True)

    #####################
    # DATA RELATED HOOKS
    ####################

    #def prepare_data(self):
    #    datasets.MNIST(self.data_dir, train=True, download=False)
    #    datasets.MNIST(self.data_dir, train=False) 

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            mnist_full = datasets.MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = torch.utils.data.random_split(mnist_full, [55000, 5000])
        if stage == "test":
            self.mnist_test = datasets.MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, shuffle="True")

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.test_batch_size)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=2, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise Exception('CUDA not found')

    torch.manual_seed(args.seed)
   
   # train_kwargs = {'batch_size': args.batch_size}
   # test_kwargs = {'batch_size': args.test_batch_size}
   # cuda_kwargs = {'num_workers': int(os.environ["SLURM_CPUS_PER_TASK"]),
   #                 'pin_memory': True,
   #                 'shuffle': True}
   # train_kwargs.update(cuda_kwargs)
   # test_kwargs.update(cuda_kwargs)

    DATA_PATH = os.getenv('TEACHER_DIR')
    MNIST_DATA = os.path.join(DATA_PATH, 'JHS_data')
    # torch.manual_seed(args.seed)

    #world_size = int(os.environ["WORLD_SIZE"])
    #rank = int(os.environ["SLURM_PROCID"])
    #gpus_per_node = torch.cuda.device_count()

    # data_module = MNISTDataModule(MNIST_DATA, args.batch_size, args.test_batch_size)
    model = MNISTClassifier(args.lr, args.gamma, MNIST_DATA, args.batch_size, args.test_batch_size)
    trainer = L.Trainer(max_epochs=5)
    trainer.fit(model)


if __name__ == '__main__':
    main()
