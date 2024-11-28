import os

from datetime import datetime

import torch
import torch.optim as optim

from torchvision import datasets, transforms

from Model import CIFAR10CNN
from Train import fit_profiling

if __name__ == "__main__":

    # Get the teacher_dir if we are in a course environment
    TEACHER_DIR = os.getenv('TEACHER_DIR', default=None)
    if TEACHER_DIR is not None:
        DATA_PATH = os.path.join(TEACHER_DIR, '/JHS_data')
    else:
        DATA_PATH = os.path.join(os.getenv('HOME'), 'CIFAR10_DATA')

    use_cuda = torch.cuda.is_available()
    print(f"CUDA is {'' if use_cuda else 'not '}available")
    device = torch.device("cuda" if use_cuda else "cpu")

    BATCH_SIZE = 256
    EPOCHS = 1
    LEARNING_RATE = 1e-4
    NUM_DATALOADER_WORKERS = 0

    LOGGING_INTERVAL = 10  # Controls how often we print the progress bar

    model = CIFAR10CNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) # optim.<OPTIMIZER_FLAVOUR>(model.parameters(), lr=LEARNING_RATE)

    transform = transforms.Compose([
        transforms.ToTensor(),
        # Normalize the data to 0 mean and 1 standard deviation, now for all channels of RGB
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    ## You can also try this, and see how bad it is:
    # train_loader, test_loader = get_dataloaders(
    #     batch_size=BATCH_SIZE,
    #     num_workers=NUM_DATALOADER_WORKERS,
    #     transform=transform
    # )

    train_loader, test_loader = (
        torch.utils.data.DataLoader(
            datasets.CIFAR10(DATA_PATH, train=train, transform=transform, download=True),
            batch_size=BATCH_SIZE,
            pin_memory=use_cuda,
            shuffle=train,
            num_workers=NUM_DATALOADER_WORKERS
        )
        for train in (True, False)
    )


    logdir = "logs/baseline/" + datetime.now().strftime("%Y%m%d-%H%M%S")

    fit_profiling(model, optimizer, EPOCHS, device, train_loader, test_loader, LOGGING_INTERVAL, logdir)