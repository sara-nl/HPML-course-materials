import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from socket import gethostname


"""

Steps for multi gpu/node training:

1. setup the (gpu) processes.
2. split up the dataloader to each process, with the distributed sampler in pytorch.
3. distribute the model across the processes as well. Wrap our model with DDP.

"""


# SIMPLE MODEL

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

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

# Simple Train
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    
    ######## throughput #######################
    repetitions = len(train_loader)
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    total_time = 0
    ######## throughput #######################
    
    for batch_idx, (data, target) in enumerate(train_loader):
        ######## throughput #######################
        starter.record()
        ######## throughput #######################
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break
        ######## throughput #######################        
        ender.record()
        # Wait for GPU Sync
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)/1000
        total_time += curr_time
        ######## throughput #######################
    
    ######## throughput #######################
    # average time per batch in seconds
    #throughput = (repetitions*args.batch_size) / total_time
    #print(f"number of images/s through the model during training: {throughput}")
    ######## throughput #######################

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            
            
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            
    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

######################## DDP ##############################
def setup(rank, world_size):
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    # clean up the distributed environment
    dist.destroy_process_group()
######################## DDP ##############################
    
    
    
def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                        help='number of epochs to train (default: 5)')
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
   
    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    cuda_kwargs = {'num_workers': int(os.environ["SLURM_CPUS_PER_TASK"]),
                    'pin_memory': True,
                    'shuffle': True}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

    DATA_PATH = os.getenv('TEACHER_DIR')
    MNIST_DATA = os.path.join(DATA_PATH, 'JHS_data')
    
    # LOAD MNIST
    train_dset = datasets.MNIST(MNIST_DATA, train=True, download=False,
                       transform=transform)
    test_dset = datasets.MNIST(MNIST_DATA, train=False,
                       transform=transform)
    
    ######################## DDP ##############################
    world_size = int(os.environ["SLURM_NTASKS"])
    rank = int(os.environ["SLURM_PROCID"])
    gpus_per_node = torch.cuda.device_count()

    setup(rank, world_size)
    if rank == 0: print(f"Group initialized? {dist.is_initialized()}", flush=True)

    local_rank = rank - gpus_per_node * (rank // gpus_per_node)
    torch.cuda.set_device(local_rank)
    print(f"host: {gethostname()}, rank: {rank}, local_rank: {local_rank}")

    ######################## DDP ##############################
    train_loader = torch.utils.data.DataLoader(train_dset, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dset, **test_kwargs)
    
    ######################## DDP ##############################
    model = Net().to(local_rank)
    ddp_model = DDP(model, device_ids=[local_rank])
    optimizer = optim.Adadelta(ddp_model.parameters(), lr=args.lr)
    ######################## DDP ##############################

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, ddp_model, local_rank, train_loader, optimizer, epoch)
        if rank == 0: test(ddp_model, local_rank, test_loader)
        scheduler.step()

    if args.save_model and rank == 0:
        torch.save(model.state_dict(), "mnist_cnn.pt")

    cleanup()


if __name__ == '__main__':
    main()
