import numpy as np

import torch
import torch.nn.functional as F
import torch.cuda.profiler as profiler
import torchmetrics.functional as metrics


from Utils import plot_metric_curve

# This is the actual train loop we will use for profiling
def train_loop(model, device, train_loader, optimizer, epoch, log_interval, logdir, idx_profile_start = 5, idx_profile_end=10):
    model.train()
    
    # Make sure autograd emits nvtx range on every operation
    with torch.autograd.profiler.emit_nvtx():
        for batch_idx, (data, target) in enumerate(train_loader):
            if batch_idx == idx_profile_start:
                # This will emit a starting signal that is respected by nsys if invoked with -c cudaProfilerApi
                profiler.start()

            # push range for current iteration
            if batch_idx >= idx_profile_start: torch.cuda.nvtx.range_push("iteration{}".format(batch_idx))

            # move data and target to the gpu, if available and used
            if batch_idx >= idx_profile_start: torch.cuda.nvtx.range_push("data_to_device")
            data, target = map(lambda tensor: tensor.to(device, non_blocking=True), (data, target))
            if batch_idx >= idx_profile_start: torch.cuda.nvtx.range_pop()
            
            # zero_grad range
            if batch_idx >= idx_profile_start: torch.cuda.nvtx.range_push("zero_grad")
            optimizer.zero_grad()
            if batch_idx >= idx_profile_start: torch.cuda.nvtx.range_pop()

            # forward range
            if batch_idx >= idx_profile_start: torch.cuda.nvtx.range_push("forward")
            output = model(data)
            if batch_idx >= idx_profile_start: torch.cuda.nvtx.range_pop()

            # loss range
            if batch_idx >= idx_profile_start: torch.cuda.nvtx.range_push("loss")
            loss = F.nll_loss(output, target)
            if batch_idx >= idx_profile_start: torch.cuda.nvtx.range_pop()

            # backward range
            if batch_idx >= idx_profile_start: torch.cuda.nvtx.range_push("backward")
            loss.backward()
            if batch_idx >= idx_profile_start: torch.cuda.nvtx.range_pop()

            # optimizer step range
            if batch_idx >= idx_profile_start: torch.cuda.nvtx.range_push("optimizer_step")
            optimizer.step()
            if batch_idx >= idx_profile_start: torch.cuda.nvtx.range_pop()
            
            # metrics step range
            if batch_idx >= idx_profile_start: torch.cuda.nvtx.range_push("metrics_step")
            accuracy = metrics.accuracy(output, target, task='multiclass', num_classes=10)
            if batch_idx >= idx_profile_start: torch.cuda.nvtx.range_pop()
            
            # Logging range
            if batch_idx >= idx_profile_start: torch.cuda.nvtx.range_push("logging")
            if batch_idx % log_interval == 0:
                print(
                    f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100 * batch_idx / len(train_loader):.0f}%)]'
                    f'\tLoss: {loss.detach().item():.6f}'
                    f'\tAccuracy: {accuracy.detach().item():.2f}'
                )
            if batch_idx >= idx_profile_start: torch.cuda.nvtx.range_pop()

            yield loss.detach().item(), accuracy.detach().item() 

            # pop iteration range
            if batch_idx >= idx_profile_start: torch.cuda.nvtx.range_pop()
                
            # Allow to break early for the purpose of shorter profiling
            if batch_idx == idx_profile_end:
                profiler.stop()


# Note that the test loop is decorated
# with the @torch.no_grad() decorator. This tells PyTorch that it doesn't need to compute gradients
# in the test loops, as those are not needed. This will speed up execution.
@torch.no_grad()
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    for data, target in test_loader:
        # move data and target to the gpu, if available and used
        data, target = map(lambda tensor: tensor.to(device, non_blocking=True), (data, target))

        # get model output
        output = model(data)

        # calculate loss
        test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss

        # get most likely class label
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

        # count the number of correct predictions
        correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100 * correct / len(test_loader.dataset)))
    
    yield test_loss, correct / len(test_loader.dataset)

def fit_profiling(model, optimizer, n_epochs, device, train_loader, test_loader, log_interval, logdir):

    # get the validation loss and accuracy of the untrained model
    start_val_loss, start_val_acc = tuple(test(model, device, test_loader))[0]

    # don't mind the following train/test loop logic too much, if you want to know what's happening, let us know :)
    # normally you would pass a logger to your train/test loops and log the respective metrics there
    (train_loss, train_acc), (val_loss, val_acc) = map(lambda arr: np.asarray(arr).transpose(2,0,1), zip(*[
        (
            [*train_loop(model, device, train_loader, optimizer, epoch, log_interval, logdir)],
            [*test(model, device, test_loader)]
        )
        for epoch in range(n_epochs)
    ]))

    # flatten the arrays
    train_loss, train_acc, val_loss, val_acc = map(np.ravel, (train_loss, train_acc, val_loss, val_acc))

    # prepend the validation loss and accuracy of the untrained model
    val_loss, val_acc = (start_val_loss, *val_loss), (start_val_acc, *val_acc)

    plot_metric_curve(train_loss, val_loss, n_epochs, 'Loss')
    plot_metric_curve(train_acc, val_acc, n_epochs, 'Accuracy')