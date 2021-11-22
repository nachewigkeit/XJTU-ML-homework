import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from time import time
from model import MLP, Conv
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def train(model, device, train_loader, optimizer):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()


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

    print('Loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    train_batch_size = 64
    test_batch_size = 1000
    epochs = 10
    gamma = 0.7

    device = torch.device("cpu")
    train_kwargs = {'batch_size': train_batch_size}
    test_kwargs = {'batch_size': test_batch_size}
    cuda_kwargs = {'num_workers': 1,
                   'pin_memory': True,
                   'shuffle': True}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        # transforms.RandomRotation(degrees=(0, 15))
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset1 = datasets.MNIST('./data', train=True, download=True, transform=train_transform)
    dataset2 = datasets.MNIST('./data', train=False, transform=test_transform)
    train_loader = DataLoader(dataset1, **train_kwargs)
    test_loader = DataLoader(dataset2, **test_kwargs)

    writer = SummaryWriter(r'log')
    model = Conv(128, 5).to(device)
    optimizer = optim.Adam(model.parameters())
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
    start = time()
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer)
        scheduler.step()

    print("train:")
    test(model, device, train_loader)
    print("test:")
    test(model, device, test_loader)
    print("Time: ", time() - start)

    torch.save(model.state_dict(), "weight/conv.pt")


if __name__ == '__main__':
    main()
