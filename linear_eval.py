import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from utils import accuracy
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset
from tqdm import tqdm


def get_cifar10_data_loaders(download, root_folder, shuffle=False, batch_size=256, num_workers=16):
    train_dataset = datasets.CIFAR10(root_folder, train=True, download=download,
                                     transform=transforms.Compose([transforms.RandomResizedCrop(size=32),
                                                                   transforms.ToTensor()])
                                     )
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              num_workers=num_workers, drop_last=False, shuffle=shuffle)

    test_dataset = datasets.CIFAR10(root_folder, train=False, download=download,
                                    transform=transforms.ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=2 * batch_size,
                             num_workers=num_workers, drop_last=False)

    return train_loader, test_loader


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)

    checkpoint = torch.load(args.ckpt, map_location=device)
    state_dict = checkpoint['state_dict']

    for k in list(state_dict.keys()):
        if k.startswith('backbone.'):
            if k.startswith('backbone') and not k.startswith('backbone.fc'):
                # remove prefix
                state_dict[k[len("backbone."):]] = state_dict[k]
        del state_dict[k]

    model = torchvision.models.resnet18(weights=None, num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model = model.to(device)
    log = model.load_state_dict(state_dict, strict=False)
    assert log.missing_keys == ['fc.weight', 'fc.bias']
    print(model)

    # freeze all layers but the last fc
    for name, param in model.named_parameters():
        if name not in ['fc.weight', 'fc.bias']:
            param.requires_grad = False

    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    assert len(parameters) == 2  # fc.weight, fc.bias

    train_loader, test_loader = get_cifar10_data_loaders(download=True,
                                                         root_folder=args.data_path,
                                                         batch_size=args.batch_size,
                                                         num_workers=args.num_workers) # batch_size=1024)

    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=0.0001, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    epochs = args.epochs
    for epoch in range(epochs):
        model.train()
        top1_train_accuracy = 0
        counter = 0
        for (x_batch, y_batch) in tqdm(train_loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            top1 = accuracy(logits, y_batch, topk=(1,))
            top1_train_accuracy += top1[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            counter += 1

        top1_train_accuracy /= counter
        top1_accuracy = 0
        top5_accuracy = 0
        model.eval()
        for counter, (x_batch, y_batch) in enumerate(test_loader):
            with torch.no_grad():
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                logits = model(x_batch)

                top1, top5 = accuracy(logits, y_batch, topk=(1, 5))
                top1_accuracy += top1[0]
                top5_accuracy += top5[0]

        top1_accuracy /= (counter + 1)
        top5_accuracy /= (counter + 1)
        print(f"Epoch {epoch}\t" +
              f"Top1 Train accuracy {top1_train_accuracy.item()}\t" +
              f"Top1 Test accuracy: {top1_accuracy.item()}\t" +
              f"Top5 test acc: {top5_accuracy.item()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', metavar='DIR', default='./datasets',
                        help='path to dataset')
    parser.add_argument('--ckpt', required=True,
                        help='path to the ckpt file')
    parser.add_argument('--batch-size', type=int,
                        help='batch size')
    parser.add_argument('--learning-rate', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to run')
    parser.add_argument('--num-workers', type=int, default=16,
                        help='number of workers for data loader')

    args = parser.parse_args()
    main(args)
