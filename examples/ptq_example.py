from loguru import logger

import time

import warnings

warnings.filterwarnings("ignore")


import torch
import torch.nn as nn
import torch.nn.functional as F

from brocolli.quantization.quantizer import PytorchQuantizer  # noqa
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from torch.hub import load_state_dict_from_url


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.relu2 = nn.ReLU()
        self.max_pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu3 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.max_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)

        return x


model = Net()
state_dict = load_state_dict_from_url(
    "http://120.224.26.73:15030/aifarm/best.pt", map_location="cpu"
)
model.load_state_dict(state_dict)
model.eval()


class MNISTCHINA(datasets.MNIST):
    mirrors = ["http://120.224.26.73:15030/aifarm/mnist/"]

    def __init__(
        self,
        root,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
    ):
        super(MNISTCHINA, self).__init__(
            root,
            train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )


def calibrate_func(model):
    test_acc = 0
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307), (0.3081))]
    )
    dataset_train = MNISTCHINA("data", download=True, train=True, transform=transform)
    dataset_train = Subset(dataset_train, indices=[_ for _ in range(0, 128)])
    train_loader = DataLoader(dataset_train, batch_size=8, shuffle=False, num_workers=8)

    with torch.no_grad():
        tick = time.time()
        for images, targets in train_loader:
            pred = model(images)
            pred_label = torch.argmax(pred, dim=1, keepdims=True)
            test_acc += pred_label.eq(targets.view_as(pred_label)).sum().item()

        tok = time.time()

    logger.info(f"float time: {tok - tick  : .4f} sec")
    test_acc /= len(train_loader.dataset)

    logger.info("calibrate acc: {}".format(test_acc))


def evaluate_func(model):
    test_acc = 0
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307), (0.3081))]
    )
    dataset_test = MNISTCHINA("data", download=True, train=False, transform=transform)
    dataset_test = Subset(dataset_test, indices=[_ for _ in range(0, 128)])
    test_loader = DataLoader(dataset_test, batch_size=8, shuffle=False, num_workers=8)

    with torch.no_grad():
        tick = time.time()
        for images, targets in test_loader:
            pred = model(images)
            pred_label = torch.argmax(pred, dim=1, keepdims=True)
            test_acc += pred_label.eq(targets.view_as(pred_label)).sum().item()
        tok = time.time()

    logger.info(f"int8 time: {tok - tick  : .4f} sec")
    test_acc /= len(test_loader.dataset)

    logger.info("evaluate acc: {}".format(test_acc))


pytorch_quantizer = PytorchQuantizer(model, (1, 1, 28, 28))
pytorch_quantizer.fuse()
pytorch_quantizer.prepare_calibration()
pytorch_quantizer.calibrate(calibrate_func)
pytorch_quantizer.convert()
pytorch_quantizer.evaluate(evaluate_func)
pytorch_quantizer.profile()
pytorch_quantizer.compare(interrested_node=["conv1", "conv2", "fc1", "fc2"])
