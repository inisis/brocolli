import os
import pytest
import warnings
import argparse


os.makedirs("tmp", exist_ok=True)


def test_mnist(shape=(1, 1, 28, 28)):
    import time
    import torch
    import torch.nn as nn
    from loguru import logger
    from brocolli.quantization.quantizer import PytorchQuantizer
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    from torchvision.models.utils import load_state_dict_from_url

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
    url = "http://180.127.11.166:15030/aifarm/best.pt"
    state_dict = load_state_dict_from_url(url, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    class MNISTCHINA(datasets.MNIST):
        mirrors = ["http://180.127.11.166:15030/aifarm/mnist/"]

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
        dataset_train = MNISTCHINA(
            "data", download=True, train=True, transform=transform
        )
        train_loader = DataLoader(dataset_train, batch_size=8, num_workers=8)
        with torch.no_grad():
            tick = time.time()
            for (images, targets) in train_loader:
                pred = model(images)
                pred_label = torch.argmax(pred, dim=1, keepdims=True)
                test_acc += pred_label.eq(targets).sum().item()
            tok = time.time()

        logger.info(f"float time: {tok - tick  : .4f} sec")
        test_acc /= len(train_loader.dataset)

        logger.info("calibrate acc: {}".format(test_acc))

    def evaluate_func(model):
        test_acc = 0
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307), (0.3081))]
        )
        dataset_test = MNISTCHINA(
            "data", download=True, train=False, transform=transform
        )
        test_loader = DataLoader(
            dataset_test, batch_size=8, shuffle=False, num_workers=8
        )
        with torch.no_grad():
            tick = time.time()
            for (images, targets) in test_loader:
                pred = model(images)
                pred_label = torch.argmax(pred, dim=1, keepdims=True)
                test_acc += pred_label.eq(targets).sum().item()
            tok = time.time()

        logger.info(f"int8 time: {tok - tick  : .4f} sec")
        test_acc /= len(test_loader.dataset)

        logger.info("evaluate acc: {}".format(test_acc))

    calibrate_func(model)

    pytorch_quantizer = PytorchQuantizer(model, shape)
    pytorch_quantizer.fuse()
    pytorch_quantizer.prepare_calibration()
    pytorch_quantizer.calibrate(calibrate_func)
    pytorch_quantizer.convert()
    pytorch_quantizer.evaluate(evaluate_func)
    pytorch_quantizer.profile(True)
    pytorch_quantizer.compare()


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description="Pytorch Quantization test.")
    parser.add_argument("--cov", help="foo help")
    args = parser.parse_args()
    if args.cov == "--cov":
        cov = ["--cov", "--cov-report=html:tmp/onnx_report"]
    else:
        cov = []

    pytest.main(
        ["-p", "no:warnings", "-v", "test/quantization/test_quanzation_nets.py"] + cov
    )
