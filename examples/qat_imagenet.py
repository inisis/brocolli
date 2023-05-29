import time
import warnings

warnings.filterwarnings("ignore")
from loguru import logger
import torch
import torchvision.models as models
from torchvision import transforms
from torch.utils.data.dataset import Subset

from brocolli.quantization.quantizer import PytorchQuantizer
from brocolli.testing.dataset import ImageNetDatasetCHINA
from brocolli.testing.quant_utils import AverageMeter, ProgressMeter, accuracy


def train(train_loader, model, criterion, optimizer, epoch, device, args):
    losses = AverageMeter("Loss", ":.4f")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    batch_time = AverageMeter("Time", ":6.3f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch),
    )

    total_sample = len(train_loader.sampler)
    batch_size = train_loader.batch_size
    logger.info(f"Training: {total_sample} samples ({batch_size} per mini-batch)")

    model.train()
    end_time = time.time()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)

        loss = criterion(outputs, targets)

        acc1, acc5 = accuracy(outputs.data, targets.data, topk=(1, 5))
    
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1.item(), inputs.size(0))
        top5.update(acc5.item(), inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        if (batch_idx + 1) % 20 == 0:
            progress.display(batch_idx + 1)

    logger.info(f" * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} Loss {losses.avg:.3f}")

    return top1.avg, top5.avg, losses.avg


def train_func(model):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    device = torch.device("cuda")
    model.to(device)
    dataset = ImageNetDatasetCHINA(
        "data",
        transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        ),
        True,
    )
    train_loader = torch.utils.data.DataLoader(
        # Subset(dataset, indices=[_ for _ in range(0, 8)]),
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )

    criterion = torch.nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                lr=0.01, momentum=0.9, weight_decay=0.0001
    )

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 30, gamma=0.1)

    for epoch in range(0, 1):
        train(train_loader, model, criterion, optimizer, epoch, device, None)
        lr_scheduler.step()


def calibrate_func(model):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    dataset = ImageNetDatasetCHINA(
        "data",
        transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        ),
        True,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    batch_time = AverageMeter("Time", ":6.3f")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(len(val_loader), [batch_time, top1, top5], prefix="Test: ")

    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            # compute output
            output = model(images)
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:
                progress.display(i)

        logger.info(f" * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}")


model = models.resnet18(pretrained=True)
model.eval()

pytorch_quantizer = PytorchQuantizer(model, (1, 3, 224, 224))
pytorch_quantizer.fuse()
pytorch_quantizer.prepare_calibration()
pytorch_quantizer.calibrate(calibrate_func)
pytorch_quantizer.prepare_finetune()
pytorch_quantizer.finetune(train_func)
pytorch_quantizer.convert()
pytorch_quantizer.evaluate(calibrate_func)
