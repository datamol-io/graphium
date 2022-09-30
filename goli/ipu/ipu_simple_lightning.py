# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import pytorch_lightning as pl

import torch
from torch import nn

import torchvision
import torchvision.transforms as transforms

import poptorch


# The simple PyTorch model used in each of these examples
class SimpleTorchModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        conv_block = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(3),
            nn.MaxPool2d(3),
        )

        self.the_network = nn.Sequential(
            conv_block, torch.nn.Flatten(), nn.Linear(64, 16), nn.ReLU(), nn.Linear(16, 10), nn.LogSoftmax(1)
        )

    def forward(self, x):
        return self.the_network(x)


# This class shows a minimal lightning example. This example uses our own
# SimpleTorchModel which is a basic 2 conv, 2 FC torch network. It can be
# found in simple_torch_model.py.
class SimpleLightning(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = SimpleTorchModel()

    def training_step(self, batch, _):
        x, label = batch
        prediction = self.model(x)
        loss = torch.nn.functional.nll_loss(prediction, label)
        return loss

    def validation_step(self, batch, _):
        x, label = batch
        prediction = self.model(x)
        preds = torch.argmax(prediction, dim=1)
        acc = torch.sum(preds == label).float() / len(label)
        return acc

    # PopTorch doesn't currently support logging within steps. Use the Lightning
    # callback hooks instead.
    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        self.log("StepLoss", outputs["loss"])

    def validation_epoch_end(self, outputs):
        self.log("val_acc", torch.stack(outputs).mean(), prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        return optimizer


if __name__ == "__main__":
    # Create the model as usual.
    model = SimpleLightning()

    # Normal PyTorch dataset.
    train_set = torchvision.datasets.FashionMNIST(
        "FashionMNIST", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()])
    )

    # Normal PyTorch dataloader.
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=16, shuffle=True)

    training_opts = poptorch.Options()
    training_opts.Jit.traceModel(True)
    inference_opts = poptorch.Options()
    inference_opts.Jit.traceModel(True)
    plugins = pl.plugins.IPUPlugin(training_opts=training_opts, inference_opts=inference_opts)
    # Run on IPU using IPUs=1. This will run on IPU but will not include any custom
    # PopTorch Options. Changing IPUs to 1 to IPUs=N will replicate the graph N
    # times. This can lead to issues with the DataLoader batching - the script
    # ipu_options_and_dataloading.py shows how these can be avoided through the
    # use of IPUOptions.
    trainer = pl.Trainer(
        logger=pl.loggers.WandbLogger(),
        ipus=1,
        max_epochs=3,
        progress_bar_refresh_rate=20,
        log_every_n_steps=1,
        plugins=plugins,
    )

    # When fit is called the model will be compiled for IPU and will run on the available IPU devices.
    trainer.fit(model, train_loader)
