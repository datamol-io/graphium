# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import pytorch_lightning as pl

import torch
from torch import nn

import torchvision
import torchvision.transforms as transforms

import poptorch
import mup

from goli.nn.base_layers import FCLayer
from goli.utils.mup import set_base_shapes

# The simple PyTorch model used in each of these examples
class SimpleTorchModel(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, kernel_size, num_classes):
        super().__init__()
        conv_block = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=hidden_dim, kernel_size=kernel_size),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size),
            nn.MaxPool2d(kernel_size),
        )

        self.the_network = nn.Sequential(
            conv_block,
            torch.nn.Flatten(),
            FCLayer(4 * hidden_dim, hidden_dim),
            FCLayer(hidden_dim, hidden_dim),
            FCLayer(hidden_dim, num_classes, activation=None, is_readout_layer=True),
            nn.LogSoftmax(1),
        )

    def forward(self, x):
        return self.the_network(x)


# This class shows a minimal lightning example. This example uses our own
# SimpleTorchModel which is a basic 2 conv, 2 FC torch network. It can be
# found in simple_torch_model.py.
class SimpleLightning(pl.LightningModule):
    def __init__(self, in_dim, hidden_dim, kernel_size, num_classes):
        super().__init__()
        self.model = SimpleTorchModel(
            in_dim=in_dim, hidden_dim=hidden_dim, kernel_size=kernel_size, num_classes=num_classes
        )

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
        loss = torch.nn.functional.nll_loss(prediction, label)
        return loss, acc

    # PopTorch doesn't currently support logging within steps. Use the Lightning
    # callback hooks instead.
    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        self.log("StepLoss", outputs["loss"])

    def validation_epoch_end(self, outputs):
        loss = [out[0] for out in outputs]
        self.log("val_loss", torch.stack(loss).mean(), prog_bar=True)

        acc = [out[1] for out in outputs]
        self.log("val_acc", torch.stack(acc).mean(), prog_bar=True)

    def configure_optimizers(self):
        optimizer = mup.optim.Adam(self.parameters(), lr=0.01)
        return optimizer


if __name__ == "__main__":

    SEED = 42
    torch.manual_seed(SEED)

    # Create the model as usual.
    base = None # SimpleLightning(in_dim=1, hidden_dim=8, kernel_size=3, num_classes=10)
    model = SimpleLightning(in_dim=1, hidden_dim=32, kernel_size=3, num_classes=10)
    model = set_base_shapes(model, base, rescale_params=False)

    torch.manual_seed(SEED)
    # Normal PyTorch dataset.
    train_set = torchvision.datasets.FashionMNIST(
        "out/FashionMNIST", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()])
    )
    val_set = torchvision.datasets.FashionMNIST(
        "out/FashionMNIST", train=False, download=True, transform=transforms.Compose([transforms.ToTensor()])
    )

    # Normal PyTorch dataloader.
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=16, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=16, shuffle=False)

    training_opts = poptorch.Options()
    training_opts.Jit.traceModel(True)
    inference_opts = poptorch.Options()
    inference_opts.Jit.traceModel(True)

    # Set the seeds
    torch.manual_seed(SEED)
    training_opts.randomSeed(SEED)
    inference_opts.randomSeed(SEED)

    plugins = pl.plugins.IPUPlugin(training_opts=training_opts, inference_opts=inference_opts)

    trainer = pl.Trainer(
        logger=pl.loggers.WandbLogger(),
        # ipus=1, # Comment this line to run on CPU
        max_epochs=3,
        progress_bar_refresh_rate=20,
        log_every_n_steps=1,
        # plugins=plugins, # Comment this line to run on CPU
    )

    # When fit is called the model will be compiled for IPU and will run on the available IPU devices.
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
