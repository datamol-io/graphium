# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import lightning
from lightning_graphcore import IPUStrategy
from lightning.pytorch.loggers import WandbLogger

import torch
from torch import nn

import torchvision
import torchvision.transforms as transforms

import mup

from graphium.nn.base_layers import FCLayer
from graphium.utils.mup import set_base_shapes


ON_IPU = True  # Change this line to run on CPU
SEED = 42


# The simple PyTorch model used in each of these examples
class SimpleTorchModel(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, kernel_size, num_classes):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_classes = num_classes

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

    def make_mup_base_kwargs(self, divide_factor: float = 2.0):
        return dict(
            in_dim=self.in_dim,
            hidden_dim=round(self.hidden_dim / divide_factor),
            kernel_size=self.kernel_size,
            num_classes=self.num_classes,
        )

    def forward(self, x):
        return self.the_network(x)


# This class shows a minimal lightning example. This example uses our own
# SimpleTorchModel which is a basic 2 conv, 2 FC torch network. It can be
# found in simple_torch_model.py.
class SimpleLightning(lightning.LightningModule):
    def __init__(self, in_dim, hidden_dim, kernel_size, num_classes, on_ipu):
        super().__init__()
        self.model = SimpleTorchModel(
            in_dim=in_dim, hidden_dim=hidden_dim, kernel_size=kernel_size, num_classes=num_classes
        )
        self.on_ipu = on_ipu

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
    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.log("StepLoss", outputs["loss"])

    def validation_epoch_end(self, outputs):
        loss = [out[0] for out in outputs]
        self.log("val_loss", torch.stack(loss).mean(), prog_bar=True)

        acc = [out[1] for out in outputs]
        self.log("val_acc", torch.stack(acc).mean(), prog_bar=True)

    def configure_optimizers(self):
        adam = torch.optim.Adam

        if self.on_ipu:
            import poptorch

            adam = poptorch.optim.Adam

        optimizer = mup.MuAdam(self.parameters(), lr=0.01, impl=adam)
        return optimizer


if __name__ == "__main__":
    torch.manual_seed(SEED)

    # Create the model as usual.
    predictor = SimpleLightning(in_dim=1, hidden_dim=32, kernel_size=3, num_classes=10, on_ipu=ON_IPU)
    model = predictor.model
    base = model.__class__(**model.make_mup_base_kwargs(divide_factor=2))
    predictor.model = set_base_shapes(model, base, rescale_params=False)

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

    torch.manual_seed(SEED)

    ipus = None
    plugins = None
    if ON_IPU:
        import poptorch

        training_opts = poptorch.Options()
        inference_opts = poptorch.Options()

        # Set the seeds
        training_opts.randomSeed(SEED)
        inference_opts.randomSeed(SEED)
        ipus = 1
        strategy = IPUStrategy(training_opts=training_opts, inference_opts=inference_opts)

    trainer = lightning.Trainer(
        logger=WandbLogger(),
        ipus=ipus,
        max_epochs=3,
        log_every_n_steps=1,
        plugins=plugins,
    )

    # When fit is called the model will be compiled for IPU and will run on the available IPU devices.
    trainer.fit(predictor, train_dataloaders=train_loader, val_dataloaders=val_loader)
