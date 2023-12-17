import lightning as L
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


class SegmentationModel(L.LightningModule):
    def __init__(self, num_classes, learning_rate, weight_decay) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.loss = nn.CrossEntropyLoss()

        #
        # Initialize the model --> updated to softmax activation
        #
        self.model = smp.Unet(
            encoder_name="resnet50",
            encoder_weights="imagenet",
            classes=self.num_classes,
            in_channels=3,
            activation=None,
            encoder_depth=5,
            decoder_channels=[256, 128, 64, 32, 16],
        )

    def forward(self, x):
        logits = self.model(x)

        return logits

    def shared_step(self, batch):
        x, y = batch
        logits = self.forward(x)
        softmax = logits.softmax(dim=1)

        loss = self.loss(logits, y)

        return loss, logits, softmax

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)

        loss = self.loss(logits, y)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)

        loss = self.loss(logits, y)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)

        loss = self.loss(logits, y)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        # Using a scheduler is optional but can be helpful.
        # The scheduler reduces the LR if the validation performance hasn't improved for the last N epochs
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.2, patience=10, min_lr=5e-5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }
