import lightning as L
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torchmetrics


class SegmentationModel(L.LightningModule):
    def __init__(
        self,
        num_classes,
        backbone,
        img_c,
        learning_rate=1e-4,
        weight_decay=1e-4,
        encoder_depth=5,
        decoder_channels=[256, 128, 64, 32, 16],
    ) -> None:
        super().__init__()

        self.save_hyperparameters()
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.loss = nn.CrossEntropyLoss()

        #
        # Initialize the model --> updated to softmax activation
        #
        self.model = smp.Unet(
            encoder_name=backbone,
            encoder_weights="imagenet",
            classes=num_classes,
            in_channels=img_c,
            activation=None,
            encoder_depth=encoder_depth,
            decoder_channels=decoder_channels,
        )

        #
        # Torch metrics
        #
        self.train_iou = torchmetrics.JaccardIndex(
            task="multiclass", average="macro", num_classes=self.num_classes
        )
        self.val_iou = torchmetrics.JaccardIndex(
            task="multiclass", average="macro", num_classes=self.num_classes
        )
        self.test_iou = torchmetrics.JaccardIndex(
            task="multiclass", average="macro", num_classes=self.num_classes
        )

    def forward(self, x):
        logits = self.model(x)

        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss(logits, y)
        argmax = logits.argmax(dim=1)

        self.train_iou.update(argmax, y)
        self.log("seg/train/loss", loss, on_epoch=True, on_step=True)
        self.log("seg/train/iou", self.train_iou, on_epoch=True, on_step=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss(logits, y)
        argmax = logits.argmax(dim=1)

        self.val_iou.update(argmax, y)
        self.log("seg/val/loss", loss, on_epoch=True, on_step=True)
        self.log("seg/val/iou", self.val_iou, on_epoch=True, on_step=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss(logits, y)
        argmax = logits.argmax(dim=1)

        self.test_iou.update(argmax, y)
        self.log("seg/test/loss", loss, on_epoch=True, on_step=True)
        self.log("seg/test/iou", self.test_iou, on_epoch=True, on_step=True)

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
            "monitor": "seg/val/iou",
        }
