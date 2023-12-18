import lightning as L
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torchmetrics


class ClassModel(L.LightningModule):
    def __init__(
        self,
        num_classes,
        backbone,
        img_w,
        img_h,
        img_c,
        hidd_l=[64],
        learning_rate=1e-4,
        weight_decay=1e-4,
        encoder_depth=5,
        decoder_channels=[256, 128, 64, 32, 16],
    ) -> None:
        super().__init__()

        self.save_hyperparameters()
        self.name=''
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
        self.encoder = self.model.encoder

        #
        # Get latent dim size
        # if there a 5 layers, channels will be imsize*2^4. Each feature
        # size will then be (imsize/2^5)^2
        #
        s1 = self.encoder.out_channels[-1]
        imsize = img_h
        s2 = imsize // (2**5)  # 5 = length of encoder layers
        #
        # Make an MLP from a list of hidden layer sizes
        #
        layers = []
        for i in range(len(hidd_l) + 1):
            if i == 0:
                layers.append(nn.Linear(s1 * s2 * s2, hidd_l[i]))
                layers.append(nn.ReLU())
            elif i == len(hidd_l):
                layers.append(nn.Linear(hidd_l[i - 1], num_classes))
            else:
                layers.append(nn.Linear(hidd_l[i - 1], hidd_l[i]))
                layers.append(nn.ReLU())

        self.classifier = nn.Sequential(*layers)

        #
        # Torch metrics
        #
        self.train_acc = torchmetrics.Accuracy(
            task="multiclass", average="macro", num_classes=self.num_classes
        )
        self.val_acc = torchmetrics.Accuracy(
            task="multiclass", average="macro", num_classes=self.num_classes
        )
        self.test_acc = torchmetrics.Accuracy(
            task="multiclass", average="macro", num_classes=self.num_classes
        )

    def forward(self, x):
        """Propogate Network"""
        x = self.encoder(x)
        x = x[-1].flatten(start_dim=1)
        logits = self.classifier(x)

        return logits

    def _get_loss(self, batch):
        """
        Given a batch of images, this function returns the classification loss
        """
        x, y = batch
        logits = self.forward(x)
        loss = self.loss(logits, y)
        return loss, logits, y

    def training_step(self, batch, batch_idx):
        loss, logits, y = self._get_loss(batch)

        self.train_acc.update(logits, y)
        self.log("class/train/loss", loss, on_epoch=True, on_step=True)
        self.log("class/train/acc", self.train_acc, on_epoch=True, on_step=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits, y = self._get_loss(batch)

        self.train_acc.update(logits, y)
        self.log("class/val/loss", loss, on_epoch=True, on_step=True)
        self.log("class/val/acc", self.val_acc, on_epoch=True, on_step=True)

        return loss

    def test_step(self, batch, batch_idx):
        loss, logits, y = self._get_loss(batch)

        self.train_acc.update(logits, y)
        self.log("class/test/loss", loss, on_epoch=True, on_step=True)
        self.log("class/test/acc", self.test_acc, on_epoch=True, on_step=True)

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
            "monitor": "class/val/acc",
        }
