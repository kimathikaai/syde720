import lightning as L
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F


class MAEUNet(L.LightningModule):
    """
    This class implements UNet model architecture with masked
    autoencoding (MAE) pretraining
    """

    def __init__(
        self,
        num_classes,
        img_w,
        img_h,
        img_c,
        backbone,
        learning_rate=1e-4,
        weight_decay=1e-4,
        encoder_depth=5,
        decoder_channels=[256, 128, 64, 32, 16],
        mae_mask_loss=True,
    ):
        super(MAEUNet, self).__init__()
        self.save_hyperparameters()
        self.name = "mae"

        self.num_classes = num_classes

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.mae_mask_loss = mae_mask_loss
        self.img_channels = img_c
        # for logging network graph of tensorboard
        self.example_input_array = torch.zeros(2, img_c, img_h, img_w)

        #
        # Initialize the model
        #
        assert len(decoder_channels) == encoder_depth
        self.model = smp.Unet(
            backbone,
            encoder_weights="imagenet",
            classes=num_classes,
            in_channels=img_c,
            activation=None,
            encoder_depth=encoder_depth,
            decoder_channels=decoder_channels,
        )

        #
        # using a shared encoder for pretraining and downstream
        #
        self.encoder = self.model.encoder

        #
        # Initialize pretraining modules
        #
        mae_activation = "sigmoid"  # pixel value range [0,1]
        out_channels = self.encoder.out_channels
        # remove skip decoder skip connections for image reconstruction
        mae_channels = [out_channels[0]] + [0] * encoder_depth + [out_channels[-1]]
        print("[info]", "self.encoder.out_channels", self.encoder.out_channels)
        print("[info]", "mae_channels", mae_channels)
        self.mae_decoder = smp.decoders.unet.decoder.UnetDecoder(
            encoder_channels=mae_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=True,
            center=False,
            attention_type=None,
        )
        self.mae_segmentation_head = smp.base.heads.SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=img_c,
            activation=mae_activation,
            kernel_size=3,
        )

    def forward(self, inputs):
        """
        Forward pass for mae image reconstruction task with
        the sepearate decoder and segmentation head. The
        encoder is shared
        """
        x_e = self.encoder(inputs)
        x_e = [x_e[0], x_e[-1]]
        x_d = self.mae_decoder(*x_e)
        x_h = self.mae_segmentation_head(x_d)
        return x_h

    def _get_naive_predictions(self, x, mask):
        """
        Generate simple predictions to baseline the model's learning
        `per_masked_pixel_batch_mean` find the average pixel intensity for each non-masked pixel in the batch
        `per_masked_batch_mean` find the average pixel intensity
        """
        zero_imgs = torch.zeros_like(x, device=x.device)

        per_masked_pixel_batch_mean = x.sum(0) / ((~mask).sum(0) + 1e-08)
        per_masked_batch_mean = x.sum() / (~mask).sum()

        return (
            zero_imgs + per_masked_pixel_batch_mean,
            zero_imgs + per_masked_batch_mean,
        )

    def _get_mae_loss(self, batch):
        """
        Given a batch of images, this function returns the MSE reconstruction loss
        """
        x, y, mask = batch
        x_hat = self.forward(x)

        # calculate losses
        loss = F.mse_loss(y, x_hat, reduction="none")

        with torch.no_grad():
            pixel_mean, batch_mean = self._get_naive_predictions(x, mask)
            pixel_mean_loss = (
                (F.mse_loss(y, pixel_mean, reduction="none") * mask)
                .sum([1, 2, 3])
                .mean(0)
            )
            batch_mean_loss = (
                (F.mse_loss(y, batch_mean, reduction="none") * mask)
                .sum([1, 2, 3])
                .mean(0)
            )

        masked_loss = (loss * mask).sum(dim=[1, 2, 3]).mean(dim=[0])
        total_loss = loss.sum(dim=[1, 2, 3]).mean(dim=[0])

        # whether to only report masked patch loss
        loss = masked_loss if self.mae_mask_loss else total_loss

        return loss, masked_loss, total_loss, pixel_mean_loss, batch_mean_loss

    def shared_step(self, batch, stage):
        (
            loss,
            masked_loss,
            total_loss,
            pixel_mean_loss,
            batch_mean_loss,
        ) = self._get_mae_loss(batch)

        metrics = {
            f"mae/{stage}/loss": loss,
            f"mae/{stage}/masked_loss": masked_loss,
            f"mae/{stage}/total_loss": total_loss,
            f"mae/{stage}/pixel_mean_loss": pixel_mean_loss,
            f"mae/{stage}/batch_mean_loss": batch_mean_loss,
        }
        self.log_dict(metrics, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch=batch, stage="train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch=batch, stage="val")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch=batch, stage="test")

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
            "monitor": "mae/train/loss",
        }
