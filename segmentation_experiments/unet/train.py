import os
import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from skimage.transform import resize
from matplotlib.image import imread
from matplotlib import pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import pandas as pd
from copy import deepcopy


# Configuration
class Config:
    """
    Config class to manage experiment parameters.
    """

    # Paths
    raw_images_path = "../../photo_data/cells_counting/raw"
    marked_images_path = "../../photo_data/cells_counting/marked"

    # Model parameters
    input_size = (512, 512)  # smaller size for faster training
    batch_size = 4
    learning_rate = 1e-4
    epochs = 50
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # For full-size inference
    patch_size = 1024  # for processing large images in patches
    overlap = 128  # overlap between patches to avoid edge artifacts


# Dataset Class
class MicroalgaeDataset(Dataset):
    """
    A PyTorch Dataset class for loading and preprocessing microalgae images and their corresponding segmentation masks.

    Attributes:
        image_files: List of paths to the image files.
        transform: Optional transform to apply to the images.
        train: Boolean indicating whether the dataset is for training or validation/testing.

    Class Methods:
    - __init__:
    """

    def __init__(self, image_files, transform=None, train=True):
        self.image_files = image_files
        self.transform = transform
        self.train = train

    def __len__(self):
        """
        Returns the size of the dataset.


            Parameters:
                None

            Returns:
                int: The number of image files in the collection.
        """

        return len(self.image_files)

    def __getitem__(self, idx):
        """
        Loads an image and generates a corresponding mask, resizing and normalizing both for model input. During training, the mask is read from file and binarized; otherwise, a blank mask is created. Both are converted to PyTorch tensors with appropriate dimensions.

        Args:
            idx: The index of the image to retrieve.

        Returns:
            tuple: A tuple containing the processed image (torch.Tensor) and mask (torch.Tensor).
                   The image is a CHW tensor with values normalized between 0 and 1,
                   and the mask is a single-channel tensor representing the segmentation.

        """

        img_path = os.path.join(Config.raw_images_path, self.image_files[idx])
        img = imread(img_path)

        if self.train:
            mask_path = os.path.join(Config.marked_images_path, self.image_files[idx])
            mask = imread(mask_path)
            # Convert mask to binary (black circles = 1, background = 0)
            mask = (mask[:, :, 0] < 128).astype(np.float32)
        else:
            mask = np.zeros(img.shape[:2], dtype=np.float32)

        # Resize both image and mask
        img = resize(img, Config.input_size, preserve_range=True, anti_aliasing=True)
        mask = resize(mask, Config.input_size, preserve_range=True, anti_aliasing=False)
        mask = (mask > 0.5).astype(np.float32)  # binarize again after resize

        # Normalize image
        img = img.astype(np.float32) / 255.0

        # Convert to PyTorch tensors
        img = torch.from_numpy(img).permute(2, 0, 1)  # HWC to CHW
        mask = torch.from_numpy(mask).unsqueeze(0)  # Add channel dimension

        return img, mask


# U-Net Model Architecture
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels // 2, in_channels // 2, kernel_size=2, stride=2
            )

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Pad x1 if sizes don't match exactly
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """
    OutConv layer.

    This class implements a convolutional layer with an outer product expansion,
    designed to reduce the number of parameters compared to standard convolutions.
    It's particularly useful for scenarios where computational efficiency is crucial.

    Attributes:
        in_channels: The number of input channels.
        out_channels: The number of output channels.
        kernel_size: The size of the convolutional kernel.
        stride: The stride of the convolution.
        padding: The padding applied to the input.
        weight: The learnable weights of the convolution.
        bias: The learnable bias of the convolution.

    Class Methods:
    - __init__:
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net model for image segmentation.

    This class implements the U-Net architecture, a convolutional neural network
    commonly used for semantic image segmentation tasks. It consists of an encoder
    path that downsamples the input and a decoder path that upsamples it back to
    the original resolution, with skip connections between corresponding layers
    in the encoder and decoder.

    """

    def __init__(self, n_channels=3, n_classes=1, bilinear=True):
        """
        Initializes the encoder and decoder blocks of the U-Net architecture, configuring them based on input and output channel specifications, and enabling or disabling bilinear upsampling.


        Args:
            n_channels: The number of input channels.
            n_classes: The number of output classes.
            bilinear: Whether to use bilinear interpolation in upsampling layers.

        Returns:
            None
        """

        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // (2 if bilinear else 1))

        self.up1 = Up(1024, 512 // (2 if bilinear else 1), bilinear)
        self.up2 = Up(512, 256 // (2 if bilinear else 1), bilinear)
        self.up3 = Up(256, 128 // (2 if bilinear else 1), bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        """
        Performs a complete forward pass through the U-Net, progressively downsampling and then upsampling the input to generate a segmentation map.

        Args:
            x: Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying sigmoid activation,
                          representing the predicted segmentation map.

        """

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return torch.sigmoid(logits)  # Using sigmoid for binary segmentation


# Metrics (same as your StarDist evaluation)
def iou(predicted, target):
    """
    No valid docstring found.
    """

    predicted = deepcopy(predicted)
    target = deepcopy(target)

    predicted[predicted != 0] = 1
    target[target != 0] = 1
    diff = predicted - target
    tp = np.sum(diff == 0)
    fp = np.sum(diff == 1)
    fn = np.sum(diff == -1)
    return np.round(tp / (tp + fp + fn), 3)


def area_error(predicted, target, percent=True):
    """
    No valid docstring found.
    """

    predicted = deepcopy(predicted)
    target = deepcopy(target)

    predicted[predicted != 0] = 1
    target[target != 0] = 1
    if percent:
        return np.round(abs(np.sum(predicted) - np.sum(target)) / np.sum(target), 3)
    else:
        return abs(np.sum(predicted) - np.sum(target))


# Training Function
def train_model(model, train_loader, val_loader, optimizer, criterion, epochs):
    """
    No valid docstring found.
    """

    best_iou = 0.0
    model = model.to(Config.device)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            images = images.to(Config.device)
            masks = masks.to(Config.device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        # Validation
        model.eval()
        val_loss = 0.0
        val_iou = 0.0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(Config.device)
                masks = masks.to(Config.device)

                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item() * images.size(0)

                # Calculate IoU
                preds = (outputs > 0.5).float()
                val_iou += iou(preds.cpu().numpy(), masks.cpu().numpy())

        # Print statistics
        train_loss = train_loss / len(train_loader.dataset)
        val_loss = val_loss / len(val_loader.dataset)
        val_iou = val_iou / len(val_loader)

        print(
            f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val IoU: {val_iou:.4f}"
        )

        # Save best model
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), "best_model.pth")

    return model


# Inference on full-size images using patch-based processing
def predict_full_image(model, image_path, threshold=0.5):
    """
    No valid docstring found.
    """

    # Load and preprocess image
    img = imread(image_path)
    original_size = img.shape[:2]
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # Add batch dimension

    # Pad image to be divisible by patch size
    pad_h = (Config.patch_size - img.shape[2] % Config.patch_size) % Config.patch_size
    pad_w = (Config.patch_size - img.shape[3] % Config.patch_size) % Config.patch_size
    img = F.pad(img, (0, pad_w, 0, pad_h), mode="reflect")

    # Process in patches
    output = torch.zeros(1, 1, img.shape[2], img.shape[3])
    count = torch.zeros(1, 1, img.shape[2], img.shape[3])

    model.eval()
    with torch.no_grad():
        for i in range(
            0, img.shape[2] - Config.patch_size + 1, Config.patch_size - Config.overlap
        ):
            for j in range(
                0,
                img.shape[3] - Config.patch_size + 1,
                Config.patch_size - Config.overlap,
            ):
                patch = img[
                    :, :, i : i + Config.patch_size, j : j + Config.patch_size
                ].to(Config.device)
                pred = model(patch).cpu()

                # Blend predictions in overlap regions
                output[
                    :, :, i : i + Config.patch_size, j : j + Config.patch_size
                ] += pred
                count[:, :, i : i + Config.patch_size, j : j + Config.patch_size] += 1

    # Average predictions
    output = output / count
    output = output[:, :, : original_size[0], : original_size[1]]  # Remove padding
    mask = (output.squeeze() > threshold).float().numpy()

    return mask


# Main execution
if __name__ == "__main__":
    # Prepare data
    all_files = os.listdir(Config.raw_images_path)
    train_files, val_files = train_test_split(all_files, test_size=0.2, random_state=42)

    train_dataset = MicroalgaeDataset(train_files)
    val_dataset = MicroalgaeDataset(val_files)

    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.batch_size, shuffle=False)

    # Initialize model, optimizer, and loss
    model = UNet(n_channels=3, n_classes=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.learning_rate)
    criterion = nn.BCELoss()  # Binary Cross-Entropy Loss

    # Train
    trained_model = train_model(
        model, train_loader, val_loader, optimizer, criterion, Config.epochs
    )

    # Example inference on a validation image
    sample_image_path = os.path.join(Config.raw_images_path, val_files[0])
    predicted_mask = predict_full_image(trained_model, sample_image_path)

    # Visualize validation
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(imread(sample_image_path))
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    ax[1].imshow(predicted_mask, cmap="gray")
    ax[1].set_title("Predicted Mask")
    ax[1].axis("off")

    plt.tight_layout()
    plt.savefig("unet_infrrence.png")
    plt.show()
