"""Image augmentations adapted from facebookresearch/eb_jepa.

Source: ``examples/image_jepa/dataset.py`` at commit
``966e61e9285b3a876f49b9774e9720d9a99a7925``.
Local adaptations: injectable crop scale and ratio, probability aliases, and
the blur/order required by the project roadmap.
"""

import torch
import torchvision.transforms as transforms

class RandomResizedCrop:
    """Random resized crop augmentation."""

    def __init__(
        self,
        size: int,
        scale: tuple[float, float] = (0.2, 1.0),
        ratio: tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
    ) -> None:
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.transform = transforms.RandomResizedCrop(
            size,
            scale=scale,
            ratio=ratio,
        )

    def __call__(self, img):
        return self.transform(img)


class ColorJitter:
    """Color jitter augmentation."""

    def __init__(
        self,
        p: float = 0.8,
        *,
        brightness: float = 0.4,
        contrast: float = 0.4,
        saturation: float = 0.2,
        hue: float = 0.1,
        prob: float | None = None,
    ) -> None:
        if prob is not None:
            p = prob
        self.transform = transforms.ColorJitter(
            brightness,
            contrast,
            saturation,
            hue,
        )
        self.p = p
        self.prob = p

    def __call__(self, img):
        if torch.rand(1) < self.prob:
            return self.transform(img)
        return img


class Grayscale:
    """Grayscale augmentation."""

    def __init__(self, p: float = 0.2, *, prob: float | None = None) -> None:
        if prob is not None:
            p = prob
        self.p = p
        self.prob = p
        self.transform = transforms.Grayscale(num_output_channels=3)

    def __call__(self, img):
        if torch.rand(1) < self.prob:
            return self.transform(img)
        return img


class Solarization:
    """Solarization augmentation."""

    def __init__(self, p: float = 0.0, *, prob: float | None = None) -> None:
        if prob is not None:
            p = prob
        self.p = p
        self.prob = p

    def __call__(self, img):
        if torch.rand(1) < self.prob:
            img = transforms.functional.solarize(img, threshold=128)
        return img


class HorizontalFlip:
    """Horizontal flip augmentation."""

    def __init__(self, p: float = 0.5, *, prob: float | None = None) -> None:
        if prob is not None:
            p = prob
        self.p = p
        self.prob = p

    def __call__(self, img):
        if torch.rand(1) < self.prob:
            return transforms.functional.hflip(img)
        return img


def get_train_transforms(
    crop_scale: tuple[float, float] = (0.2, 1.0),
) -> transforms.Compose:
    """Get training transforms for self-supervised learning."""
    transform = transforms.Compose(
        [
            RandomResizedCrop(32, scale=crop_scale),
            HorizontalFlip(p=0.5),
            ColorJitter(p=0.8),
            Grayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            Solarization(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010),
            ),
        ]
    )

    return transform
