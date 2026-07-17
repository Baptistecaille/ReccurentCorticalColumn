import torch

import comparison.backbone as backbone_module


def test_mlp_encoder_maps_cifar_images_to_512_features():
    encoder_class = getattr(backbone_module, "MLPEncoder", None)
    assert encoder_class is not None, "MLPEncoder has not been implemented"

    encoder = encoder_class()
    features = encoder(torch.randn(4, 3, 32, 32))

    assert features.shape == (4, 512)


def test_mlp_encoder_has_expected_trainable_parameter_count():
    encoder_class = getattr(backbone_module, "MLPEncoder", None)
    assert encoder_class is not None, "MLPEncoder has not been implemented"

    encoder = encoder_class()
    trainable = sum(
        parameter.numel()
        for parameter in encoder.parameters()
        if parameter.requires_grad
    )

    assert trainable == 1_574_400
