import torch
import torchvision


class CustomResnet(torchvision.models.ResNet):
    def _forward_impl(self, x, get_feat=False):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if get_feat:
            return x, self.fc(x)
        x = self.fc(x)

        return x

    def forward(self, x, get_feat=False):
        return self._forward_impl(x, get_feat=get_feat)


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = CustomResnet(block, layers, **kwargs)
    if pretrained:
        state_dict = torchvision.models.resnet.load_state_dict_from_url(
            torchvision.models.resnet.model_urls[arch], progress=progress)
        model.load_state_dict(state_dict, strict=False)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', torchvision.models.resnet.BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet50(pretrained=False, progress=True, freeze=False, ckpt=None, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        freeze (bool): freeze backbone
        ckpt:
    """
    model = _resnet('resnet50', torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)
    if ckpt is not None:
        model.load_state_dict(torch.load(ckpt)['state_dict'], strict=False)
        print(f"loaded weights from: {ckpt}")
    if freeze:
        print("freezing model backbone")
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True
    return model
