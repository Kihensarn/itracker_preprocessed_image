import timm
from functools import partial


__all__ = ['resnet50']

resnet50 = partial(
    timm.create_model,
    model_name='resnet50',
    num_classes=0
)

if __name__ == '__main__':
    import torch
    image = torch.rand(2, 3, 224, 224)
    model = resnet50()
    print(model)
    output= model(image)
    print(output.shape)