from functools import partial
import timm


__all__ = [
    'resnest269e'
]

resnest269e = partial(
    timm.create_model,
    model_name='resnest269e',
    num_classes=0,
    # global_pool=''
)


if __name__ == '__main__':
    import torch
    # model = resnest269e(pretrained=True)
    # # print(model)
    # data = torch.rand(2, 3, 224, 224)
    # print(data.shape)
    # output = model(data)
    # print(output.shape)
    if 'densenet121' in timm.list_models(pretrained=True):
        print('exist')