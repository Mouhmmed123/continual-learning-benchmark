import torch

def test_pytorch_installation():
    x = torch.tensor([1.0, 2.0, 3.0])
    assert x.shape == (3,)
    assert x.sum().item() == 6.0
