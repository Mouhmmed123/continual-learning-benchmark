import torch
from cl_benchmark.models import MLP, SimpleConvNet, ContinualModel

def test_mlp_single_and_multi_head():
    backbone = MLP(input_dim=784, hidden_dim=256)
    dummy_input = torch.randn(16, 1, 28, 28)

    # Test Single-Head (5 tasks * 2 classes = 10 classes)
    single_model = ContinualModel(backbone, num_classes_per_task=2, num_tasks=5, multi_head=False)
    out_single = single_model(dummy_input)
    assert out_single.shape == (16, 10), f"Expected (16, 10), got {out_single.shape}"

    # Test Multi-Head (2 output logits for active task head)
    multi_model = ContinualModel(backbone, num_classes_per_task=2, num_tasks=5, multi_head=True)
    out_multi_t0 = multi_model(dummy_input, task_id=0)
    assert out_multi_t0.shape == (16, 2), f"Expected (16, 2), got {out_multi_t0.shape}"
    print("✓ MLP Single/Multi-Head: Pass")

def test_convnet_forward():
    backbone = SimpleConvNet(hidden_dim=512)
    dummy_input = torch.randn(8, 3, 32, 32)
    model = ContinualModel(backbone, num_classes_per_task=2, num_tasks=5, multi_head=True)
    out = model(dummy_input, task_id=2)
    assert out.shape == (8, 2), f"Expected (8, 2), got {out.shape}"
    print("✓ ConvNet Multi-Head: Pass")

if __name__ == "__main__":
    test_mlp_single_and_multi_head()
    test_convnet_forward()
    print("\nAll model architecture tests passed successfully!")