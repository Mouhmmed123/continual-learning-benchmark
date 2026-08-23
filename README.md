# Continual Learning Benchmark Suite

A PyTorch framework for benchmarking continual learning algorithms under class-incremental and task-incremental setups, with a focus on mitigating catastrophic forgetting.

## 📌 Features
- **Data Streams:** Split-CIFAR100, Permuted MNIST.
- **Methods:** Naive Fine-Tuning, EWC, Experience Replay, Nearest Class Mean (NCM).
- **Metrics:** Average Accuracy ($A_K$), Backward Transfer ($BWT$), Forgetting Measure.

## 🛠️ Installation & Setup
```bash
git clone [https://github.com/mouhmmed123/continual-learning-benchmark.git](https://github.com/mouhmmed123/continual-learning-benchmark.git)
cd continual-learning-benchmark
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .