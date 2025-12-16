# BERT Hyperparameter Optimization Comparison

A comprehensive comparison of hyperparameter optimization methods (Genetic Algorithm, Particle Swarm Optimization, and Bayesian Optimization) for BERT fine-tuning on sentiment analysis tasks.

## Features

- **Three Optimization Methods**: Compare GA, PSO, and Bayesian optimization side-by-side
- **Real Dataset**: Uses IMDB movie review dataset (with synthetic data fallback)
- **Comprehensive Analysis**: Detailed visualizations for convergence, parameters, and performance
- **Modern Tooling**: Uses `uv` for dependency management and `ruff` for linting

## Project Structure

```
Bert-Hyperopt-Comparison/
├── src/
│   ├── __init__.py                  # Package initialization
│   ├── bert_classifier.py           # BERT model and training pipeline
│   ├── experiment.py                # Main experiment runner
│   ├── visualize.py                 # Visualization tools
│   └── optimizers/
│       ├── __init__.py              # Optimizers package
│       ├── bayesian_optimizer.py    # Bayesian optimization
│       ├── ga_optimizer.py          # Genetic algorithm
│       └── pso_optimizer.py         # Particle swarm optimization
├── results/                         # Experiment results (JSON)
├── visualizations/                  # Generated charts
├── pyproject.toml                   # Project configuration
└── README.md
```

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/Bert-Hyperopt-Comparison.git
cd Bert-Hyperopt-Comparison

# Install with uv (recommended)
uv sync

# Or install with pip
pip install -e .
```

### Run Experiment

```python
from src.experiment import HyperparameterOptimizationExperiment

# Run with small dataset (1000 samples)
experiment = HyperparameterOptimizationExperiment(data_size='small')
experiment.run_all_experiments()
```

### Generate Visualizations

```python
from src.visualize import ExperimentVisualizer

visualizer = ExperimentVisualizer()
visualizer.create_all_visualizations()
```

### Command Line

```bash
# Run experiment
uv run python -m src.experiment

# Generate visualizations
uv run python -m src.visualize
```

## Hyperparameters Optimized

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `learning_rate` | Continuous | 1e-5 to 5e-5 | AdamW optimizer learning rate |
| `batch_size` | Discrete | [8, 16, 32] | Training batch size |
| `epochs` | Discrete | [2, 3, 4, 5] | Number of training epochs |
| `dropout_rate` | Continuous | 0.1 to 0.5 | Dropout probability |
| `max_length` | Discrete | [64, 128, 256] | Maximum sequence length |
| `warmup_ratio` | Continuous | 0.1 to 0.5 | Learning rate warmup ratio (Bayesian only) |

## Optimization Methods

### Genetic Algorithm (GA)
Evolves a population of hyperparameter sets using selection, crossover, and mutation operators.

### Particle Swarm Optimization (PSO)
Particles explore the search space while sharing information about good solutions found.

### Bayesian Optimization
Uses Gaussian processes to model the objective function and intelligently select promising hyperparameters.

## Development

```bash
# Install dev dependencies
uv sync --dev

# Run linting
uv run ruff check src

# Format code
uv run ruff format src

# Run tests
uv run pytest
```

## Requirements

- Python 3.9+
- PyTorch 2.0+
- Transformers 4.30+
- CUDA (optional, for GPU acceleration)

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

```bibtex
@misc{bert-hyperopt-comparison,
  title={BERT Hyperparameter Optimization Comparison},
  author={Levi},
  year={2025},
  url={https://github.com/your-username/Bert-Hyperopt-Comparison}
}
```

---

# BERT 超參數優化比較

針對 BERT 情感分析任務微調的超參數優化方法（遺傳演算法、粒子群優化、貝葉斯優化）綜合比較工具。

## 功能特色

- **三種優化方法**：並排比較 GA、PSO 和貝葉斯優化
- **真實資料集**：使用 IMDB 電影評論資料集（含合成資料備援）
- **完整分析**：收斂曲線、參數分佈和效能比較的詳細視覺化
- **現代工具鏈**：使用 `uv` 管理依賴項，`ruff` 進行程式碼檢查

## 專案結構

```
Bert-Hyperopt-Comparison/
├── src/
│   ├── __init__.py                  # 套件初始化
│   ├── bert_classifier.py           # BERT 模型與訓練流程
│   ├── experiment.py                # 實驗執行器
│   ├── visualize.py                 # 視覺化工具
│   └── optimizers/
│       ├── __init__.py              # 優化器套件
│       ├── bayesian_optimizer.py    # 貝葉斯優化
│       ├── ga_optimizer.py          # 遺傳演算法
│       └── pso_optimizer.py         # 粒子群優化
├── results/                         # 實驗結果 (JSON)
├── visualizations/                  # 生成的圖表
├── pyproject.toml                   # 專案配置
└── README.md
```

## 快速開始

### 安裝

```bash
# 複製專案
git clone https://github.com/your-username/Bert-Hyperopt-Comparison.git
cd Bert-Hyperopt-Comparison

# 使用 uv 安裝（推薦）
uv sync

# 或使用 pip 安裝
pip install -e .
```

### 執行實驗

```python
from src.experiment import HyperparameterOptimizationExperiment

# 使用小型資料集（1000 筆樣本）執行
experiment = HyperparameterOptimizationExperiment(data_size='small')
experiment.run_all_experiments()
```

### 生成視覺化圖表

```python
from src.visualize import ExperimentVisualizer

visualizer = ExperimentVisualizer()
visualizer.create_all_visualizations()
```

### 命令列執行

```bash
# 執行實驗
uv run python -m src.experiment

# 生成視覺化圖表
uv run python -m src.visualize
```

## 優化的超參數

| 參數 | 類型 | 範圍 | 說明 |
|------|------|------|------|
| `learning_rate` | 連續 | 1e-5 至 5e-5 | AdamW 優化器學習率 |
| `batch_size` | 離散 | [8, 16, 32] | 訓練批次大小 |
| `epochs` | 離散 | [2, 3, 4, 5] | 訓練週期數 |
| `dropout_rate` | 連續 | 0.1 至 0.5 | Dropout 機率 |
| `max_length` | 離散 | [64, 128, 256] | 最大序列長度 |
| `warmup_ratio` | 連續 | 0.1 至 0.5 | 學習率預熱比例（僅限貝葉斯） |

## 優化方法

### 遺傳演算法 (GA)
使用選擇、交叉和突變運算子演化超參數群體。

### 粒子群優化 (PSO)
粒子在搜尋空間中探索，同時分享已發現的良好解決方案。

### 貝葉斯優化
使用高斯過程建模目標函數，智慧選擇有潛力的超參數組合。

## 開發

```bash
# 安裝開發依賴
uv sync --dev

# 執行程式碼檢查
uv run ruff check src

# 格式化程式碼
uv run ruff format src

# 執行測試
uv run pytest
```

## 系統需求

- Python 3.9+
- PyTorch 2.0+
- Transformers 4.30+
- CUDA（選用，用於 GPU 加速）

## 授權

MIT 授權 - 詳見 [LICENSE](LICENSE)

## 引用

```bibtex
@misc{bert-hyperopt-comparison,
  title={BERT Hyperparameter Optimization Comparison},
  author={Levi},
  year={2025},
  url={https://github.com/your-username/Bert-Hyperopt-Comparison}
}
```
