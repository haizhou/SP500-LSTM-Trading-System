# S&P 500 Trading Decision System

An LSTM-based trading signal generator for the S&P 500 index, with emphasis on data integrity, leak-free time-series evaluation, and reproducible results.

## Project Overview

This project implements an end-to-end pipeline for automated trading decisions:

1. **Data Acquisition** - Fetch S&P 500 and VIX data from Yahoo Finance with retry logic
2. **Data Validation** - Schema checks, OHLC consistency, duplicate removal, outlier audit
3. **Data Storage** - SQLite database with composite primary keys
4. **Feature Engineering** - Technical indicators (returns, MA ratio, RSI, volatility, VIX level)
5. **Label Generation** - 3-class labels (Buy/Hold/Sell) from 5-day forward returns
6. **Model Training** - LSTM with class-weighted loss, AdamW, early stopping, gradient clipping
7. **Evaluation & Backtesting** - Classification metrics, multiple baselines, daily P&L simulation
8. **Additional Experiments** - Threshold sensitivity, architecture comparison, VIX regime analysis

## Project Structure

```
.
├── main.py              # Single entry point - runs complete pipeline
├── environment.yml      # Conda environment configuration
├── requirements.txt     # Python dependencies
├── README.md            # This file
├── .gitignore           # Git ignore rules
│
├── data/
│   └── data.db          # SQLite database (created on first run)
│
└── outputs/
    └── figures/         # Generated plots and CSV tables
```

## Installation

```bash
# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate daps-final
```

## Usage

```bash
# Run the complete pipeline
python main.py
```

The pipeline executes 9 steps:

1. **Data Acquisition** - Download or load cached data
2. **Feature Engineering** - Compute technical indicators
3. **Dataset Preparation** - Chronological split (70/15/15), normalize with train-only stats
4. **Model Training** - Train LSTM across 3 seeds
5. **Baseline Comparison** - Random, Always-Hold, Logistic Regression
6. **Backtesting** - Daily mark-to-market with transaction costs
7. **Results Summary** - Per-class metrics, cross-seed statistics
8. **Visualization** - Generate all figures
9. **Additional Experiments** - Threshold sensitivity, architecture comparison, VIX regime, backtest variants

## Model Architecture

The core model follows the assignment specification:

```
Input(6) → Linear(128) → LSTM(128) → LSTM(128) → Linear(128) → Output(3)
```

Implemented in `TradingLSTMBasic` (282K parameters).

A configurable variant `TradingLSTMConfigurable` is included for architecture comparison experiments, testing different layer counts and hidden sizes.

### Baselines

Three baselines contextualize LSTM performance:

| Baseline | Description |
|----------|-------------|
| Random | Uniform random predictions |
| Always-Hold | Predicts majority class |
| Logistic Regression | Last-day features, class-balanced |

### Feature Ablation

One-feature-drop ablation using Logistic Regression:
- Train full-feature model
- Remove each feature and retrain
- Report change in balanced accuracy

Evaluation uses validation set to avoid test leakage.

## Output

All outputs saved to `outputs/figures/`:

**Figures**
| File | Description |
|------|-------------|
| `training_curves.png` | Train/Val loss and accuracy |
| `confusion_matrix.png` | Test set confusion matrix |
| `label_distribution.png` | Buy/Hold/Sell class counts |
| `hidden_pca.png` | PCA of LSTM hidden states |
| `backtest.png` | Strategy vs Buy-and-Hold wealth |
| `ablation_logreg.png` | Feature importance bar chart |

**Tables**
| File | Description |
|------|-------------|
| `ablation_logreg.csv` | Feature ablation results |
| `threshold_sensitivity.csv` | Label threshold analysis |
| `architecture_comparison.csv` | LSTM variant comparison |
| `vix_regime_analysis.csv` | Accuracy by VIX level |
| `backtest_variants.csv` | Strategy variant returns |

## Key Design Decisions

- **Chronological splits** - No shuffling to prevent future leakage
- **Train-only normalization** - Scaler fitted on training data only
- **Class weighting** - Inverse frequency weights for imbalanced labels
- **Early stopping** - Patience=15 on validation loss
- **Multiple seeds** - 3 seeds for reproducibility assessment
- **Daily backtesting** - Position persistence with 5 bps transaction costs

## Dependencies

See `requirements.txt`. Key packages:

| Package | Purpose |
|---------|---------|
| yfinance | Data acquisition |
| pandas, numpy | Data processing |
| torch | Deep learning |
| scikit-learn | Metrics, preprocessing |
| matplotlib | Visualization |