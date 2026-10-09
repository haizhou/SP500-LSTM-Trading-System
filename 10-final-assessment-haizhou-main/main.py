"""
S&P 500 Trading Decision System

LSTM-based trading signal generation (Buy/Hold/Sell) with emphasis on
data integrity and reproducible evaluation.
"""

# pylint: disable=too-many-lines
# Reason: single-file submission recommended by assignment.
from __future__ import annotations

import os
import random
import time
import sqlite3
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler

import yfinance as yf

SEED = 42  # fix random seed
DB_PATH = "data/data.db"  # db path
OUTPUT_DIR = "outputs/figures"  # output dir

# Data acquisition configuration
START_DATE = "2020-01-01"  # from 2020 (COVID period)
END_DATE = "2024-12-31"  # to recent
SYMBOLS = ["^GSPC", "^VIX"]  # S&P 500 + VIX

# Label constants (centralized to avoid bugs)
LABEL_BUY = 0
LABEL_HOLD = 1
LABEL_SELL = 2
LABEL_NAMES = {LABEL_BUY: "Buy", LABEL_HOLD: "Hold", LABEL_SELL: "Sell"}

# Feature engineering configuration
LOOKBACK = 20  # past 20 days (~1 month)
FUTURE_DAYS = 5  # predict 5-day return
BUY_THRESH = 0.02  # >2% = Buy
SELL_THRESH = -0.02  # <-2% = Sell

# Model architecture configuration (as specified in assignment)
HIDDEN_SIZE = 128  # hidden size (from assignment)
BATCH_SIZE = 32  # batch size for training
LR = 0.0005  # lr set low for stability
EPOCHS = 100  # max 100 epochs (early stop)
NUM_SEEDS = 3  # number of random seeds for robustness

# Train/Val/Test split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def acquire_with_retry(
    symbol: str,
    start: str,
    end: str,
    conn: sqlite3.Connection | None = None,
    max_retries: int = 5,
    base_sleep: float = 1.0,
) -> pd.DataFrame:
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    """Fetch data with retry + robust fallback.
    If Yahoo Finance fails, we try the local SQLite cache. As a last resort
    (network + cache both unavailable), we generate deterministic synthetic
    data so the pipeline can still run end-to-end.
    """
    for attempt in range(1, max_retries + 1):
        # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start, end=end)
            if df is None or df.empty:
                raise ValueError(f"Empty download for {symbol}")
            df = df.reset_index()
            df["symbol"] = symbol
            df["date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
            df = df.rename(
                columns={
                    "Open": "open_price",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                }
            )
            return df[
                ["date", "symbol", "open_price", "high", "low", "close", "volume"]
            ]
        except Exception as e:  # pylint: disable=broad-exception-caught
            # yfinance can throw various exceptions
            sleep_s = base_sleep * (2 ** (attempt - 1))
            print(
                f"[WARN] yfinance failed for {symbol} (attempt {attempt}/{max_retries}): {e}"
            )
            if attempt < max_retries:
                time.sleep(sleep_s)

    # Fallback 1: use DB if available
    if conn is not None:
        try:
            df_db = load(conn, symbol)
            if len(df_db) > 0:
                print(f"[WARN] Using DB fallback for {symbol} (offline mode).")
                return df_db
        except Exception as e:  # pylint: disable=broad-exception-caught
            print(f"[WARN] DB fallback also failed for {symbol}: {e}")

    # Fallback 2: deterministic synthetic data (last resort)
    print(
        f"[WARN] Using SYNTHETIC fallback data for {symbol} due to network/DB failure."
    )
    dates = pd.date_range(start=start, end=end, freq="B")  # business days
    rng = np.random.default_rng(SEED + abs(hash(symbol)) % 10_000)
    price = 100.0 + np.cumsum(rng.normal(0, 0.5, size=len(dates)))
    close = np.maximum(price, 1.0)
    open_p = close * (1 + rng.normal(0, 0.002, size=len(dates)))
    high = np.maximum(open_p, close) * (
        1 + np.abs(rng.normal(0, 0.003, size=len(dates)))
    )
    low = np.minimum(open_p, close) * (
        1 - np.abs(rng.normal(0, 0.003, size=len(dates)))
    )
    volume = rng.integers(1_000_000, 5_000_000, size=len(dates))
    out = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "symbol": symbol,
            "open_price": open_p,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )
    return out


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Basic checks + OHLC consistency + simple outlier audit."""
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    n_before = len(df)

    # Remove duplicate entries for the same date
    df = df.drop_duplicates(subset=["date", "symbol"])
    n_after_dup = len(df)

    # Filter out invalid price data (prices must be positive)
    df = df[
        (df["close"] > 0) & (df["high"] > 0) & (df["low"] > 0) & (df["open_price"] > 0)
    ]

    # OHLC consistency: high should be >= max(open, close) and low <= min(open, close)
    max_oc = df[["open_price", "close"]].max(axis=1)
    min_oc = df[["open_price", "close"]].min(axis=1)
    ohlc_ok = (df["high"] >= max_oc) & (df["low"] <= min_oc)
    bad_ohlc = int((~ohlc_ok).sum())
    df = df[ohlc_ok]

    # Filter out invalid volume (must be non-negative)
    df = df[df["volume"] >= 0]

    # Simple outlier audit (do NOT drop by default, just report):
    # Flag days with extremely large absolute close-to-close jumps.
    if "close" in df.columns and len(df) > 30:
        ret = df["close"].pct_change().abs()
        # 99.5% percentile as a crude threshold
        thr = float(ret.quantile(0.995))
        n_flag = int((ret > thr).sum())
        print(f"    Outlier audit: |ret_1d| > {thr:.2%} flagged days = {n_flag}")

    # Remove any rows with missing values
    df = df.dropna()
    n_after = len(df)

    print(f"  Validation: {n_before} -> {n_after} records")
    print(f"    Duplicates removed: {n_before - n_after_dup}")
    print(f"    Bad OHLC rows removed: {bad_ohlc}")
    print(f"    Invalid records removed: {n_after_dup - n_after}")

    return df


def init_db(path: str) -> sqlite3.Connection:
    """Initialize SQLite database and create table."""
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS prices (
            date TEXT NOT NULL,
            symbol TEXT NOT NULL,
            open_price REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume INTEGER NOT NULL,
            PRIMARY KEY (date, symbol)
        )
    """
    )
    conn.commit()
    print(f"Database initialized: {path}")
    return conn


def store(conn: sqlite3.Connection, df: pd.DataFrame) -> None:
    """Save DataFrame to database."""
    for _, row in df.iterrows():
        conn.execute(
            "INSERT OR REPLACE INTO prices VALUES (?,?,?,?,?,?,?)",
            (
                row["date"],
                row["symbol"],
                row["open_price"],
                row["high"],
                row["low"],
                row["close"],
                row["volume"],
            ),
        )
    conn.commit()
    print(f"  Stored {len(df)} records to database")


def load(conn: sqlite3.Connection, symbol: str) -> pd.DataFrame:
    """Load symbol data from DB."""
    df = pd.read_sql_query(
        "SELECT * FROM prices WHERE symbol=? ORDER BY date", conn, params=(symbol,)
    )
    return df


def data_exists(conn: sqlite3.Connection, symbol: str) -> bool:
    """Check if symbol exists in DB."""
    cursor = conn.execute("SELECT COUNT(*) FROM prices WHERE symbol=?", (symbol,))
    return cursor.fetchone()[0] > 0


FEATURE_COLS = ["ret_1d", "ret_5d", "ma_ratio", "volatility", "rsi", "vix_level"]
"""List of feature column names used for model training.
Note: Removed vol_ratio (low signal), added vix_level (market fear indicator).
"""


def compute_features(df: pd.DataFrame, vix_df: pd.DataFrame = None) -> pd.DataFrame:
    """Generate technical indicators (MA, RSI, volatility) and merge with VIX."""
    df = df.copy()

    # Price returns (percentage change)
    df["ret_1d"] = df["close"].pct_change(1)
    df["ret_5d"] = df["close"].pct_change(5)

    # Next-day return for DAILY backtesting (t -> t+1), not used as model feature
    df["ret_next_1d"] = df["close"].shift(-1) / df["close"] - 1

    # MA ratio (short-term vs long-term trend)
    df["ma_5"] = df["close"].rolling(window=5).mean()
    df["ma_20"] = df["close"].rolling(window=20).mean()
    df["ma_ratio"] = df["ma_5"] / df["ma_20"]

    # Volatility (rolling standard deviation of daily returns)
    df["volatility"] = df["ret_1d"].rolling(window=10).std()

    # RSI - Relative Strength Index (momentum oscillator, >70 overbought, <30 oversold)
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    # VIX level (market fear indicator) - merge by date
    if vix_df is not None and len(vix_df) > 0:
        vix_df = vix_df[["date", "close"]].copy()
        vix_df = vix_df.rename(columns={"close": "vix_level"})
        df = df.merge(vix_df, on="date", how="left")
        # Fill missing VIX values with forward fill then backward fill
        missing = df["vix_level"].isna().mean()
        print(f"  VIX missing ratio before fill: {missing:.2%}")
        df["vix_level"] = df["vix_level"].ffill().bfill()
    else:
        # Fallback: use volatility as proxy
        df["vix_level"] = df["volatility"] * 100

    return df


def generate_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Create Buy/Hold/Sell labels from 5-day forward returns."""
    df = df.copy()

    # Calculate future return (this intentionally looks ahead for labels)
    df["future_ret"] = df["close"].shift(-FUTURE_DAYS) / df["close"] - 1

    # Assign labels based on thresholds using constants
    df["label"] = LABEL_HOLD  # Default: HOLD
    df.loc[df["future_ret"] > BUY_THRESH, "label"] = LABEL_BUY
    df.loc[df["future_ret"] < SELL_THRESH, "label"] = LABEL_SELL

    return df


class StockDataset(Dataset):
    """PyTorch dataset for LOOKBACK-day sequences."""

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        """Initialize dataset with features and labels."""
        self.features = features
        self.labels = labels

    def __len__(self) -> int:
        """Return number of sequences."""
        return len(self.features) - LOOKBACK + 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get sequence and label."""
        # Extract sequence of LOOKBACK days
        x = self.features[idx : idx + LOOKBACK]
        # Label corresponds to the last day of the sequence
        y = self.labels[idx + LOOKBACK - 1]
        return torch.FloatTensor(x), torch.LongTensor([y])[0]


def prepare_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Convert DataFrame to feature/label arrays."""
    # Remove rows with missing values
    df = df.dropna(subset=FEATURE_COLS + ["label"])

    features = df[FEATURE_COLS].values
    labels = df["label"].values.astype(int)

    return features, labels


def split_data(features: np.ndarray, labels: np.ndarray) -> Tuple:
    """Chronological split for time series (no shuffle to avoid leakage).
    Note: Sampling is not i.i.d. by design; for time series we prioritise
    leak-free evaluation over i.i.d. assumptions.
    """
    n = len(features)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

    train_feat = features[:train_end]
    train_lab = labels[:train_end]

    val_feat = features[train_end:val_end]
    val_lab = labels[train_end:val_end]

    test_feat = features[val_end:]
    test_lab = labels[val_end:]

    return (train_feat, train_lab), (val_feat, val_lab), (test_feat, test_lab)


def normalize_features(
    train_feat: np.ndarray, val_feat: np.ndarray, test_feat: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Standardize using train statistics, then apply to val/test.
    Args:
        train_feat: Training features.
        val_feat: Validation features.
        test_feat: Test features.
    Returns:
        Tuple of (normalized_train, normalized_val, normalized_test, scaler).
    """
    scaler = StandardScaler()

    # Fit ONLY on training data
    train_norm = scaler.fit_transform(train_feat)

    # Transform val and test using training statistics
    val_norm = scaler.transform(val_feat)
    test_norm = scaler.transform(test_feat)

    return train_norm, val_norm, test_norm, scaler


def make_last_day_dataset(ds: StockDataset) -> Tuple[np.ndarray, np.ndarray]:
    """Convert sequence dataset to tabular data using the last day of each window.

    This allows classic ML models (e.g. Logistic Regression) to be used as baselines.
    """
    x_list, y_list = [], []

    for item in ds:
        x_seq, y = item
        x_list.append(x_seq[-1].numpy())  # last timestep features
        y_list.append(int(y.item()))

    return np.asarray(x_list), np.asarray(y_list)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute core classification metrics for Buy/Hold/Sell."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def train_eval_logreg_baseline(
    train_ds: StockDataset,
    test_ds: StockDataset,
) -> Dict[str, float]:
    """Train and evaluate a Logistic Regression baseline.

    Uses the last day of each input sequence as features.
    """
    # Prepare tabular datasets
    x_train, ytr = make_last_day_dataset(train_ds)
    x_test, yte = make_last_day_dataset(test_ds)

    # Multinomial Logistic Regression with class balancing
    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=SEED,
    )

    clf.fit(x_train, ytr)
    pred = clf.predict(x_test)

    return compute_metrics(yte, pred)


class TradingLSTMBasic(nn.Module):
    # Assignment arch: Linear→LSTM→LSTM→Linear→Output
    """Strictly follows assignment architecture: Linear→LSTM→LSTM→Linear"""

    def __init__(self, input_size: int):
        super().__init__()
        self.linear1 = nn.Linear(input_size, 128)
        self.lstm1 = nn.LSTM(128, 128, batch_first=True)
        self.lstm2 = nn.LSTM(128, 128, batch_first=True)
        self.linear2 = nn.Linear(128, 128)
        self.output = nn.Linear(128, 3)  # 3 classes

    def forward(self, x: torch.Tensor, return_hidden: bool = False):
        """Forward pass.
        If return_hidden=True, also return the last-timestep hidden representation
        (before the final classifier) for visualisation (PCA/t-SNE).
        """
        x = self.linear1(x)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)

        h_last = x[:, -1, :]  # (B, 128) last timestep hidden state
        z = self.linear2(h_last)  # (B, 128)
        logits = self.output(z)  # (B, 3)

        if return_hidden:
            return logits, h_last
        return logits


class TradingLSTMConfigurable(nn.Module):
    """Configurable LSTM for architecture comparison experiments."""

    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.lstm_layers = nn.ModuleList(
            [
                nn.LSTM(hidden_size, hidden_size, batch_first=True)
                for _ in range(num_layers)
            ]
        )
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, 3)

    def forward(self, x: torch.Tensor, return_hidden: bool = False):
        """Forward pass with configurable architecture."""
        x = self.linear1(x)
        for lstm in self.lstm_layers:
            x, _ = lstm(x)
        h_last = x[:, -1, :]
        z = self.linear2(h_last)
        logits = self.output(z)
        if return_hidden:
            return logits, h_last
        return logits

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# pylint: disable=too-many-instance-attributes
class NaiveBaseline:
    """Baseline model that always predicts the most frequent class.
    This serves as a simple baseline to compare against the LSTM model.
    If our model can't beat this, it's not learning anything useful.
    """

    def __init__(self):
        self.pred_class = 1  # Default to HOLD

    def fit(self, labels: np.ndarray) -> None:
        """Find the most common class in training data."""
        unique, counts = np.unique(labels, return_counts=True)
        self.pred_class = unique[np.argmax(counts)]

    def predict(self, n: int) -> np.ndarray:
        """Predict the most common class for n samples."""
        return np.full(n, self.pred_class)


# pylint: disable=too-few-public-methods
class RandomBaseline:
    """Baseline model that predicts random classes.
    This represents the expected performance of random guessing,
    providing a lower bound for model performance.
    """

    def __init__(self, n_classes: int = 3):
        self.n_classes = n_classes

    def predict(self, n: int) -> np.ndarray:
        """Generate random predictions for n samples."""
        return np.random.randint(0, self.n_classes, n)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    return_hidden: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate model on test data.
    Args:
        model: Trained neural network model.
        loader: Test data loader.
        device: Device to run computations on.
    Returns:
        Tuple of (true_labels, predicted_labels) as numpy arrays.
    """
    model.eval()
    all_labels, all_preds, all_hidden = [], [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.cpu()
            if return_hidden:
                outputs = model(x, return_hidden=True)
                logits, hidden = outputs
                all_hidden.append(hidden.detach().cpu().numpy())
            else:
                logits = model(x)

            _, predicted = torch.max(logits, 1)
            all_labels.extend(y.numpy().tolist())
            all_preds.extend(predicted.cpu().numpy().tolist())

        hidden_np = (
            np.concatenate(all_hidden, axis=0)
            if (return_hidden and all_hidden)
            else np.empty((0, 128))
        )
    return np.array(all_labels), np.array(all_preds), hidden_np


def train_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    class_weights: torch.Tensor = None,
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """Train LSTM with AdamW, early stopping, and LR scheduler."""
    # pylint: disable=too-many-locals,too-many-statements
    model_cls = TradingLSTMBasic  # Assignment-compliant architecture
    model = model_cls(len(FEATURE_COLS)).to(device)

    # Use weighted loss if class weights provided
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()

    # AdamW optimizer with weight decay (L2 regularization)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # Training history
    history = {"loss": [], "acc": [], "val_loss": [], "val_acc": []}

    # Early stopping parameters
    best_val_loss = float("inf")
    patience = 15  # Wait 15 epochs before early stopping
    patience_counter = 0
    best_model_state = None

    # HARD TIME CUTOFF (15 minutes)
    start_time = time.time()
    max_seconds = 15 * 60

    # Training loop with early stopping
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)

        train_loss /= len(train_loader)
        train_acc = correct / total

        # Validation phase (using val_loader, NOT test_loader)
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                val_correct += (predicted == y).sum().item()
                val_total += y.size(0)

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total

        # Update history
        history["loss"].append(train_loss)
        history["acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"    Epoch {epoch+1}/{EPOCHS}: loss={train_loss:.4f}, acc={train_acc:.4f}, "
                f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, lr={current_lr:.6f}"
            )

        # Early stopping
        if patience_counter >= patience:
            print(f"    Early stopping at epoch {epoch+1}")
            break

        # HARD TIME CUTOFF
        if time.time() - start_time > max_seconds:
            print("    Time cutoff reached (15 minutes). Stopping training.")
            break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Return model and history (final evaluation done separately on test set)
    return model, history


# pylint: disable=too-many-locals
def feature_ablation_logreg(
    df: pd.DataFrame,
    feature_cols: List[str],
    output_csv: str,
    output_fig: str,
    eval_split: str = "val",  # NEW: "val" or "test"
) -> pd.DataFrame:
    """One-feature-drop ablation using Logistic Regression baseline.
    IMPORTANT: run ablations on VAL (default) to avoid peeking at the TEST set.
    """
    if eval_split not in {"val", "test"}:
        raise ValueError("eval_split must be 'val' or 'test'")

    # Prepare full dataset (drop missing)
    df2 = df.dropna(subset=feature_cols + ["label"]).copy()
    features = df2[feature_cols].values
    labels = df2["label"].values.astype(int)

    (train_x, train_y), (val_x, val_y), (test_x, test_y) = split_data(features, labels)

    # Normalise using ONLY train stats
    train_x, val_x, test_x, _ = normalize_features(train_x, val_x, test_x)

    # Choose evaluation set
    if eval_split == "val":
        eval_x, eval_y = val_x, val_y
    else:
        eval_x, eval_y = test_x, test_y

    def train_eval(x_train, ytr, x_eval, yev) -> float:
        clf = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=SEED,
        )
        clf.fit(x_train, ytr)
        pred = clf.predict(x_eval)
        return balanced_accuracy_score(yev, pred)

    base_score = train_eval(train_x, train_y, eval_x, eval_y)

    rows = []
    for f in feature_cols:
        keep = [c for c in feature_cols if c != f]
        dfk = df.dropna(subset=keep + ["label"]).copy()
        feats = dfk[keep].values
        labs = dfk["label"].values.astype(int)

        (tr_x, tr_y), (va_x, va_y), (te_x, te_y) = split_data(feats, labs)
        tr_x, va_x, te_x, _ = normalize_features(tr_x, va_x, te_x)

        xev, yev = (va_x, va_y) if eval_split == "val" else (te_x, te_y)
        score = train_eval(tr_x, tr_y, xev, yev)

        rows.append(
            {"dropped": f, "balanced_acc": score, "delta_vs_full": score - base_score}
        )

    out = pd.DataFrame(rows).sort_values("delta_vs_full")
    out.to_csv(output_csv, index=False)

    plt.figure(figsize=(10, 5))
    plt.bar(out["dropped"], out["delta_vs_full"])
    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel(f"Δ Balanced Accuracy vs Full (eval={eval_split})")
    plt.title("Feature Ablation (LogReg baseline)")
    plt.tight_layout()
    plt.savefig(output_fig, dpi=150)
    plt.close()
    return out


# ============================================================
# EXPERIMENT FUNCTIONS (for report tables)
# ============================================================


def threshold_sensitivity_analysis(
    df: pd.DataFrame,
    thresholds: List[float] = None,
) -> pd.DataFrame:
    """Test different threshold values for label generation (Table 2 in report)."""
    if thresholds is None:
        thresholds = [0.01, 0.015, 0.02, 0.025, 0.03]

    results = []
    for thresh in thresholds:
        # Generate labels with this threshold
        df_temp = df.copy()
        df_temp["future_ret"] = (
            df_temp["close"].shift(-FUTURE_DAYS) / df_temp["close"] - 1
        )
        df_temp["label"] = LABEL_HOLD
        df_temp.loc[df_temp["future_ret"] > thresh, "label"] = LABEL_BUY
        df_temp.loc[df_temp["future_ret"] < -thresh, "label"] = LABEL_SELL

        # Calculate class distribution
        df_clean = df_temp.dropna(subset=FEATURE_COLS + ["label"])
        labels = df_clean["label"].values.astype(int)
        total = len(labels)
        buy_pct = (labels == LABEL_BUY).sum() / total * 100
        hold_pct = (labels == LABEL_HOLD).sum() / total * 100
        sell_pct = (labels == LABEL_SELL).sum() / total * 100

        # Train LogReg and evaluate
        features = df_clean[FEATURE_COLS].values
        (tr_x, tr_y), (va_x, va_y), (te_x, _) = split_data(features, labels)
        tr_x, va_x, te_x, _ = normalize_features(tr_x, va_x, te_x)

        clf = LogisticRegression(
            max_iter=2000, class_weight="balanced", solver="lbfgs", random_state=SEED
        )
        clf.fit(tr_x, tr_y)
        pred = clf.predict(va_x)
        bal_acc = balanced_accuracy_score(va_y, pred)
        f1 = f1_score(va_y, pred, average="macro", zero_division=0)

        results.append(
            {
                "threshold": f"{thresh*100:.1f}%",
                "buy_pct": buy_pct,
                "hold_pct": hold_pct,
                "sell_pct": sell_pct,
                "bal_acc": bal_acc,
                "macro_f1": f1,
            }
        )

    return pd.DataFrame(results)


def architecture_comparison(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    class_weights: torch.Tensor,
) -> pd.DataFrame:
    # pylint: disable=too-many-locals
    """Compare different LSTM architectures (Table 4 in report)."""
    configs = [
        {"num_layers": 1, "hidden_size": 64},
        {"num_layers": 1, "hidden_size": 128},
        {"num_layers": 2, "hidden_size": 128},  # assignment spec
        {"num_layers": 2, "hidden_size": 256},
    ]

    results = []
    for cfg in configs:
        set_seed(SEED)
        model = TradingLSTMConfigurable(
            input_size=len(FEATURE_COLS),
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["num_layers"],
        ).to(device)

        n_params = model.count_parameters()
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

        # Quick training (fewer epochs for comparison)
        best_val_loss = float("inf")
        patience_counter = 0
        for _epoch in range(50):
            model.train()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    logits = model(x)
                    val_loss += criterion(logits, y).item()
            val_loss /= len(val_loader)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = model.state_dict().copy()
            else:
                patience_counter += 1
            if patience_counter >= 10:
                break

        best_state = {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        }
        model.load_state_dict(best_state)

        # Test evaluation
        model.eval()
        all_labels, all_preds = [], []
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                logits = model(x)
                _, predicted = torch.max(logits, 1)
                all_labels.extend(y.numpy().tolist())
                all_preds.extend(predicted.cpu().numpy().tolist())

        test_acc = accuracy_score(all_labels, all_preds)
        test_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

        results.append(
            {
                "variant": f"{cfg['num_layers']}-layer, h={cfg['hidden_size']}",
                "params": f"{n_params // 1000}K",
                "val_loss": best_val_loss,
                "test_acc": test_acc,
                "test_f1": test_f1,
            }
        )

    return pd.DataFrame(results)


def vix_regime_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    vix_values: np.ndarray,
) -> pd.DataFrame:
    """Analyze accuracy by VIX regime (Table 7 in report)."""
    results = []

    # Define VIX regimes
    regimes = [
        ("Low VIX (<15)", vix_values < 15),
        ("Mid VIX (15-20)", (vix_values >= 15) & (vix_values <= 20)),
        ("High VIX (>20)", vix_values > 20),
    ]

    for name, mask in regimes:
        if mask.sum() > 0:
            acc = accuracy_score(y_true[mask], y_pred[mask])
            results.append(
                {
                    "regime": name,
                    "samples": int(mask.sum()),
                    "accuracy": acc,
                }
            )

    return pd.DataFrame(results)


def backtest_confidence_weighted(
    model: nn.Module,
    test_loader: DataLoader,
    daily_returns: np.ndarray,
    device: torch.device,
    tc_bps: float = 5.0,
) -> Dict[str, float]:
    # pylint: disable=too-many-locals
    """Backtest with position size scaled by prediction confidence."""
    model.eval()
    all_probs = []

    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())

    probs = np.concatenate(all_probs, axis=0)
    preds = np.argmax(probs, axis=1)
    confidence = np.max(probs, axis=1)

    # Scale position by confidence
    # Buy -> +conf, Sell -> -conf, Hold -> 0
    pos = np.where(
        preds == LABEL_BUY, confidence, np.where(preds == LABEL_SELL, -confidence, 0)
    )

    # Align lengths
    n = min(len(pos), len(daily_returns))
    pos = pos[:n]
    returns = daily_returns[:n]

    # Transaction costs on position changes
    changes = np.abs(np.diff(pos, prepend=pos[0]))
    tc = (tc_bps / 10000.0) * changes

    strat_ret = pos * returns - tc
    wealth = np.cumprod(1 + strat_ret)

    # Metrics
    total_return = float(wealth[-1] - 1)
    running_max = np.maximum.accumulate(wealth)
    drawdown = (running_max - wealth) / running_max
    max_dd = float(np.max(drawdown))
    sharpe = 0.0
    if np.std(strat_ret) > 1e-12:
        sharpe = float(np.mean(strat_ret) / np.std(strat_ret) * np.sqrt(252))

    return {
        "total_return": total_return,
        "max_drawdown": max_dd,
        "sharpe_ratio": sharpe,
    }


# pylint: disable=too-many-arguments,too-many-positional-arguments
def backtest_filtered(
    model: nn.Module,
    test_loader: DataLoader,
    daily_returns: np.ndarray,
    device: torch.device,
    prob_threshold: float = 0.5,
    tc_bps: float = 5.0,
) -> Dict[str, float]:
    # pylint: disable=too-many-locals
    """Backtest only when prediction confidence exceeds threshold."""
    model.eval()
    all_probs = []

    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())

    probs = np.concatenate(all_probs, axis=0)
    preds = np.argmax(probs, axis=1)
    confidence = np.max(probs, axis=1)

    # Only trade when confident
    pos = np.zeros(len(preds))
    for i, (pred, conf) in enumerate(zip(preds, confidence)):
        if conf >= prob_threshold:
            if pred == LABEL_BUY:
                pos[i] = 1
            elif pred == LABEL_SELL:
                pos[i] = -1

    # Align lengths
    n = min(len(pos), len(daily_returns))
    pos = pos[:n]
    returns = daily_returns[:n]

    # Transaction costs
    changes = np.abs(np.diff(pos, prepend=pos[0]))
    tc = (tc_bps / 10000.0) * changes

    strat_ret = pos * returns - tc
    wealth = np.cumprod(1 + strat_ret)

    # Metrics
    total_return = float(wealth[-1] - 1)
    running_max = np.maximum.accumulate(wealth)
    drawdown = (running_max - wealth) / running_max
    max_dd = float(np.max(drawdown))
    sharpe = 0.0
    if np.std(strat_ret) > 1e-12:
        sharpe = float(np.mean(strat_ret) / np.std(strat_ret) * np.sqrt(252))

    return {
        "total_return": total_return,
        "max_drawdown": max_dd,
        "sharpe_ratio": sharpe,
    }


def backtest_persistent_positions(
    signals: np.ndarray, daily_returns: np.ndarray, tc_bps: float = 0.0
) -> Dict[str, float]:
    # pylint: disable=too-many-locals
    """
    Backtest with position persistence and daily mark-to-market.
    signals: 0=Buy, 1=Hold, 2=Sell
    daily_returns: aligned daily returns (same length as signals),
                   e.g. next-day returns ret_{t+1}
    tc_bps: transaction cost in basis points per position change
            (e.g. 5 = 0.05%)
    """

    # Sanity check: signals and returns must be aligned in time
    assert len(signals) == len(daily_returns)

    # Map discrete trading signals to target positions
    #   Buy  -> +1 (long)
    #   Sell -> -1 (short)
    #   Hold ->  0 (flat / no position)
    target_pos = np.where(
        signals == LABEL_BUY,
        1,
        np.where(signals == LABEL_SELL, -1, 0),
    )

    # Build actual position time series with persistence
    # Once a position is entered, it is held until a new signal
    # indicates a different target position
    pos = np.zeros_like(target_pos, dtype=float)

    # Initialize position on the first day
    pos[0] = target_pos[0]

    for t in range(1, len(target_pos)):
        # pylint: disable=too-many-locals
        # By default, keep yesterday's position
        pos[t] = pos[t - 1]

        # Change position only if the target position changes
        if target_pos[t] != pos[t - 1]:
            pos[t] = target_pos[t]

    # A cost is incurred whenever the position changes
    # Absolute position change (0 = no trade, 1 or 2 = trade)
    changes = np.abs(np.diff(pos, prepend=pos[0]))

    # Convert basis points to decimal cost
    tc = (tc_bps / 10000.0) * changes

    # Compute daily strategy returns - position * market return - transaction cost
    strat_ret = pos * daily_returns - tc

    # Compute cumulative wealth curve
    wealth = np.cumprod(1 + strat_ret)
    total_return = wealth[-1] - 1

    # Risk metrics: maximum drawdown
    running_max = np.maximum.accumulate(wealth)
    drawdown = (running_max - wealth) / running_max
    max_drawdown = float(np.max(drawdown))

    # Risk-adjusted return: annualised Sharpe ratio
    sharpe = 0.0
    if len(strat_ret) > 1 and np.std(strat_ret) > 1e-12:
        sharpe = float(np.mean(strat_ret) / np.std(strat_ret) * np.sqrt(252))

    # Trading statistics - number of trades equals number of non-zero position changes
    n_trades = int(np.sum(changes > 0))

    # Win rate computed over days when a trade occurred
    win_rate = 0.0
    trade_days = strat_ret[changes > 0]
    if len(trade_days) > 0:
        win_rate = float(np.mean(trade_days > 0))

    # Buy-and-hold benchmark (always fully invested long)
    buy_hold = np.cumprod(1 + daily_returns)[-1] - 1

    # Buy-and-hold risk metrics (match strategy definitions)
    bh_wealth = np.cumprod(1 + daily_returns)

    bh_running_max = np.maximum.accumulate(bh_wealth)
    bh_drawdown = (bh_running_max - bh_wealth) / bh_running_max
    bh_max_drawdown = float(np.max(bh_drawdown))

    bh_sharpe = 0.0
    if len(daily_returns) > 1 and np.std(daily_returns) > 1e-12:
        bh_sharpe = float(np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252))

        # Return all backtest metrics
    return {
        "total_return": float(total_return),
        "buy_hold_return": float(buy_hold),
        "max_drawdown": max_drawdown,
        "buy_hold_max_drawdown": bh_max_drawdown,
        "sharpe_ratio": sharpe,
        "buy_hold_sharpe_ratio": bh_sharpe,
        "n_trades": n_trades,
        "win_rate": win_rate,
        "wealth": wealth,
        "buy_hold_wealth": bh_wealth,
    }


def plot_backtest_from_wealth(
    wealth: np.ndarray, buy_hold_wealth: np.ndarray, path: str
) -> None:
    """Plot backtest wealth curves (consistent with metrics)."""
    plt.figure(figsize=(12, 6))
    plt.plot(wealth, linewidth=1.5, label="Strategy")
    plt.plot(buy_hold_wealth, linestyle="--", linewidth=1.5, label="Buy & Hold")
    plt.axhline(y=1, color="gray", linestyle=":", alpha=0.5)
    plt.xlabel("Trading Days")
    plt.ylabel("Cumulative Wealth (1 = initial)")
    plt.title("Backtest: Strategy vs Buy & Hold")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, path: str) -> None:
    """Plot and save confusion matrix heatmap using matplotlib only.
    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        path: File path to save the figure.
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar()

    labels = ["Buy", "Hold", "Sell"]
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)

    # Add counts inside cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix - LSTM Trading Model")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_hidden_pca_2d(hidden: np.ndarray, labels: np.ndarray, path: str) -> None:
    """Visualise LSTM hidden representations with 2D PCA."""
    pca = PCA(n_components=2, random_state=SEED)
    z = pca.fit_transform(hidden)

    plt.figure(figsize=(8, 6))
    for c, name in LABEL_NAMES.items():
        mask = labels == c
        if np.any(mask):
            plt.scatter(z[mask, 0], z[mask, 1], s=18, alpha=0.7, label=name)

    var = pca.explained_variance_ratio_.sum() * 100
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"LSTM Hidden State (PCA 2D, explained={var:.1f}%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_training_curves(history: Dict, path: str) -> None:
    """Plot training and validation loss/accuracy curves.
    Args:
        history: Dictionary with 'loss', 'acc', 'val_loss', 'val_acc' lists.
        path: File path to save the figure.
    """
    _, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["loss"]) + 1)

    # Loss curve
    axes[0].plot(epochs, history["loss"], "b-", linewidth=1.5, label="Train Loss")
    if "val_loss" in history and len(history["val_loss"]) > 0:
        axes[0].plot(
            epochs, history["val_loss"], "r--", linewidth=1.5, label="Val Loss"
        )
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy curve
    axes[1].plot(epochs, history["acc"], "b-", linewidth=1.5, label="Train Acc")
    if "val_acc" in history and len(history["val_acc"]) > 0:
        axes[1].plot(epochs, history["val_acc"], "r--", linewidth=1.5, label="Val Acc")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training and Validation Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_label_dist(labels: np.ndarray, path: str) -> None:
    """Plot distribution of trading labels.
    Args:
        labels: Array of labels.
        path: File path to save the figure.
    """
    unique, counts = np.unique(labels, return_counts=True)
    names = ["Buy", "Hold", "Sell"]
    colors = ["green", "gray", "red"]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(
        [names[i] for i in unique], counts, color=[colors[i] for i in unique]
    )

    # Add count labels on bars
    for bar_rect, count in zip(bars, counts):
        plt.text(
            bar_rect.get_x() + bar_rect.get_width() / 2,
            bar_rect.get_height() + 10,
            str(count),
            ha="center",
            va="bottom",
        )

    plt.xlabel("Trading Signal")
    plt.ylabel("Count")
    plt.title("Distribution of Trading Labels")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    """Run the full pipeline."""
    # pylint: disable=too-many-locals,too-many-statements
    # Initialize
    set_seed(SEED)
    os.makedirs("data", exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ===== Step 1: Data Acquisition =====
    print("\n[Step 1] Data Acquisition")
    print("-" * 25)
    conn = init_db(DB_PATH)

    for symbol in SYMBOLS:
        if data_exists(conn, symbol):
            print(f"  {symbol}: Already in database, skipping download")
        else:
            records = acquire_with_retry(symbol, START_DATE, END_DATE, conn=conn)
            records = validate(records)
            store(conn, records)

    # Load main dataset (S&P 500) and VIX
    df = load(conn, "^GSPC")
    vix_df = load(conn, "^VIX")
    print(f"  Loaded {len(vix_df)} VIX records")

    # ===== Step 2: Feature Engineering =====
    print("\n[Step 2] Feature Engineering")
    print("-" * 25)
    df = compute_features(df, vix_df)  # Now uses VIX data
    df = generate_labels(df)
    print(f"  Features: {FEATURE_COLS}")
    print(f"  Lookback window: {LOOKBACK} days")
    print(f"  Prediction horizon: {FUTURE_DAYS} days")

    # ===== Step 3: Dataset Preparation =====
    print("\n[Step 3] Dataset Preparation")
    print("-" * 25)

    # Prepare raw features and labels
    features, labels = prepare_data(df)

    # Split chronologically: 70% train, 15% val, 15% test
    (train_feat, train_lab), (val_feat, val_lab), (test_feat, test_lab) = split_data(
        features, labels
    )

    #  Normalize using ONLY training statistics to prevent leakage
    train_feat, val_feat, test_feat, _ = normalize_features(
        train_feat, val_feat, test_feat
    )

    # Create datasets
    train_ds = StockDataset(train_feat, train_lab)
    val_ds = StockDataset(val_feat, val_lab)
    test_ds = StockDataset(test_feat, test_lab)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    print(f"  Training samples: {len(train_ds)}")
    print(f"  Validation samples: {len(val_ds)}")
    print(f"  Test samples: {len(test_ds)}")
    print("  Note: Scaler fit ONLY on training data (no leakage)")

    # Calculate class weights to handle imbalance
    class_counts = np.bincount(train_lab, minlength=3)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    class_weights = torch.FloatTensor(class_weights)
    print(
        f"  Class weights: Buy={class_weights[0]:.2f}, "
        f"Hold={class_weights[1]:.2f}, Sell={class_weights[2]:.2f}"
    )

    # ===== Step 4: Model Training (Multiple Seeds) =====
    print("\n[Step 4] Model Training")
    print("-" * 25)
    all_metrics = []
    last_history = None
    last_y_true = None
    last_y_pred = None
    last_hidden = None
    last_model = None

    for i in range(NUM_SEEDS):
        seed = SEED + i
        print(f"\n  Training with seed {seed} ({i+1}/{NUM_SEEDS}):")
        set_seed(seed)

        # Train using val_loader for early stopping (NOT test_loader)
        model, history = train_model(
            train_loader, val_loader, device, class_weights=class_weights
        )
        last_history = history
        last_model = model

        # Final evaluation on held-out TEST set
        y_true, y_pred, hidden = evaluate(
            model, test_loader, device, return_hidden=True
        )
        metrics = compute_metrics(y_true, y_pred)
        all_metrics.append(metrics)
        last_y_true = y_true
        last_y_pred = y_pred
        last_hidden = hidden

        print(
            f"    Test Results: Acc={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}"
        )

    # ===== Step 5: Baseline Comparison =====
    print("\n[Step 5] Baseline Comparison")
    print("-" * 25)

    test_labels = np.array(
        [test_ds[i][1].item() for i in range(len(test_ds))]
    )  # Align with dataset indexing

    # Naive baseline (most frequent class)
    naive = NaiveBaseline()
    naive.fit(train_lab)
    naive_pred = naive.predict(len(test_labels))
    naive_metrics = compute_metrics(test_labels, naive_pred)
    print(
        f"  Naive Baseline: Acc={naive_metrics['accuracy']:.4f}, F1={naive_metrics['f1']:.4f}"
    )

    # Random baseline
    set_seed(SEED)
    random_baseline = RandomBaseline()
    random_pred = random_baseline.predict(len(test_labels))
    random_metrics = compute_metrics(test_labels, random_pred)
    print(
        f"  Random Baseline: Acc={random_metrics['accuracy']:.4f}, F1={random_metrics['f1']:.4f}"
    )
    # Logistic Regression baseline (strong classical baseline)
    logreg_metrics = train_eval_logreg_baseline(train_ds, test_ds)
    print(
        "  Logistic Regression Baseline: "
        f"Acc={logreg_metrics['accuracy']:.4f}, "
        f"BalAcc={logreg_metrics['balanced_accuracy']:.4f}, "
        f"F1={logreg_metrics['f1']:.4f}"
    )
    # ===== Step 6: Backtesting =====
    print("\n[Step 6] Backtesting (Daily Mark-to-Market)")
    print("-" * 25)

    # Build a clean frame aligned with features/labels and daily next-day returns
    df_clean = df.dropna(subset=FEATURE_COLS + ["label", "ret_next_1d"]).copy()

    # Split indices consistent with split_data() on the cleaned rows
    n = len(df_clean)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

    # Our StockDataset returns labels starting at index (LOOKBACK-1) within each split.
    # For the TEST split, predictions correspond to df_clean rows starting at:
    # test_start = val_end + (LOOKBACK - 1)
    test_start_idx = val_end + (LOOKBACK - 1)

    test_daily_returns = df_clean["ret_next_1d"].values[
        test_start_idx : test_start_idx + len(last_y_pred)
    ]

    assert len(test_daily_returns) == len(
        last_y_pred
    ), f"Mismatch: returns={len(test_daily_returns)} preds={len(last_y_pred)}"
    assert not np.isnan(test_daily_returns).any(), "test_daily_returns contains NaN"

    # Run backtest with position persistence and transaction costs (example: 5 bps)
    backtest_results = backtest_persistent_positions(
        signals=last_y_pred,
        daily_returns=test_daily_returns,
        tc_bps=5.0,
    )

    print(f"  Strategy Total Return: {backtest_results['total_return']:.2%}")
    print(f"  Buy&Hold Total Return: {backtest_results['buy_hold_return']:.2%}")
    print(
        f"  Max Drawdown: Strategy={backtest_results['max_drawdown']:.2%}, "
        f"Buy&Hold={backtest_results['buy_hold_max_drawdown']:.2%}"
    )
    print(
        f"  Sharpe Ratio: Strategy={backtest_results['sharpe_ratio']:.2f}, "
        f"Buy&Hold={backtest_results['buy_hold_sharpe_ratio']:.2f}"
    )
    print(f"  Win Rate (trade days): {backtest_results['win_rate']:.2%}")
    print(f"  Number of Trades: {backtest_results['n_trades']}")

    plot_backtest_from_wealth(
        backtest_results["wealth"],
        backtest_results["buy_hold_wealth"],
        f"{OUTPUT_DIR}/backtest.png",
    )

    # ===== Step 7: Results Summary =====
    print("\n[Step 7] Results Summary")
    print("-" * 25)
    print("\n  Per-class performance (last run):")
    print(
        classification_report(
            last_y_true,
            last_y_pred,
            target_names=["Buy", "Hold", "Sell"],
            digits=3,
            zero_division=0,
        )
    )

    # Compute statistics across seeds
    print("\n  LSTM Model (mean ± std across seeds):")
    for metric in ["accuracy", "precision", "recall", "f1"]:
        values = [m[metric] for m in all_metrics]
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f"    {metric.capitalize():12s}: {mean_val:.4f} ± {std_val:.4f}")

    print("\n  Baseline Comparison:")
    mean_acc = np.mean([m["accuracy"] for m in all_metrics])
    print(f"    LSTM Accuracy:   {mean_acc:.4f}")
    print(f"    Naive Accuracy:  {naive_metrics['accuracy']:.4f}")
    print(f"    Random Accuracy: {random_metrics['accuracy']:.4f}")

    # ===== Step 8: Visualization =====
    print("\n[Step 8] Generating Visualizations")
    print("-" * 25)

    plot_training_curves(last_history, f"{OUTPUT_DIR}/training_curves.png")
    plot_confusion(last_y_true, last_y_pred, f"{OUTPUT_DIR}/confusion_matrix.png")
    plot_hidden_pca_2d(last_hidden, last_y_true, f"{OUTPUT_DIR}/hidden_pca.png")
    plot_label_dist(labels, f"{OUTPUT_DIR}/label_distribution.png")
    _ = feature_ablation_logreg(
        df=df,
        feature_cols=FEATURE_COLS,
        output_csv=f"{OUTPUT_DIR}/ablation_logreg.csv",
        output_fig=f"{OUTPUT_DIR}/ablation_logreg.png",
        eval_split="val",
    )

    print(
        f"  Generated 6 outputs (5 figures + 1 table). Saved to: {os.path.abspath(OUTPUT_DIR)}"
    )

    # ===== Step 9: Additional Experiments (for report tables) =====
    print("\n[Step 9] Additional Experiments")
    print("-" * 25)

    # 9.1 Threshold Sensitivity Analysis (Table 2)
    print("\n  9.1 Threshold Sensitivity Analysis:")
    thresh_results = threshold_sensitivity_analysis(df)
    print(thresh_results.to_string(index=False))
    thresh_results.to_csv(f"{OUTPUT_DIR}/threshold_sensitivity.csv", index=False)

    # 9.2 Architecture Comparison (Table 4)
    print("\n  9.2 Architecture Comparison:")
    arch_results = architecture_comparison(
        train_loader, val_loader, test_loader, device, class_weights
    )
    print(arch_results.to_string(index=False))
    arch_results.to_csv(f"{OUTPUT_DIR}/architecture_comparison.csv", index=False)

    # 9.3 VIX Regime Analysis (Table 7)
    print("\n  9.3 VIX Regime Analysis:")
    # Get VIX values for test set
    test_vix = df_clean["vix_level"].values[
        test_start_idx : test_start_idx + len(last_y_pred)
    ]
    vix_results = vix_regime_analysis(last_y_true, last_y_pred, test_vix)
    print(vix_results.to_string(index=False))
    vix_results.to_csv(f"{OUTPUT_DIR}/vix_regime_analysis.csv", index=False)

    # 9.4 Backtesting Variants (Table 8)
    print("\n  9.4 Backtesting Variants:")
    conf_weighted = backtest_confidence_weighted(
        last_model, test_loader, test_daily_returns, device, tc_bps=5.0
    )
    filtered = backtest_filtered(
        last_model,
        test_loader,
        test_daily_returns,
        device,
        prob_threshold=0.5,
        tc_bps=5.0,
    )
    backtest_variants = pd.DataFrame(
        [
            {
                "strategy": "Full Position",
                "return": backtest_results["total_return"],
                "max_drawdown": backtest_results["max_drawdown"],
                "sharpe": backtest_results["sharpe_ratio"],
            },
            {
                "strategy": "Confidence-Weighted",
                "return": conf_weighted["total_return"],
                "max_drawdown": conf_weighted["max_drawdown"],
                "sharpe": conf_weighted["sharpe_ratio"],
            },
            {
                "strategy": "Filtered (prob>0.5)",
                "return": filtered["total_return"],
                "max_drawdown": filtered["max_drawdown"],
                "sharpe": filtered["sharpe_ratio"],
            },
            {
                "strategy": "Buy & Hold",
                "return": backtest_results["buy_hold_return"],
                "max_drawdown": backtest_results["buy_hold_max_drawdown"],
                "sharpe": backtest_results["buy_hold_sharpe_ratio"],
            },
        ]
    )
    print(backtest_variants.to_string(index=False))
    backtest_variants.to_csv(f"{OUTPUT_DIR}/backtest_variants.csv", index=False)

    print(f"\n  Experiment results saved to: {os.path.abspath(OUTPUT_DIR)}")

    # Cleanup
    conn.close()

    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")


if __name__ == "__main__":
    main()
