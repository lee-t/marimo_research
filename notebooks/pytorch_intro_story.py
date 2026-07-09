# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "torch",
#     "numpy",
#     "matplotlib",
# ]
# ///

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    intro = """
# Intro to PyTorch — the moment tensors become trainable

The source article is a broad introduction to PyTorch, covering tensors, autograd,
and a simple neural network. For a compact notebook, the clearest story is this one:

> **Tensors store numbers. Autograd turns those numbers into a model that can learn.**

To make that concrete, this notebook builds a tiny **waterfront apartment pricing** task.
A purely linear model can learn the obvious trends, but a small neural network can also
learn the sharp nonlinear premium for being very close to the water.

That gives us a clean PyTorch lesson:
- define tensors
- define a model
- define a loss
- call `.backward()`
- let autograd do the differentiation
"""
    mo.md(intro)
    return


@app.cell(hide_code=True)
def _(mo):
    market_nonlinearity = mo.ui.slider(
        start=0.4,
        stop=1.8,
        step=0.2,
        value=1.2,
        label="How nonlinear is the housing market?",
    )
    hidden_units = mo.ui.slider(
        start=4,
        stop=32,
        step=4,
        value=12,
        label="Hidden units in the neural net",
    )
    mo.vstack([
        mo.md("## Controls"),
        market_nonlinearity,
        hidden_units,
    ])
    return hidden_units, market_nonlinearity


@app.cell(hide_code=True)
def _(metrics, mo):
    gap = metrics["linear_mae"] - metrics["mlp_mae"]
    if gap > 0:
        verdict = (
            f"The neural network is better by **£{gap:.1f}k MAE** because it can bend toward "
            "the steep waterfront premium."
        )
    elif gap < 0:
        verdict = (
            f"The linear model is better by **£{-gap:.1f}k MAE** here; in this milder setting, "
            "extra flexibility is not buying much."
        )
    else:
        verdict = "Both models are essentially tied on this run."

    summary = f"""
## What happened

- **Linear model MAE:** **£{metrics['linear_mae']:.1f}k**
- **Neural net MAE:** **£{metrics['mlp_mae']:.1f}k**
- **Best model:** **{metrics['winner']}**

{verdict}

This is the PyTorch idea from the article in one picture: once the model and loss are
written as tensor operations, **autograd supplies the gradients automatically** for both
models — even though one is just a line and the other has a hidden layer.
"""
    mo.md(summary)
    return


@app.cell(hide_code=True)
def _(metrics, np, plt):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)

    actual = metrics["y_test"]
    linear_pred = metrics["linear_pred"]
    mlp_pred = metrics["mlp_pred"]
    lo = min(actual.min(), linear_pred.min(), mlp_pred.min())
    hi = max(actual.max(), linear_pred.max(), mlp_pred.max())

    axes[0, 0].scatter(actual, linear_pred, s=22, alpha=0.7, color="#1f77b4")
    axes[0, 0].plot([lo, hi], [lo, hi], "k--", lw=1.5)
    axes[0, 0].set_title(f"Linear model\nMAE £{metrics['linear_mae']:.1f}k")
    axes[0, 0].set_xlabel("Actual price (£k)")
    axes[0, 0].set_ylabel("Predicted price (£k)")
    axes[0, 0].grid(alpha=0.25)

    axes[0, 1].scatter(actual, mlp_pred, s=22, alpha=0.7, color="#d62728")
    axes[0, 1].plot([lo, hi], [lo, hi], "k--", lw=1.5)
    axes[0, 1].set_title(f"Neural net\nMAE £{metrics['mlp_mae']:.1f}k")
    axes[0, 1].set_xlabel("Actual price (£k)")
    axes[0, 1].set_ylabel("Predicted price (£k)")
    axes[0, 1].grid(alpha=0.25)

    axes[1, 0].plot(metrics["epochs"], metrics["linear_loss"], label="linear", lw=2)
    axes[1, 0].plot(metrics["epochs"], metrics["mlp_loss"], label="neural net", lw=2)
    axes[1, 0].set_title("Held-out MSE during training")
    axes[1, 0].set_xlabel("epoch")
    axes[1, 0].set_ylabel("MSE on standardized price")
    axes[1, 0].grid(alpha=0.25)
    axes[1, 0].legend()

    axes[1, 1].plot(
        metrics["waterfront_km"],
        metrics["true_curve"],
        label="true market",
        lw=2.5,
        color="black",
    )
    axes[1, 1].plot(
        metrics["waterfront_km"],
        metrics["linear_curve"],
        label="linear model",
        lw=2,
        color="#1f77b4",
    )
    axes[1, 1].plot(
        metrics["waterfront_km"],
        metrics["mlp_curve"],
        label="neural net",
        lw=2,
        color="#d62728",
    )
    axes[1, 1].set_title("How price changes with waterfront distance")
    axes[1, 1].set_xlabel("Distance to waterfront (km)")
    axes[1, 1].set_ylabel("Predicted price (£k)")
    axes[1, 1].grid(alpha=0.25)
    axes[1, 1].legend()

    fig
    return


@app.cell(hide_code=True)
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn as nn
    return nn, np, plt, torch


@app.cell(hide_code=True)
def _(nn, np, torch):
    def true_price_function(features, nonlinearity_strength):
        size = features[:, 0]
        age = features[:, 1]
        station = features[:, 2]
        waterfront = features[:, 3]
        floor = features[:, 4]
        school = features[:, 5]

        sea_bonus = 140.0 * np.exp(-waterfront / 0.65)
        commute_bonus = 55.0 / (1.0 + np.exp(2.4 * (station - 1.2)))
        view_bonus = (
            28.0
            / (1.0 + np.exp(-(floor - 7.0) / 1.6))
            * np.exp(-waterfront / 1.0)
        )
        family_bonus = 22.0 * np.tanh((size - 95.0) / 35.0) * (school / 100.0)

        return (
            120.0
            + 2.9 * size
            - 1.1 * age
            - 20.0 * station
            + 0.75 * school
            + nonlinearity_strength * (sea_bonus + commute_bonus + view_bonus + family_bonus)
        )

    def make_dataset(n_samples=900, nonlinearity_strength=1.2, seed=7):
        rng = np.random.default_rng(seed)
        size = rng.uniform(35.0, 185.0, n_samples)
        age = rng.uniform(0.0, 85.0, n_samples)
        station = rng.uniform(0.1, 4.5, n_samples)
        waterfront = rng.uniform(0.05, 6.0, n_samples)
        floor = rng.integers(0, 18, n_samples).astype(float)
        school = rng.uniform(20.0, 100.0, n_samples)

        X = np.column_stack([size, age, station, waterfront, floor, school]).astype(np.float32)
        clean_price = true_price_function(X, nonlinearity_strength)
        noisy_price = clean_price + rng.normal(0.0, 18.0, n_samples)
        return X, noisy_price.astype(np.float32)

    def standardize_train_test(X, y):
        order = np.arange(len(X))
        np.random.default_rng(0).shuffle(order)
        cutoff = int(0.8 * len(X))
        train_idx = order[:cutoff]
        test_idx = order[cutoff:]

        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        x_mean = X_train.mean(axis=0, keepdims=True)
        x_std = X_train.std(axis=0, keepdims=True) + 1e-6
        y_mean = y_train.mean()
        y_std = y_train.std() + 1e-6

        X_train_scaled = (X_train - x_mean) / x_std
        X_test_scaled = (X_test - x_mean) / x_std
        y_train_scaled = (y_train - y_mean) / y_std
        y_test_scaled = (y_test - y_mean) / y_std

        return {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "X_train_scaled": X_train_scaled.astype(np.float32),
            "X_test_scaled": X_test_scaled.astype(np.float32),
            "y_train_scaled": y_train_scaled.astype(np.float32),
            "y_test_scaled": y_test_scaled.astype(np.float32),
            "x_mean": x_mean.astype(np.float32),
            "x_std": x_std.astype(np.float32),
            "y_mean": float(y_mean),
            "y_std": float(y_std),
        }

    def train_model(model, X_train, y_train, X_test, y_test, lr, epochs):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        history_epochs = []
        history_loss = []

        for epoch in range(epochs):
            optimizer.zero_grad()
            pred_train = model(X_train)
            loss = loss_fn(pred_train, y_train)
            loss.backward()
            optimizer.step()

            if epoch % 4 == 0 or epoch == epochs - 1:
                with torch.no_grad():
                    test_loss = loss_fn(model(X_test), y_test).item()
                history_epochs.append(epoch + 1)
                history_loss.append(test_loss)

        return np.array(history_epochs), np.array(history_loss)

    def run_story(hidden_units=12, nonlinearity_strength=1.2, seed=7):
        X, y = make_dataset(
            n_samples=900,
            nonlinearity_strength=nonlinearity_strength,
            seed=seed,
        )
        data = standardize_train_test(X, y)

        X_train = torch.tensor(data["X_train_scaled"], dtype=torch.float32)
        X_test = torch.tensor(data["X_test_scaled"], dtype=torch.float32)
        y_train = torch.tensor(data["y_train_scaled"][:, None], dtype=torch.float32)
        y_test = torch.tensor(data["y_test_scaled"][:, None], dtype=torch.float32)

        torch.manual_seed(0)
        linear_model = nn.Linear(X_train.shape[1], 1)
        mlp_model = nn.Sequential(
            nn.Linear(X_train.shape[1], hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, 1),
        )

        epochs_linear, linear_loss = train_model(
            linear_model,
            X_train,
            y_train,
            X_test,
            y_test,
            lr=0.05,
            epochs=240,
        )
        epochs_mlp, mlp_loss = train_model(
            mlp_model,
            X_train,
            y_train,
            X_test,
            y_test,
            lr=0.03,
            epochs=240,
        )

        with torch.no_grad():
            linear_pred_scaled = linear_model(X_test).squeeze(-1).numpy()
            mlp_pred_scaled = mlp_model(X_test).squeeze(-1).numpy()

        y_mean = data["y_mean"]
        y_std = data["y_std"]
        y_test_raw = data["y_test"]
        linear_pred = linear_pred_scaled * y_std + y_mean
        mlp_pred = mlp_pred_scaled * y_std + y_mean

        linear_mae = float(np.mean(np.abs(linear_pred - y_test_raw)))
        mlp_mae = float(np.mean(np.abs(mlp_pred - y_test_raw)))
        winner = "neural net" if mlp_mae < linear_mae else "linear model"

        waterfront_km = np.linspace(0.05, 6.0, 180).astype(np.float32)
        probe = np.column_stack(
            [
                np.full_like(waterfront_km, 105.0),
                np.full_like(waterfront_km, 12.0),
                np.full_like(waterfront_km, 0.8),
                waterfront_km,
                np.full_like(waterfront_km, 9.0),
                np.full_like(waterfront_km, 82.0),
            ]
        ).astype(np.float32)
        probe_scaled = (probe - data["x_mean"]) / data["x_std"]
        probe_tensor = torch.tensor(probe_scaled, dtype=torch.float32)

        with torch.no_grad():
            linear_curve = linear_model(probe_tensor).squeeze(-1).numpy() * y_std + y_mean
            mlp_curve = mlp_model(probe_tensor).squeeze(-1).numpy() * y_std + y_mean

        true_curve = true_price_function(probe, nonlinearity_strength)

        return {
            "y_test": y_test_raw,
            "linear_pred": linear_pred,
            "mlp_pred": mlp_pred,
            "linear_mae": linear_mae,
            "mlp_mae": mlp_mae,
            "winner": winner,
            "epochs": epochs_linear,
            "linear_loss": linear_loss,
            "mlp_loss": mlp_loss,
            "waterfront_km": waterfront_km,
            "true_curve": true_curve,
            "linear_curve": linear_curve,
            "mlp_curve": mlp_curve,
            "autograd_note": "Both models use the same backward() call despite different complexity.",
        }

    return (run_story,)


@app.cell(hide_code=True)
def _(hidden_units, market_nonlinearity, run_story):
    metrics = run_story(
        hidden_units=hidden_units.value,
        nonlinearity_strength=market_nonlinearity.value,
        seed=7,
    )
    return (metrics,)


if __name__ == "__main__":
    app.run()
