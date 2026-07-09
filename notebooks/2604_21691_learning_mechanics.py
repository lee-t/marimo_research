# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "numpy",
#     "matplotlib",
# ]
# ///

import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    intro = r"""
    # There Will Be a Scientific Theory of Deep Learning

    Jamie Simon et al. argue that one route to a science of deep learning is to study
    simple limits that reveal fundamental behavior. A central example is the contrast
    between **lazy / kernel-like learning** and **rich feature learning**.

    This notebook tells one small story from that paper:

    > With the same tiny network budget, what changes when features are allowed to move?

    The example is a stylized **shipping-lane detection** problem. Positive points lie on
    two curved sea lanes plus a harbor hub. A model with frozen hidden features behaves like
    a simple fixed-basis method; the end-to-end model can reshape its internal features to
    follow the curved structure.
    """
    mo.md(intro)
    return


@app.cell(hide_code=True)
def _(mo):
    hidden_units = mo.ui.slider(
        start=8,
        stop=32,
        step=4,
        value=12,
        label="Hidden units",
    )
    label_noise = mo.ui.slider(
        start=0.08,
        stop=0.20,
        step=0.02,
        value=0.12,
        label="Observation noise",
    )
    mo.vstack([
        mo.md("## Controls"),
        hidden_units,
        label_noise,
    ])
    return hidden_units, label_noise


@app.cell(hide_code=True)
def _(mo, results):
    gap = 100 * (results["learned_test_acc"] - results["frozen_test_acc"])
    drift = results["learned_drift"]
    baseline = 100 * results["frozen_test_acc"]
    improved = 100 * results["learned_test_acc"]

    if gap >= 0:
        verdict = (
            f"The feature-learning model beats the frozen-feature proxy by {gap:.1f} "
            f"accuracy points on held-out data."
        )
    else:
        verdict = (
            f"On this setting the frozen-feature proxy is ahead by {-gap:.1f} "
            f"accuracy points, which is a useful reminder that richer dynamics do not guarantee a win."
        )

    mo.md(
        f"""
        ## What to look for

        - **Frozen hidden layer (lazy/kernel proxy):** the readout learns on top of a fixed random basis.
        - **End-to-end training (rich proxy):** the hidden layer moves, so the representation can align with the lanes.

        **Held-out accuracy**
        - Frozen hidden features: **{baseline:.1f}%**
        - Learned hidden features: **{improved:.1f}%**

        **Representation movement**
        - Mean hidden-feature drift for the learned model: **{drift:.2f}**
        - Mean hidden-feature drift for the frozen model: **0.00** by construction

        **Takeaway:** {verdict}
        """
    )
    return


@app.cell(hide_code=True)
def _(np, plt, results):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)

    grid_x = results["grid_x"]
    grid_y = results["grid_y"]
    X_train = results["X_train"]
    y_train = results["y_train"]

    panels = [
        (
            axes[0, 0],
            results["frozen_grid"],
            f"Frozen hidden features\nTest acc: {100 * results['frozen_test_acc']:.1f}%",
        ),
        (
            axes[0, 1],
            results["learned_grid"],
            f"Learned hidden features\nTest acc: {100 * results['learned_test_acc']:.1f}%",
        ),
    ]

    for ax, surface, title in panels:
        cf = ax.contourf(
            grid_x,
            grid_y,
            surface,
            levels=np.linspace(0, 1, 11),
            cmap="coolwarm",
            alpha=0.75,
        )
        ax.contour(grid_x, grid_y, surface, levels=[0.5], colors="black", linewidths=1.5)
        ax.scatter(
            X_train[y_train == 0, 0],
            X_train[y_train == 0, 1],
            s=18,
            c="#1f77b4",
            edgecolor="white",
            linewidth=0.25,
            alpha=0.8,
            label="negative",
        )
        ax.scatter(
            X_train[y_train == 1, 0],
            X_train[y_train == 1, 1],
            s=18,
            c="#d62728",
            edgecolor="white",
            linewidth=0.25,
            alpha=0.8,
            label="positive",
        )
        ax.set_title(title)
        ax.set_xlabel("x position")
        ax.set_ylabel("y position")
        ax.set_xlim(grid_x.min(), grid_x.max())
        ax.set_ylim(grid_y.min(), grid_y.max())

    axes[0, 0].legend(loc="upper right", frameon=True)
    cbar = fig.colorbar(cf, ax=axes[0, :], shrink=0.9)
    cbar.set_label("Predicted probability of lane / harbor")

    axes[1, 0].plot(results["history_epochs"], results["frozen_acc_history"], label="frozen", lw=2)
    axes[1, 0].plot(results["history_epochs"], results["learned_acc_history"], label="learned", lw=2)
    axes[1, 0].set_title("Held-out accuracy during training")
    axes[1, 0].set_xlabel("epoch")
    axes[1, 0].set_ylabel("accuracy")
    axes[1, 0].set_ylim(0.45, 1.0)
    axes[1, 0].grid(alpha=0.25)
    axes[1, 0].legend()

    axes[1, 1].plot(
        results["history_epochs"],
        np.zeros_like(results["learned_drift_history"]),
        label="frozen",
        lw=2,
    )
    axes[1, 1].plot(
        results["history_epochs"],
        results["learned_drift_history"],
        label="learned",
        lw=2,
    )
    axes[1, 1].set_title("How much the hidden representation moved")
    axes[1, 1].set_xlabel("epoch")
    axes[1, 1].set_ylabel("mean hidden-feature drift")
    axes[1, 1].grid(alpha=0.25)
    axes[1, 1].legend()

    fig
    return


@app.cell(hide_code=True)
def _():
    import matplotlib.pyplot as plt
    import numpy as np

    return np, plt


@app.cell(hide_code=True)
def _(np):
    def make_shipping_lanes(n_samples=500, noise=0.12, seed=0):
        rng = np.random.default_rng(seed)
        total = max(6000, n_samples * 12)
        points = rng.uniform(-2.3, 2.3, size=(total, 2))

        x = points[:, 0]
        y = points[:, 1]

        lane_1 = 0.85 * np.sin(1.35 * x) + 0.25
        lane_2 = -0.75 * np.sin(1.10 * x + 0.90) - 0.45
        dist_1 = np.abs(y - lane_1)
        dist_2 = np.abs(y - lane_2)
        harbor = np.sqrt((x + 0.35) ** 2 + (y - 0.05) ** 2)

        score = (
            np.exp(-(dist_1 / 0.24) ** 2)
            + 0.95 * np.exp(-(dist_2 / 0.21) ** 2)
            + 0.55 * np.exp(-(harbor / 0.45) ** 4)
        )
        margin = score - 0.92 + noise * rng.normal(size=total)
        labels = (margin > 0).astype(float)

        positive = points[labels == 1]
        negative = points[labels == 0]
        count = min(len(positive), len(negative), n_samples // 2)

        X = np.vstack([positive[:count], negative[:count]])
        y_out = np.concatenate([np.ones(count), np.zeros(count)])
        order = rng.permutation(len(X))
        return X[order], y_out[order]

    def sigmoid(logits):
        clipped = np.clip(logits, -50.0, 50.0)
        return 1.0 / (1.0 + np.exp(-clipped))

    def predict_proba(X_scaled, W, b, v, c):
        hidden = np.tanh(X_scaled @ W.T + b)
        return sigmoid(hidden @ v + c), hidden

    def train_story(hidden_units=12, noise=0.12, seed=2, epochs=320):
        X_train, y_train = make_shipping_lanes(n_samples=520, noise=noise, seed=seed)
        X_test, y_test = make_shipping_lanes(n_samples=520, noise=noise, seed=seed + 1)

        mean = X_train.mean(axis=0, keepdims=True)
        std = X_train.std(axis=0, keepdims=True) + 1e-6
        X_train_scaled = (X_train - mean) / std
        X_test_scaled = (X_test - mean) / std

        rng = np.random.default_rng(seed)
        W0 = rng.normal(scale=1.0 / np.sqrt(2), size=(hidden_units, 2))
        b0 = rng.normal(scale=0.15, size=hidden_units)
        v0 = rng.normal(scale=0.8 / np.sqrt(hidden_units), size=hidden_units)
        c0 = 0.0

        frozen_W = W0.copy()
        frozen_b = b0.copy()
        frozen_v = v0.copy()
        frozen_c = c0

        learned_W = W0.copy()
        learned_b = b0.copy()
        learned_v = v0.copy()
        learned_c = c0

        _, hidden_start = predict_proba(
            X_train_scaled, learned_W, learned_b, learned_v, learned_c
        )

        frozen_acc_history = []
        learned_acc_history = []
        learned_drift_history = []
        history_epochs = []
        checkpoint = 4
        frozen_lr = 0.40
        learned_lr = 0.16
        weight_decay = 1e-4

        for epoch in range(epochs):
            frozen_probs, frozen_hidden = predict_proba(
                X_train_scaled, frozen_W, frozen_b, frozen_v, frozen_c
            )
            frozen_delta = (frozen_probs - y_train) / len(X_train_scaled)
            frozen_v -= frozen_lr * (frozen_hidden.T @ frozen_delta + weight_decay * frozen_v)
            frozen_c -= frozen_lr * frozen_delta.sum()

            learned_probs, learned_hidden = predict_proba(
                X_train_scaled, learned_W, learned_b, learned_v, learned_c
            )
            learned_delta = (learned_probs - y_train) / len(X_train_scaled)
            grad_v = learned_hidden.T @ learned_delta + weight_decay * learned_v
            grad_c = learned_delta.sum()
            grad_hidden = learned_delta[:, None] * learned_v[None, :]
            grad_pre = grad_hidden * (1.0 - learned_hidden**2)
            grad_W = grad_pre.T @ X_train_scaled + weight_decay * learned_W
            grad_b = grad_pre.sum(axis=0)

            learned_W -= learned_lr * grad_W
            learned_b -= learned_lr * grad_b
            learned_v -= learned_lr * grad_v
            learned_c -= learned_lr * grad_c

            if epoch % checkpoint == 0 or epoch == epochs - 1:
                frozen_test_probs, _ = predict_proba(
                    X_test_scaled, frozen_W, frozen_b, frozen_v, frozen_c
                )
                learned_test_probs, learned_hidden_now = predict_proba(
                    X_test_scaled, learned_W, learned_b, learned_v, learned_c
                )
                frozen_pred = (frozen_test_probs >= 0.5).astype(float)
                learned_pred = (learned_test_probs >= 0.5).astype(float)

                frozen_acc_history.append((frozen_pred == y_test).mean())
                learned_acc_history.append((learned_pred == y_test).mean())
                train_hidden_now = np.tanh(X_train_scaled @ learned_W.T + learned_b)
                learned_drift_history.append(
                    np.mean(np.linalg.norm(train_hidden_now - hidden_start, axis=1))
                )
                history_epochs.append(epoch + 1)

        x1 = np.linspace(-2.3, 2.3, 220)
        x2 = np.linspace(-2.3, 2.3, 220)
        grid_x, grid_y = np.meshgrid(x1, x2)
        grid = np.column_stack([grid_x.ravel(), grid_y.ravel()])
        grid_scaled = (grid - mean) / std

        frozen_grid, _ = predict_proba(grid_scaled, frozen_W, frozen_b, frozen_v, frozen_c)
        learned_grid, _ = predict_proba(grid_scaled, learned_W, learned_b, learned_v, learned_c)
        frozen_test_probs, _ = predict_proba(X_test_scaled, frozen_W, frozen_b, frozen_v, frozen_c)
        learned_test_probs, learned_hidden_final = predict_proba(
            X_test_scaled, learned_W, learned_b, learned_v, learned_c
        )

        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
            "grid_x": grid_x,
            "grid_y": grid_y,
            "frozen_grid": frozen_grid.reshape(grid_x.shape),
            "learned_grid": learned_grid.reshape(grid_x.shape),
            "frozen_test_acc": ((frozen_test_probs >= 0.5).astype(float) == y_test).mean(),
            "learned_test_acc": ((learned_test_probs >= 0.5).astype(float) == y_test).mean(),
            "frozen_acc_history": np.array(frozen_acc_history),
            "learned_acc_history": np.array(learned_acc_history),
            "learned_drift_history": np.array(learned_drift_history),
            "history_epochs": np.array(history_epochs),
            "learned_drift": np.mean(
                np.linalg.norm(
                    np.tanh(X_train_scaled @ learned_W.T + learned_b) - hidden_start,
                    axis=1,
                )
            ),
        }

    return (train_story,)


@app.cell(hide_code=True)
def _(hidden_units, label_noise, train_story):
    results = train_story(
        hidden_units=hidden_units.value,
        noise=label_noise.value,
        seed=2,
        epochs=320,
    )
    return (results,)


if __name__ == "__main__":
    app.run()
