# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "matplotlib",
#     "numpy",
# ]
# ///

import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    return mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    header = mo.Html(
        """
        <style>
        .jumper-header {
            padding: 24px 28px;
            border-radius: 16px;
            color: #f7f3ea;
            background: linear-gradient(135deg, #173b47 0%, #28606a 55%, #b96d3b 100%);
            box-shadow: 0 10px 30px rgba(23, 59, 71, 0.18);
            margin-bottom: 18px;
        }
        .jumper-header h1 {
            margin: 0 0 8px 0;
            font-size: 28px;
            letter-spacing: -0.03em;
        }
        .jumper-header p {
            margin: 0;
            max-width: 760px;
            line-height: 1.55;
            font-size: 15px;
        }
        .jumper-note {
            border-left: 4px solid #b96d3b;
            padding: 10px 14px;
            background: rgba(185, 109, 59, 0.08);
            border-radius: 4px;
        }
        @media (prefers-color-scheme: dark) {
            .jumper-note {
                background: rgba(185, 109, 59, 0.16);
            }
        }
        </style>
        <div class="jumper-header">
            <h1>Let the backbone move; let the side chains equilibrate</h1>
            <p>
                A visual companion to John Jumper's 2017 dissertation:
                replace many fast, discrete side-chain rotamers with one
                backbone free-energy surface.
            </p>
        </div>
        """
    )
    header
    return


@app.cell(hide_code=True)
def _(mo):
    explanation = mo.md(
        """
        ## The idea in one experiment

        Imagine a small protein hinge with a single slow backbone coordinate
        `q`: negative values are open and positive values are closed. Each of
        seven residues also has three possible side-chain rotamers `s`.
        The full energy is `E(q, s)`.

        Upside's key move is to avoid dragging every `s` through the
        backbone dynamics. For each fixed `q`, it computes the side-chain
        free energy

        `F(q) = -T log sum_s exp(-E(q, s) / T)`.

        This is not the same as keeping one favorite rotamer: the sum keeps
        both packing interactions and the entropy of many possible rotamers.
        The experiment below compares a sampler that moves on `F(q)` with a
        sampler that explicitly carries all rotamers.

        <div class="jumper-note">
        <strong>Read the story from left to right:</strong>
        the free-energy curve, the equilibrium distribution it implies, the
        two trajectories that target that same distribution, and the number
        of rotamer combinations contributing at each backbone position.
        </div>

        This is a deliberately small teaching model, not a calibrated protein
        force field. Its purpose is to make the statistical-mechanics move
        visible on a CPU.
        """
    )
    explanation
    return


@app.cell(hide_code=True)
def _(mo):
    temperature = mo.ui.slider(
        start=0.35,
        stop=1.35,
        step=0.05,
        value=0.75,
        label="Temperature T",
    )
    coupling = mo.ui.slider(
        start=0.0,
        stop=2.0,
        step=0.1,
        value=1.0,
        label="Side-chain packing coupling",
    )
    barrier = mo.ui.slider(
        start=0.8,
        stop=2.8,
        step=0.1,
        value=1.6,
        label="Backbone barrier",
    )

    controls = mo.vstack(
        [
            mo.md("### Experiment controls"),
            mo.hstack([temperature, coupling, barrier], justify="start", gap=1.5),
        ]
    )
    controls
    return barrier, coupling, temperature


@app.cell(hide_code=True)
def _(barrier, coupling, run_experiment, temperature):
    results = run_experiment(
        temperature=temperature.value,
        coupling=coupling.value,
        barrier=barrier.value,
        seed=7,
    )
    return (results,)


@app.cell(hide_code=True)
def _(mo, results):
    free_iat = results["free_iat"]
    joint_iat = results["joint_iat"]
    mixing_ratio = joint_iat / free_iat
    closed_mass = 100 * results["exact_probability"][results["q"] > 0].sum()
    max_effective_rotamers = results["effective_rotamers"].max()
    free_tv = 100 * results["free_total_variation"]
    joint_tv = 100 * results["joint_total_variation"]

    if mixing_ratio > 1.05:
        mixing_sentence = (
            f"The explicit sampler's backbone autocorrelation time is "
            f"{mixing_ratio:.1f}x longer in this setting."
        )
    else:
        mixing_sentence = (
            "The two samplers mix similarly in this setting; increase packing "
            "coupling to make the smoothing benefit more pronounced."
        )

    mo.md(
        f"""
        ### What to look for

        - **Thermodynamics is preserved:** both histograms should follow the
          black exact marginal distribution. Their total-variation errors are
          {free_tv:.1f}% (free-energy sampler) and {joint_tv:.1f}% (explicit
          sampler).
        - **The shortcut is dynamical:** {mixing_sentence} The smoother
          backbone-only chain can cross the hinge landscape without waiting
          for a particular rotamer assignment to rearrange.
        - **Entropy matters:** the equilibrium ensemble puts about
          **{closed_mass:.0f}%** of its weight in the closed half, while as
          many as           **{max_effective_rotamers:.0f} effective rotamer combinations**
          contribute at one backbone position. Taking only the lowest-energy
          rotamer would discard that information.

        **Paper connection:** this is the essence of integrating out side
        chains into an adiabatic free energy, the coarse-grained move used in
        the Upside model to reduce steric rattling while retaining side-chain
        energetics.
        """
    )
    return


@app.cell(hide_code=True)
def _(np, plt, results):
    q = results["q"]
    free_energy = results["free_energy"]
    minimum_energy = results["minimum_energy"]
    free_histogram = results["free_histogram"]
    joint_histogram = results["joint_histogram"]
    exact_probability = results["exact_probability"]
    trace_length = min(2500, len(results["free_trace"]))
    trace_steps = np.arange(trace_length)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    colors = {
        "teal": "#28606a",
        "orange": "#b96d3b",
        "ink": "#173b47",
        "sand": "#d6b58a",
        "slate": "#66727a",
    }

    free_relative = free_energy - free_energy.min()
    minimum_relative = minimum_energy - minimum_energy.min()
    axes[0, 0].plot(
        q,
        minimum_relative,
        color=colors["orange"],
        linestyle="--",
        linewidth=2,
        label="minimum over rotamers",
    )
    axes[0, 0].plot(
        q,
        free_relative,
        color=colors["teal"],
        linewidth=2.5,
        label="free energy F(q)",
    )
    axes[0, 0].axvline(0, color=colors["slate"], linewidth=1, alpha=0.5)
    axes[0, 0].set_title("The backbone sees a smoother potential")
    axes[0, 0].set_xlabel("backbone coordinate q  (open  <  0  <  closed)")
    axes[0, 0].set_ylabel("relative energy")
    axes[0, 0].legend(frameon=False, fontsize=9)
    axes[0, 0].grid(alpha=0.2)

    axes[0, 1].plot(
        q,
        exact_probability,
        color=colors["ink"],
        linewidth=2.5,
        label="exact marginal",
    )
    axes[0, 1].step(
        q,
        free_histogram,
        where="mid",
        color=colors["teal"],
        linewidth=1.5,
        alpha=0.9,
        label="free-energy sampler",
    )
    axes[0, 1].step(
        q,
        joint_histogram,
        where="mid",
        color=colors["orange"],
        linewidth=1.5,
        alpha=0.9,
        label="explicit rotamers",
    )
    axes[0, 1].set_title("Both methods target the same ensemble")
    axes[0, 1].set_xlabel("backbone coordinate q")
    axes[0, 1].set_ylabel("probability on q grid")
    axes[0, 1].legend(frameon=False, fontsize=9)
    axes[0, 1].grid(alpha=0.2)

    axes[1, 0].plot(
        trace_steps,
        q[results["free_trace"][:trace_length]],
        color=colors["teal"],
        linewidth=1,
        alpha=0.85,
        label="free-energy sampler",
    )
    axes[1, 0].plot(
        trace_steps,
        q[results["joint_trace"][:trace_length]],
        color=colors["orange"],
        linewidth=1,
        alpha=0.75,
        label="explicit rotamers",
    )
    axes[1, 0].axhline(0, color=colors["slate"], linewidth=1, alpha=0.5)
    axes[1, 0].set_title("The same target can have different dynamics")
    axes[1, 0].set_xlabel("post-burn-in step")
    axes[1, 0].set_ylabel("backbone coordinate q")
    axes[1, 0].legend(frameon=False, fontsize=9)
    axes[1, 0].grid(alpha=0.2)

    effective_rotamer_curve = results["effective_rotamers"]
    axes[1, 1].plot(
        q,
        effective_rotamer_curve,
        color=colors["ink"],
        linewidth=2.5,
    )
    axes[1, 1].fill_between(
        q,
        effective_rotamer_curve,
        color=colors["sand"],
        alpha=0.3,
    )
    axes[1, 1].set_title("How many rotamer combinations matter?")
    axes[1, 1].set_xlabel("backbone coordinate q")
    axes[1, 1].set_ylabel("effective count  exp(entropy)")
    axes[1, 1].grid(alpha=0.2)

    fig.suptitle(
        "Side-chain marginalization in a seven-residue hinge",
        fontsize=15,
        color=colors["ink"],
    )
    fig
    return


@app.cell(hide_code=True)
def _(np):
    def run_experiment(temperature, coupling, barrier, seed=7):
        q = np.linspace(-2.2, 2.2, 221)
        n_sidechains = 7
        n_states = 3**n_sidechains
        state_codes = np.arange(n_states, dtype=int)
        state_powers = 3 ** np.arange(n_sidechains - 1, -1, -1)
        configurations = (
            (state_codes[:, None] // state_powers[None, :]) % 3
        ).astype(float)

        contact = 1.0 / (1.0 + np.exp(-4.0 * (q + 0.1)))
        backbone_energy = barrier * (q**2 - 1.0) ** 2 - 0.8 * q
        phases = np.linspace(-0.9, 0.9, n_sidechains)
        preferred_rotamer = (
            0.55
            + 1.25 * contact[:, None]
            + 0.18
            * np.sin(1.6 * q[:, None] + phases[None, :])
        )
        local_energy = 0.48 * np.sum(
            (
                configurations[None, :, :]
                - preferred_rotamer[:, None, :]
            )
            ** 2,
            axis=2,
        )

        pair_score = np.zeros(n_states)
        for site in range(n_sidechains - 1):
            pair_score += np.where(
                configurations[:, site] == configurations[:, site + 1],
                -0.15,
                0.0,
            )
            pair_score += np.where(
                (configurations[:, site] == 2)
                & (configurations[:, site + 1] == 2),
                1.0,
                0.0,
            )

        energy = (
            backbone_energy[:, None]
            + local_energy
            + coupling * contact[:, None] * pair_score[None, :]
        )

        def logsumexp(values):
            peak = np.max(values, axis=1, keepdims=True)
            return (
                peak
                + np.log(np.sum(np.exp(values - peak), axis=1, keepdims=True))
            ).ravel()

        log_weights = -energy / temperature
        log_partition = logsumexp(log_weights)
        free_energy = -temperature * log_partition
        probabilities = np.exp(log_weights - log_partition[:, None])
        entropy = -np.sum(
            probabilities * np.log(probabilities + 1e-300),
            axis=1,
        )
        exact_probability = np.exp(
            -(free_energy - free_energy.min()) / temperature
        )
        exact_probability /= exact_probability.sum()

        def integrated_autocorrelation(values, max_lag=250):
            centered = values.astype(float) - np.mean(values)
            variance = np.mean(centered**2)
            if variance < 1e-12:
                return 1.0

            estimate = 1.0
            for lag in range(1, min(max_lag, len(values) - 1)):
                correlation = np.mean(
                    centered[:-lag] * centered[lag:]
                ) / variance
                if correlation <= 0:
                    break
                estimate += 2.0 * correlation
            return max(1.0, estimate)

        def sample_free_energy(rng, steps=18000, burn_in=3000):
            current = int(np.argmin(free_energy))
            trace = np.empty(steps - burn_in, dtype=int)
            attempts = 0
            accepted = 0

            for step in range(steps):
                proposal = current + int(
                    rng.choice([-1, 1]) * rng.integers(1, 19)
                )
                if 0 <= proposal < len(q):
                    attempts += 1
                    log_acceptance = min(
                        0.0,
                        -(free_energy[proposal] - free_energy[current])
                        / temperature,
                    )
                    if np.log(rng.random()) < log_acceptance:
                        current = proposal
                        accepted += 1
                if step >= burn_in:
                    trace[step - burn_in] = current

            return trace, accepted / attempts

        state_lookup = np.full(n_states, -1, dtype=int)
        encoded_states = configurations.astype(int) @ state_powers
        state_lookup[encoded_states] = np.arange(n_states)

        def sample_joint(rng, steps=18000, burn_in=3000):
            current_q = int(np.argmin(free_energy))
            current_state = int(np.argmin(energy[current_q]))
            trace = np.empty(steps - burn_in, dtype=int)
            q_attempts = 0
            q_accepted = 0
            side_attempts = 0
            side_accepted = 0

            for step in range(steps):
                if rng.random() < 0.75:
                    proposal_q = current_q + int(
                        rng.choice([-1, 1]) * rng.integers(1, 19)
                    )
                    if 0 <= proposal_q < len(q):
                        q_attempts += 1
                        log_acceptance = min(
                            0.0,
                            -(energy[proposal_q, current_state]
                            - energy[current_q, current_state])
                            / temperature,
                        )
                        if np.log(rng.random()) < log_acceptance:
                            current_q = proposal_q
                            q_accepted += 1
                else:
                    side_attempts += 1
                    proposal_state = configurations[
                        current_state
                    ].astype(int)
                    site = int(rng.integers(n_sidechains))
                    old_state = proposal_state[site]
                    proposal_state[site] = (
                        old_state + int(rng.integers(1, 3))
                    ) % 3
                    proposal_state_index = int(
                        state_lookup[proposal_state @ state_powers]
                    )
                    log_acceptance = min(
                        0.0,
                        -(energy[current_q, proposal_state_index]
                        - energy[current_q, current_state])
                        / temperature,
                    )
                    if np.log(rng.random()) < log_acceptance:
                        current_state = proposal_state_index
                        side_accepted += 1

                if step >= burn_in:
                    trace[step - burn_in] = current_q

            return (
                trace,
                q_accepted / q_attempts,
                side_accepted / side_attempts,
            )

        free_trace, free_acceptance = sample_free_energy(
            np.random.default_rng(seed)
        )
        joint_trace, joint_q_acceptance, side_acceptance = sample_joint(
            np.random.default_rng(seed + 1)
        )
        free_histogram = np.bincount(
            free_trace,
            minlength=len(q),
        ).astype(float)
        free_histogram /= free_histogram.sum()
        joint_histogram = np.bincount(
            joint_trace,
            minlength=len(q),
        ).astype(float)
        joint_histogram /= joint_histogram.sum()

        return {
            "q": q,
            "free_energy": free_energy,
            "minimum_energy": energy.min(axis=1),
            "exact_probability": exact_probability,
            "effective_rotamers": np.exp(entropy),
            "free_trace": free_trace,
            "joint_trace": joint_trace,
            "free_histogram": free_histogram,
            "joint_histogram": joint_histogram,
            "free_acceptance": free_acceptance,
            "joint_q_acceptance": joint_q_acceptance,
            "side_acceptance": side_acceptance,
            "free_iat": integrated_autocorrelation(free_trace),
            "joint_iat": integrated_autocorrelation(joint_trace),
            "free_total_variation": 0.5
            * np.abs(free_histogram - exact_probability).sum(),
            "joint_total_variation": 0.5
            * np.abs(joint_histogram - exact_probability).sum(),
        }

    return (run_experiment,)


if __name__ == "__main__":
    app.run()
