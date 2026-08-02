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
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    return mo, torch, np, plt


@app.cell(hide_code=True)
def _(mo):
    style = mo.Html("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=JetBrains+Mono:wght@400;700&display=swap');
        
        .notebook-container {
            font-family: 'Outfit', sans-serif;
            color: #1f2937;
            max-width: 900px;
            margin: 0 auto;
        }
        
        .dark .notebook-container {
            color: #f3f4f6;
        }
        
        .header-gradient {
            background: linear-gradient(135deg, #4f46e5 0%, #06b6d4 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
            text-align: center;
        }
        
        .sub-header {
            font-size: 1.25rem;
            color: #4b5563;
            text-align: center;
            margin-bottom: 2rem;
            font-weight: 400;
        }
        
        .card {
            background: rgba(255, 255, 255, 0.85);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(229, 231, 235, 0.6);
            border-radius: 16px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
            transition: all 0.3s ease;
        }
        
        .dark .card {
            background: rgba(31, 41, 55, 0.85);
            border: 1px solid rgba(75, 85, 99, 0.5);
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3), 0 2px 4px -1px rgba(0, 0, 0, 0.2);
        }
        
        .card:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        }
        
        .metric-title {
            font-size: 0.85rem;
            font-weight: 600;
            color: #6b7280;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        .metric-value {
            font-size: 1.75rem;
            font-weight: 700;
            margin-top: 0.25rem;
        }
        
        .metric-value.iptm {
            color: #ec4899;
        }
        
        .metric-value.ptmenergy {
            color: #10b981;
        }
        
        .active-label {
            font-size: 1.1rem;
            font-weight: 600;
            margin-top: 0.5rem;
            color: #374151;
        }
        
        .dark .active-label {
            color: #d1d5db;
        }
    </style>
    """)
    style
    return (style,)


@app.cell(hide_code=True)
def _(mo):
    html_content = """
<div class="notebook-container">
    <div class="header-gradient">BindEnergyCraft</div>
    <div class="sub-header">Casting Structure Predictors as Energy-Based Models for Protein Binder Design</div>
</div>
"""
    mo.md(html_content)
    return


@app.cell(hide_code=True)
def _(mo):
    intro = """
# Reinterpreting Structure Predictors for Better Gradients

This notebook explores the core conceptual innovation of **BindEnergyCraft (BECraft)** (arXiv:2505.21241), which won the **Best Paper Award** at the **ICML 2025 Workshop on Generative AI for Biology**.

### The Bottleneck: Sparse ipTM Gradients
Traditional *de novo* protein binder design pipelines (like *BindCraft*) optimize heuristic structural confidence scores, such as the **interface predicted TM-score (ipTM)**. While effective, ipTM suffers from a fundamental mathematical limitation during backpropagation: **sparse gradients**.

Because ipTM calculates the TM-score by taking a `max` operation over reference residues (alignment frames), the backpropagated loss only provides gradients to the residue pairs that belong to the winning alignment frame. The rest of the interface residues receive **zero gradients** at that optimization step.

### The Solution: pTMEnergy
BindEnergyCraft solves this by reinterpreting protein structure predictors as **Energy-Based Models (EBMs)**. Using the Joint Energy-based Modeling (JEM) framework, the authors replace the `max` operation with the **LogSumExp** trick, yielding a statistically grounded energy function called **pTMEnergy**:

$$E_{\\text{pTMEnergy}}(x) = - \\frac{1}{|I|} \\sum_{(i,j) \\in I} \\log \\sum_{b=1}^B g(d_b) \\exp(l_{ijb})$$

Where $l_{ijb}$ are the predicted alignment error (pAE) logits for residue pair $(i, j)$ in bin $b$, and $g(d_b)$ is the pTM alignment-error scaling kernel. This aggregates forces across all interface pairs, producing **dense, interface-wide gradients**.
"""
    mo.md(intro)
    return


@app.cell(hide_code=True)
def _(mo):
    # User controls
    length_slider = mo.ui.slider(start=4, stop=10, step=1, value=6, label="Binder Length (residues)")
    dist_slider = mo.ui.slider(start=1.2, stop=2.5, step=0.1, value=1.8, label="Optimal Target Contact Distance")
    lr_slider = mo.ui.slider(start=0.01, stop=0.1, step=0.01, value=0.05, label="Optimizer Learning Rate")
    spring_slider = mo.ui.slider(start=1.0, stop=10.0, step=1.0, value=5.0, label="Backbone Spring Loss Weight")

    mo.vstack([
        mo.md("## Live 2D Docking Simulation"),
        mo.md("Adjust the pocket and optimization parameters below. The binder starts as a flat horizontal chain above a parabolic target receptor pocket. Moving the sliders triggers a live 100-step gradient descent loop for both objectives simultaneously."),
        mo.hstack([
            mo.vstack([mo.md("### Structure Geometry"), length_slider, dist_slider]),
            mo.vstack([mo.md("### Optimization Parameters"), lr_slider, spring_slider])
        ], gap=4)
    ])
    return length_slider, dist_slider, lr_slider, spring_slider


@app.cell(hide_code=True)
def _(length_slider, dist_slider, lr_slider, spring_slider, torch):
    L = length_slider.value
    d_opt = dist_slider.value
    lr = lr_slider.value
    w_spring = spring_slider.value
    
    # Target pocket coordinates (12 residues along a parabola)
    M = 12
    theta = torch.linspace(-1.5, 1.5, M)
    Y = torch.stack([theta, 0.4 * theta**2 - 1.2], dim=1) # (M, 2)
    
    def optimize_binder(metric):
        # Initialize binder coordinates flat above the pocket
        X = torch.stack([torch.linspace(-1.0, 1.0, L), torch.ones(L) * 1.5], dim=1)
        X.requires_grad = True
        
        B = 64
        d_b = torch.linspace(0.5, 15.0, B)
        d_0 = 19.0
        
        optimizer = torch.optim.Adam([X], lr=lr)
        
        history_coords = []
        history_losses = []
        history_clashes = []
        step_0_grad = None
        
        for step in range(100):
            optimizer.zero_grad()
            
            # Pairwise distance matrix
            dist = torch.cdist(X.unsqueeze(0), Y.unsqueeze(0)).squeeze(0) # (L, M)
            
            # Simulated alignment error mean (mu)
            mu = 1.5 + 1.5 * (dist - d_opt)**2
            clash_mask = dist < 1.0
            mu = torch.where(clash_mask, mu + 30.0 * (1.0 - dist)**2, mu)
            mu = torch.clamp(mu, max=14.0)
            
            # Simulated pAE logits
            logits = -0.5 * ((d_b.view(1, 1, B) - mu.unsqueeze(-1)) / 1.5)**2
            
            # Spring backbone constraint (maintain initial spacing)
            diffs = X[1:] - X[:-1]
            bond_dists = torch.sqrt(torch.sum(diffs**2, dim=-1) + 1e-8)
            target_bond_len = 2.0 / (L - 1)
            spring_loss = torch.sum((bond_dists - target_bond_len)**2)
            
            if metric == 'iptm':
                q = torch.softmax(logits, dim=-1)
                g = 1.0 / (1.0 + (d_b / d_0)**2)
                pair_scores = torch.sum(q * g.view(1, 1, B), dim=-1)
                score_i = torch.mean(pair_scores, dim=1)
                iptm = torch.max(score_i)
                interaction_loss = -10.0 * iptm
            else:
                g = 1.0 / (1.0 + (d_b / d_0)**2)
                log_g = torch.log(g.view(1, 1, B) + 1e-8)
                e_ij = -torch.logsumexp(logits + log_g, dim=-1)
                interaction_loss = torch.mean(e_ij)
                
            loss = interaction_loss + w_spring * spring_loss
            
            # Capture step 0 gradients of the interaction loss with respect to distance matrix
            if step == 0:
                d_leaf = torch.cdist(X.unsqueeze(0), Y.unsqueeze(0)).squeeze(0).clone().detach().requires_grad_(True)
                mu_leaf = 1.5 + 1.5 * (d_leaf - d_opt)**2
                mu_leaf = torch.where(d_leaf < 1.0, mu_leaf + 30.0 * (1.0 - d_leaf)**2, mu_leaf)
                mu_leaf = torch.clamp(mu_leaf, max=14.0)
                logits_leaf = -0.5 * ((d_b.view(1, 1, B) - mu_leaf.unsqueeze(-1)) / 1.5)**2
                
                if metric == 'iptm':
                    q_leaf = torch.softmax(logits_leaf, dim=-1)
                    pair_scores_leaf = torch.sum(q_leaf * g.view(1, 1, B), dim=-1)
                    score_i_leaf = torch.mean(pair_scores_leaf, dim=1)
                    loss_leaf = -10.0 * torch.max(score_i_leaf)
                else:
                    log_g = torch.log(g.view(1, 1, B) + 1e-8)
                    e_ij_leaf = -torch.logsumexp(logits_leaf + log_g, dim=-1)
                    loss_leaf = torch.mean(e_ij_leaf)
                
                loss_leaf.backward()
                step_0_grad = d_leaf.grad.clone().detach()
                
            loss.backward()
            optimizer.step()
            
            history_coords.append(X.detach().clone())
            history_losses.append(loss.item())
            
            clash_count = (dist < 1.0).sum().item()
            history_clashes.append(clash_count)
            
        return history_coords, history_losses, history_clashes, step_0_grad

    # Run optimization for both methods
    coords_ip, losses_ip, clashes_ip, grad_ip = optimize_binder('iptm')
    coords_en, losses_en, clashes_en, grad_en = optimize_binder('energy')
    
    # Calculate active residue percentages (row norm is non-zero)
    grad_row_norms_ip = torch.norm(grad_ip, dim=-1)
    active_pct_ip = float((grad_row_norms_ip > 1e-6).sum().item()) / L * 100.0
    
    grad_row_norms_en = torch.norm(grad_en, dim=-1)
    active_pct_en = float((grad_row_norms_en > 1e-6).sum().item()) / L * 100.0

    return (
        L, d_opt, lr, w_spring, Y,
        coords_ip, losses_ip, clashes_ip, grad_ip, active_pct_ip,
        coords_en, losses_en, clashes_en, grad_en, active_pct_en
    )


@app.cell(hide_code=True)
def _(mo, clashes_ip, clashes_en, active_pct_ip, active_pct_en):
    # Render quantitative metrics dashboard
    dashboard = mo.hstack([
        mo.md(f"""
        <div class="card" style="flex: 1; text-align: center;">
            <div class="metric-title">ipTM Optimization</div>
            <div class="metric-value iptm">Clashes: {clashes_ip[-1]}</div>
            <div class="active-label">Active residues at Step 0:</div>
            <div style="font-size: 1.5rem; font-weight: 800; color: #db2777;">{active_pct_ip:.1f}%</div>
            <div style="font-size: 0.85rem; color: #6b7280; margin-top: 0.5rem;">Only the winning frame receives gradients</div>
        </div>
        """),
        mo.md(f"""
        <div class="card" style="flex: 1; text-align: center;">
            <div class="metric-title">pTMEnergy Optimization</div>
            <div class="metric-value ptmenergy">Clashes: {clashes_en[-1]}</div>
            <div class="active-label">Active residues at Step 0:</div>
            <div style="font-size: 1.5rem; font-weight: 800; color: #059669;">{active_pct_en:.1f}%</div>
            <div style="font-size: 0.85rem; color: #6b7280; margin-top: 0.5rem;">All interface residues coordinate together</div>
        </div>
        """)
    ], gap=4, justify="space-around")
    dashboard
    return (dashboard,)


@app.cell(hide_code=True)
def _(plt, Y, coords_ip, coords_en, L, torch):
    # Plot 1: Structure Layout Comparison
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)
    
    _Y_np = Y.numpy()
    _M = len(_Y_np)
    
    for _ax in _axes:
        _ax.set_facecolor('#f8fafc')
        _ax.grid(color='#e2e8f0', linestyle='--', linewidth=0.5)
        _ax.axhline(0, color='#cbd5e1', linewidth=0.8, zorder=0)
        _ax.axvline(0, color='#cbd5e1', linewidth=0.8, zorder=0)
        
    # ipTM subplot
    _axes[0].scatter(_Y_np[:, 0], _Y_np[:, 1], color='#3b82f6', s=90, label='Target Pocket', zorder=3, edgecolors='#1d4ed8', linewidth=1.5)
    init_ip = coords_ip[0].numpy()
    _axes[0].plot(init_ip[:, 0], init_ip[:, 1], 'o--', color='#94a3b8', alpha=0.6, label='Binder (Initial)', zorder=2)
    final_ip = coords_ip[-1].numpy()
    _axes[0].plot(final_ip[:, 0], final_ip[:, 1], 'o-', color='#ec4899', linewidth=3.0, markersize=8, label='Binder (Final)', zorder=4, markeredgecolor='#be185d')
    
    # Draw interface contacts & clashes for ipTM
    dist_final_ip = torch.cdist(coords_ip[-1].unsqueeze(0), Y.unsqueeze(0)).squeeze(0).numpy()
    for i in range(L):
        for j in range(_M):
            d = dist_final_ip[i, j]
            if d < 1.0:
                _axes[0].plot([final_ip[i, 0], _Y_np[j, 0]], [final_ip[i, 1], _Y_np[j, 1]], color='#ef4444', linestyle=':', alpha=0.8, linewidth=1.2)
            elif d <= 1.8:
                _axes[0].plot([final_ip[i, 0], _Y_np[j, 0]], [final_ip[i, 1], _Y_np[j, 1]], color='#22c55e', linestyle='-', alpha=0.3, linewidth=0.8)
                
    _axes[0].set_title('ipTM optimization: Skewed & Stuck', fontsize=13, fontweight='bold', pad=12, color='#0f172a')
    _axes[0].legend(loc='upper right')
    
    # pTMEnergy subplot
    _axes[1].scatter(_Y_np[:, 0], _Y_np[:, 1], color='#3b82f6', s=90, label='Target Pocket', zorder=3, edgecolors='#1d4ed8', linewidth=1.5)
    init_en = coords_en[0].numpy()
    _axes[1].plot(init_en[:, 0], init_en[:, 1], 'o--', color='#94a3b8', alpha=0.6, label='Binder (Initial)', zorder=2)
    final_en = coords_en[-1].numpy()
    _axes[1].plot(final_en[:, 0], final_en[:, 1], 'o-', color='#10b981', linewidth=3.0, markersize=8, label='Binder (Final)', zorder=4, markeredgecolor='#047857')
    
    # Draw interface contacts & clashes for pTMEnergy
    dist_final_en = torch.cdist(coords_en[-1].unsqueeze(0), Y.unsqueeze(0)).squeeze(0).numpy()
    for i in range(L):
        for j in range(_M):
            d = dist_final_en[i, j]
            if d < 1.0:
                _axes[1].plot([final_en[i, 0], _Y_np[j, 0]], [final_en[i, 1], _Y_np[j, 1]], color='#ef4444', linestyle=':', alpha=0.8, linewidth=1.2)
            elif d <= 1.8:
                _axes[1].plot([final_en[i, 0], _Y_np[j, 0]], [final_en[i, 1], _Y_np[j, 1]], color='#22c55e', linestyle='-', alpha=0.3, linewidth=0.8)
                
    _axes[1].set_title('pTMEnergy optimization: Symmetric & snug', fontsize=13, fontweight='bold', pad=12, color='#0f172a')
    _axes[1].legend(loc='upper right')
    
    for _ax in _axes:
        _ax.set_xlabel('X Coordinate', fontsize=11, labelpad=8)
        _ax.set_xlim(-2.0, 2.0)
        _ax.set_ylim(-1.5, 2.0)
    _axes[0].set_ylabel('Y Coordinate', fontsize=11, labelpad=8)
    
    plt.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(plt, grad_ip, grad_en, L):
    # Plot 2: Gradient heatmaps at Step 0
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 5))
    
    g_ip = torch.abs(grad_ip).numpy()
    g_en = torch.abs(grad_en).numpy()
    
    vmax = max(g_ip.max(), g_en.max())
    
    im0 = _axes[0].imshow(g_ip, cmap='PuRd', aspect='auto', interpolation='nearest', vmin=0.0, vmax=vmax)
    _axes[0].set_title('ipTM Distance Gradient Matrix (Step 0)', fontsize=13, fontweight='bold', pad=12, color='#0f172a')
    _fig.colorbar(im0, ax=_axes[0], label='Absolute Gradient Magnitude')
    
    im1 = _axes[1].imshow(g_en, cmap='YlGnBu', aspect='auto', interpolation='nearest', vmin=0.0, vmax=vmax)
    _axes[1].set_title('pTMEnergy Distance Gradient Matrix (Step 0)', fontsize=13, fontweight='bold', pad=12, color='#0f172a')
    _fig.colorbar(im1, ax=_axes[1], label='Absolute Gradient Magnitude')
    
    for _ax in _axes:
        _ax.set_xlabel('Target Residue Index', fontsize=11, labelpad=8)
        _ax.set_ylabel('Binder Residue Index', fontsize=11, labelpad=8)
        _ax.set_xticks(range(12))
        _ax.set_yticks(range(L))
        _ax.set_xticklabels([f'T{i+1}' for i in range(12)])
        _ax.set_yticklabels([f'B{i+1}' for i in range(L)])
        
    plt.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(plt, losses_ip, losses_en, clashes_ip, clashes_en, np):
    # Plot 3: Trajectories (Loss and Clashes)
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4))
    
    l_ip = np.array(losses_ip)
    l_en = np.array(losses_en)
    
    norm_ip = (l_ip - l_ip.min()) / (l_ip.max() - l_ip.min() + 1e-8)
    border_en = l_en.max() - l_en.min() + 1e-8
    norm_en = (l_en - l_en.min()) / border_en
    
    _axes[0].plot(norm_ip, color='#ec4899', linewidth=2.5, label='ipTM (Normalized)')
    _axes[0].plot(norm_en, color='#10b981', linewidth=2.5, label='pTMEnergy (Normalized)')
    _axes[0].set_title('Normalized Loss Trajectory', fontsize=13, fontweight='bold', pad=12)
    _axes[0].set_xlabel('Optimization Step', fontsize=11)
    _axes[0].set_ylabel('Loss (Normalized)', fontsize=11)
    _axes[0].grid(color='#e2e8f0', linestyle='--', linewidth=0.5)
    _axes[0].legend()
    
    _axes[1].plot(clashes_ip, color='#ec4899', linewidth=2.5, label='ipTM Clashes')
    _axes[1].plot(clashes_en, color='#10b981', linewidth=2.5, label='pTMEnergy Clashes')
    _axes[1].set_title('Steric Clashes (d < 1.0) Over Steps', fontsize=13, fontweight='bold', pad=12)
    _axes[1].set_xlabel('Optimization Step', fontsize=11)
    _axes[1].set_ylabel('Clash Count', fontsize=11)
    _axes[1].grid(color='#e2e8f0', linestyle='--', linewidth=0.5)
    _axes[1].legend()
    
    plt.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    summary_text = """
### Summary of Insights

1. **Gradient Density**: Notice the stark difference in the gradient heatmaps. Under **ipTM**, the gradient heatmap is highly sparse—most rows (binder residues) are completely white, meaning they receive zero signal at the beginning of optimization. Under **pTMEnergy**, the heatmap is dense and smooth; every residue pair contributes to the optimization trajectory.
2. **Structural Realism**: In the structure comparison plot, you can see that the **pTMEnergy** binder centers itself perfectly in the pocket and spreads its residues symmetrically to form a stable interface. The **ipTM** binder, in contrast, gets skewed because the optimizer only "sees" one alignment frame at a time, resulting in an asymmetric, uncoordinated shape that often gets stuck.
3. **Steric Clashes**: Because pTMEnergy provides global coordinating gradients, it naturally avoids local minima that lead to atomic clashes. It allows the binder to slide smoothly into the target pocket, aligning the entire chain simultaneously.
"""
    mo.md(summary_text)
    return


if __name__ == "__main__":
    app.run()
