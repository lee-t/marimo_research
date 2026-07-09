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
    import numpy as np
    import matplotlib.pyplot as plt

    return mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    # CSS styling to make the notebook look highly premium and polished
    styles = mo.Html("""
    <style>
        .paper-header {
            background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
            color: white;
            padding: 28px;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(59, 130, 246, 0.15);
            margin-bottom: 24px;
        }
        .paper-title {
            font-size: 28px;
            font-weight: 800;
            margin: 0;
            font-family: 'Outfit', 'Inter', sans-serif;
            letter-spacing: -0.025em;
        }
        .paper-subtitle {
            font-size: 15px;
            opacity: 0.95;
            margin-top: 8px;
            font-family: 'Inter', sans-serif;
            line-height: 1.5;
        }
        .card {
            background: white;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 24px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
            margin-bottom: 20px;
        }
        .control-row {
            display: flex;
            gap: 16px;
            align-items: center;
            flex-wrap: wrap;
            margin-bottom: 12px;
        }
        .terminal-window {
            border-radius: 8px;
            background-color: #0f172a;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25);
            overflow: hidden;
            margin: 20px 0;
        }
        .terminal-bar {
            background-color: #1e293b;
            padding: 10px 16px;
            display: flex;
            align-items: center;
            border-bottom: 1px solid #334155;
        }
        .dot {
            height: 10px;
            width: 10px;
            border-radius: 50%;
            display: inline-block;
            margin-right: 8px;
        }
        .red { background-color: #ef4444; }
        .yellow { background-color: #eab308; }
        .green { background-color: #22c55e; }
        .terminal-title {
            color: #94a3b8;
            font-size: 12px;
            font-family: monospace;
            margin-left: auto;
            margin-right: auto;
        }
        .terminal-body {
            padding: 16px;
            color: #e2e8f0;
            font-family: 'Fira Code', 'Courier New', Courier, monospace;
            font-size: 12.5px;
            line-height: 1.6;
            max-height: 350px;
            overflow-y: auto;
        }
        .log-line {
            margin-bottom: 6px;
            white-space: pre-wrap;
        }
        .log-info { color: #38bdf8; }
        .log-scan { color: #c084fc; }
        .log-llm { color: #fbbf24; }
        .log-test { color: #f472b6; }
        .log-error { color: #f87171; font-weight: bold; }
        .log-refiner { color: #fb7185; }
        .log-success { color: #4ade80; font-weight: bold; }
        .log-deploy { color: #2dd4bf; font-weight: bold; }
        
        .chat-container {
            display: flex;
            flex-direction: column;
            gap: 16px;
            margin: 20px 0;
        }
        .chat-bubble {
            padding: 16px;
            border-radius: 12px;
            max-width: 85%;
            line-height: 1.5;
            font-family: 'Inter', sans-serif;
            box-shadow: 0 2px 4px rgba(0,0,0,0.02);
        }
        .chat-user {
            background-color: #2563eb;
            color: white;
            align-self: flex-end;
            border-bottom-right-radius: 2px;
        }
        .chat-agent {
            background-color: #f8fafc;
            border: 1px solid #e2e8f0;
            color: #1e293b;
            align-self: flex-start;
            border-bottom-left-radius: 2px;
        }
        .chat-agent-thought {
            background-color: #f1f5f9;
            border-left: 3px solid #64748b;
            padding: 8px 12px;
            font-size: 12px;
            color: #475569;
            font-style: italic;
            margin-bottom: 10px;
            border-radius: 4px;
            font-family: 'Inter', sans-serif;
        }
        .chat-agent-call {
            background-color: #0f172a;
            color: #38bdf8;
            padding: 8px 12px;
            font-family: monospace;
            font-size: 12px;
            border-radius: 4px;
            margin-bottom: 10px;
            border-left: 3px solid #0284c7;
        }
        .chat-agent-call-fail {
            background-color: #0f172a;
            color: #f87171;
            padding: 8px 12px;
            font-family: monospace;
            font-size: 12px;
            border-radius: 4px;
            margin-bottom: 10px;
            border-left: 3px solid #ef4444;
        }
        .chat-agent-response {
            font-size: 14px;
        }
        .chat-agent-fail {
            background-color: #fef2f2;
            border: 1px solid #fca5a5;
            color: #991b1b;
            align-self: flex-start;
            border-bottom-left-radius: 2px;
        }
        .badge {
            padding: 3px 8px;
            border-radius: 9999px;
            font-size: 11px;
            font-weight: 600;
            display: inline-block;
            margin-bottom: 8px;
        }
        .badge-blue { background-color: #dbeafe; color: #1e40af; }
        .badge-purple { background-color: #f3e8ff; color: #6b21a8; }
        .badge-green { background-color: #dcfce7; color: #166534; }
        .badge-red { background-color: #fee2e2; color: #991b1b; }
    </style>
    """)
    styles
    return (styles,)


@app.cell(hide_code=True)
def _(mo):
    # App header block
    header = mo.Html("""
    <div class="paper-header">
        <div class="paper-title">Paper2Agent 🤖🔬</div>
        <div class="paper-subtitle">Reimagining Research Papers As Interactive and Reliable AI Agents (Miao et al., Stanford 2025)</div>
    </div>
    """)
    header
    return (header,)


@app.cell(hide_code=True)
def _(mo):
    # Short narrative introduction explaining the concept
    intro_text = mo.md("""
    Scientific publications are **static**. Even with open-source repositories, reproducing or applying a paper's code requires resolving complex dependencies, debugging environment setups, and interpreting APIs. 

    If we ask an LLM (like Claude or GPT) to write code using the paper's repository on the fly, it frequently **hallucinates class structures or inputs** because it lacks the context of how the methods depend on each other.

    **Paper2Agent** solves this by using a multi-agent pipeline to automatically convert a research paper and its codebase into a **Model Context Protocol (MCP)** server of **verified, tested tools**. 
    
    This notebook lets you simulate this **Test-Verifier-Improver** refinement loop and interact with the resulting reliable Paper Agent!
    """)
    intro_text
    return (intro_text,)


@app.cell(hide_code=True)
def _(mo):
    # Setup interactive controls for the simulation
    selected_paper = mo.ui.dropdown(
        options={
            "alphagenome": "AlphaGenome (Genomic Variant Interpretation)",
            "tissue": "TISSUE (Spatial Transcriptomics Uncertainty)",
            "scanpy": "Scanpy (Single-Cell Preprocessing & Clustering)"
        },
        value="alphagenome",
        label="🔬 Select Case Study"
    )

    selected_strategy = mo.ui.radio(
        options={
            "naive": "Naive LLM Code Generation (No test loop)",
            "paper2agent": "Paper2Agent Pipeline (Iterative Test-Verifier-Improver)"
        },
        value="paper2agent",
        label="⚙️ Agentization Strategy"
    )

    max_iterations = mo.ui.slider(
        start=1,
        stop=5,
        step=1,
        value=3,
        label="🔄 Max Refinement Iterations"
    )

    run_btn = mo.ui.button(
        label="🚀 Run Agentization Pipeline",
        kind="neutral"
    )

    return max_iterations, run_btn, selected_paper, selected_strategy


@app.cell(hide_code=True)
def _(max_iterations, mo, run_btn, selected_paper, selected_strategy):
    # Control Panel Layout
    control_panel = mo.md(f"""
    ### 🛠️ Configuration Panel
    
    Choose a research paper/library and a strategy, then trigger the pipeline.
    
    <div class="control-row">
        <div>{selected_paper}</div>
        <div>{selected_strategy}</div>
        <div>{max_iterations}</div>
    </div>
    <div style="margin-top: 15px;">
        {run_btn}
    </div>
    """)
    
    mo.Html(f"<div class='card'>{control_panel.text}</div>")
    return (control_panel,)


@app.cell(hide_code=True)
def _(
    CASE_STUDIES,
    max_iterations,
    mo,
    run_btn,
    selected_paper,
    selected_strategy,
):
    # Reactively compute the agentization terminal logs and result codes
    if not run_btn.value:
        terminal_output = mo.md("""
        <div style="text-align: center; color: #64748b; padding: 20px;">
            <h3>Waiting for execution...</h3>
            <p>Select a case study and click <strong>Run Agentization Pipeline</strong> above.</p>
        </div>
        """)
        pipeline_run = False
        active_code_tabs = None
    else:
        pipeline_run = True
        _case = CASE_STUDIES[selected_paper.value]
        
        # Build simulated terminal log
        log_lines = []
        if selected_strategy.value == "naive":
            log_lines.append('<div class="log-line log-info">[INFO] Initializing sandbox environment...</div>')
            log_lines.append('<div class="log-line log-info">[INFO] Extracting repository contents...</div>')
            log_lines.append(f'<div class="log-line log-llm">[LLM] Analyzing paper structure and writing naive tool candidate for \'{selected_paper.value}\'...</div>')
            log_lines.append('<div class="log-line log-info">[INFO] Bypassing test verification. Packing candidate tool.</div>')
            log_lines.append('<div class="log-line log-success">[SUCCESS] Tool package generated in 1.4s! (Unverified)</div>')
        else:
            # Paper2Agent Test-driven refinement log
            log_lines.append('<div class="log-line log-info">[INFO] Initializing isolated Docker sandbox...</div>')
            log_lines.append(f'<div class="log-line log-info">[INFO] Environment configured with dependencies: \'{selected_paper.value}\'</div>')
            log_lines.append('<div class="log-line log-scan">[SCAN] Tutorial Scanner: Running repository tutorials end-to-end...</div>')
            log_lines.append('<div class="log-line log-scan">[SCAN] Tutorial Scanner: Captured baseline inputs and numerical outputs.</div>')
            log_lines.append('<div class="log-line log-llm">[LLM] Tool Extractor: Generating parameterizable MCP tool candidate...</div>')
            
            # Show iterative tests up to max_iterations
            limit = max_iterations.value
            ver_logs = _case["verified_log"]
            
            error_count = 0
            for type_, text in ver_logs:
                if type_ == "ERROR":
                    error_count += 1
                    if error_count > limit:
                        # Exceeded iterations, break early with failure
                        log_lines.append('<div class="log-line log-error">[TEST FAILURE] Iteration limit reached! Stopping refinement loop.</div>')
                        log_lines.append('<div class="log-line log-error">[ERROR] MCP tool deployment failed: tests do not pass.</div>')
                        break
                    else:
                        log_lines.append(f'<div class="log-line log-error">[TEST ERROR] {text} (Refinement Iteration {error_count})</div>')
                elif type_ == "REFINER":
                    log_lines.append(f'<div class="log-line log-refiner">[REFINER AGENT] {text}</div>')
                elif type_ == "SUCCESS":
                    log_lines.append(f'<div class="log-line log-success">[TEST SUCCESS] {text}</div>')
                elif type_ == "DEPLOY":
                    log_lines.append(f'<div class="log-line log-deploy">[DEPLOY] {text}</div>')
                elif type_ == "INFO":
                    pass # already added similar header
                elif type_ == "SCAN":
                    pass # already added scanner header
                else:
                    color_class = f"log-{type_.lower()}"
                    log_lines.append(f'<div class="log-line {color_class}">[{type_}] {text}</div>')
        
        terminal_html = f"""
        <div class="terminal-window">
            <div class="terminal-bar">
                <span class="dot red"></span>
                <span class="dot yellow"></span>
                <span class="dot green"></span>
                <span class="terminal-title">Paper2Agent Pipeline Terminal</span>
            </div>
            <div class="terminal-body">
                {"".join(log_lines)}
            </div>
        </div>
        """
        terminal_output = mo.Html(terminal_html)

        # Tabbed code viewer displaying: Original tutorial, Naive Tool, and Verified Tool
        _is_failed_run = (selected_strategy.value == "paper2agent" and max_iterations.value < 3) # AlphaGenome and TISSUE need 3 iterations to succeed
        
        if selected_strategy.value == "naive":
            tool_status = mo.md("⚠️ **Warning**: The Naive Tool was generated without testing and contains hidden bugs.")
        elif _is_failed_run:
            tool_status = mo.md("❌ **Failure**: The refinement loop was cut short before passing all tests. The tool remains buggy.")
        else:
            tool_status = mo.md("✅ **Success**: The tool has passed all test verifications and replicates original results exactly.")
            
        code_tabs = mo.ui.tabs({
            "📝 Original Codebase Tutorial": mo.md(f"```python\n{_case['tutorial_code']}\n```"),
            "❌ Naive Tool Candidate (Buggy)": mo.md(f"```python\n{_case['naive_code']}\n```"),
            "✨ Paper2Agent Verified Tool": mo.md(f"```python\n{_case['verified_code']}\n```" if not _is_failed_run else "```python\n# [Refinement incomplete - code remains in buggy state]\n" + _case['naive_code'] + "\n```")
        })
        
        active_code_tabs = mo.md(f"""
        ### 📂 Code Generation Results
        
        {tool_status}
        
        {code_tabs}
        """)

    return active_code_tabs, pipeline_run, terminal_output


@app.cell(hide_code=True)
def _(pipeline_run, terminal_output):
    # Render terminal output
    terminal_output_res = mo.Html("")
    if pipeline_run:
        terminal_output_res = mo.md(f"""
        ### 🖥️ Pipeline Execution Logs
        {terminal_output}
        """)
    return (terminal_output_res,)


@app.cell(hide_code=True)
def _(active_code_tabs, pipeline_run):
    # Render code viewer tabs
    code_tabs_res = mo.Html("")
    if pipeline_run and active_code_tabs is not None:
        code_tabs_res = active_code_tabs
    return (code_tabs_res,)


@app.cell(hide_code=True)
def _(CASE_STUDIES, mo, pipeline_run, selected_paper):
    # Setup the Chat Playground UI (only clickable/visible once pipeline runs)
    if not pipeline_run:
        chat_section = mo.Html("")
        selected_prompt = None
        send_query_btn = None
    else:
        _case = CASE_STUDIES[selected_paper.value]
        prompts = _case["prompts"]
        
        selected_prompt = mo.ui.dropdown(
            options={p: p for p in prompts},
            label="💬 Select Natural Language Prompt"
        )
        
        send_query_btn = mo.ui.button(
            label="💬 Ask Paper Agent",
            kind="primary"
        )
        
        chat_controls = mo.md(f"""
        ### 💬 Agent Chat Playground
        
        Ask the conversational agent to execute operations using its toolsets.
        
        <div class="control-row">
            <div style="flex-grow: 1;">{selected_prompt}</div>
            <div>{send_query_btn}</div>
        </div>
        """)
        chat_section = mo.Html(f"<div class='card'>{chat_controls.text}</div>")
        
    return chat_section, selected_prompt, send_query_btn


@app.cell(hide_code=True)
def _(
    CASE_STUDIES,
    make_alphagenome_plot,
    make_scanpy_plot,
    make_tissue_plot,
    max_iterations,
    mo,
    selected_paper,
    selected_prompt,
    selected_strategy,
    send_query_btn,
):
    # Process the user query and render the interactive chat trace
    if selected_prompt is None or send_query_btn is None or not send_query_btn.value:
        chat_output = mo.Html("")
    else:
        _case = CASE_STUDIES[selected_paper.value]
        prompt_val = selected_prompt.value
        
        # Determine if the tool deployed successfully
        is_naive = (selected_strategy.value == "naive")
        # For simplicity, if Paper2Agent iterations < 3, it's failed/buggy
        _is_failed_run = (selected_strategy.value == "paper2agent" and max_iterations.value < 3)
        
        if is_naive or _is_failed_run:
            # Buggy execution
            chat_data = _case["naive_chat"][prompt_val]
            
            chat_html = f"""
            <div class="chat-container">
                <div class="chat-bubble chat-user">
                    {prompt_val}
                </div>
                <div class="chat-bubble chat-agent-fail">
                    <span class="badge badge-purple">Agent Thoughts</span>
                    <div class="chat-agent-thought">{chat_data['thought']}</div>
                    <span class="badge badge-red">Attempted Tool Call</span>
                    <div class="chat-agent-call-fail">run_tutorial_code()</div>
                    <span class="badge badge-red">Execution Error</span>
                    <div class="chat-agent-response">{chat_data['output']}</div>
                </div>
            </div>
            """
            chat_output = mo.md(f"""
            ### 🗨️ Conversation Trace
            {chat_html}
            """)
        else:
            # Verified tool execution
            chat_data = _case["verified_chat"][prompt_val]
            
            chat_html = f"""
            <div class="chat-container">
                <div class="chat-bubble chat-user">
                    {prompt_val}
                </div>
                <div class="chat-bubble chat-agent">
                    <span class="badge badge-purple">Agent Thoughts</span>
                    <div class="chat-agent-thought">{chat_data['thought']}</div>
                    <span class="badge badge-blue">Invoking Verified MCP Tool</span>
                    <div class="chat-agent-call">{chat_data['call']} -> {chat_data['result']}</div>
                    <span class="badge badge-green">Response</span>
                    <div class="chat-agent-response">{chat_data['output']}</div>
                </div>
            </div>
            """
            
            # Generate the plot
            plot_data = chat_data["plot_data"]
            if selected_paper.value == "alphagenome":
                fig = make_alphagenome_plot(plot_data)
            elif selected_paper.value == "tissue":
                fig = make_tissue_plot(plot_data)
            else:
                fig = make_scanpy_plot(plot_data)
                
            chat_output = mo.md(f"""
            ### 🗨️ Conversation Trace
            {chat_html}
            
            #### 📊 Results Visualizer
            {mo.as_html(fig).text}
            """)
            plt.close(fig) # free memory

    return (chat_output,)


# ----------------------------------------------------------------------------------
# HELPER FUNCTIONS AND STATIC DATASTRUCTURES (Placed at the bottom to clean up UI)
# ----------------------------------------------------------------------------------

@app.cell(hide_code=True)
def _(np, plt):
    # Plot generation helper functions
    def make_alphagenome_plot(plot_data):
        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        if plot_data["type"] == "bar":
            colors = ['#94a3b8', '#2563eb', '#64748b', '#cbd5e1']
            bars = ax.bar(plot_data["labels"], plot_data["values"], color=colors, width=0.5)
            ax.set_ylabel("Predicted variant effect score", fontsize=10, color='#475569')
            ax.set_title("AlphaGenome Variant Score Across Tissues", fontsize=11, fontweight='bold', color='#1e293b')
            ax.set_ylim(0, 1.0)
            
            for bar in bars:
                yval = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.2f}", ha='center', va='bottom', fontsize=9, fontweight='bold')
        elif plot_data["type"] == "scatter":
            x = plot_data["x"]
            y = plot_data["y"]
            labels = plot_data["labels"]
            colors = ['#94a3b8'] * len(x)
            sizes = [40] * len(x)
            
            # Highlight rs629301
            for i, l in enumerate(labels):
                if "rs629301" in l:
                    colors[i] = '#ef4444'
                    sizes[i] = 120
            
            ax.scatter(x, y, c=colors, s=sizes, zorder=3, alpha=0.9, edgecolors='#1e293b', linewidths=0.5)
            ax.axhline(y=0.5, color='#e2e8f0', linestyle='--', linewidth=1, zorder=1)
            ax.set_xlabel("Genomic Position (Mb on Chr1)", fontsize=10, color='#475569')
            ax.set_ylabel("Variant Regulatory Score", fontsize=10, color='#475569')
            ax.set_title("Manhattan-style Locus 1p13 Fine-Mapping", fontsize=11, fontweight='bold', color='#1e293b')
            
            # Annotate the causal variant
            for i, txt in enumerate(labels):
                if "rs629301" in txt:
                    ax.annotate(txt, (x[i], y[i]), textcoords="offset points", xytext=(0, 12), ha='center', fontweight='bold', color='#ef4444', fontsize=9.5)
            ax.set_ylim(0, 1.1)
            
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#cbd5e1')
        ax.spines['bottom'].set_color('#cbd5e1')
        ax.tick_params(colors='#475569', labelsize=9)
        ax.set_facecolor('#f8fafc')
        fig.patch.set_facecolor('#ffffff')
        plt.tight_layout()
        return fig

    def make_tissue_plot(plot_data):
        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        if plot_data["type"] == "tissue_interval":
            x = plot_data["x"]
            y = plot_data["y"]
            widths = plot_data["width"]
            scat = ax.scatter(x, y, c=widths, cmap="plasma", s=60, alpha=0.85, zorder=2, edgecolors='white', linewidths=0.3)
            cbar = fig.colorbar(scat, ax=ax)
            cbar.set_label("Prediction Interval Width (Uncertainty)", color='#475569', fontsize=9.5)
            cbar.ax.tick_params(colors='#475569', labelsize=8.5)
            cbar.outline.set_visible(False)
            ax.set_title("MERFISH Hypothalamus: Conformally Bound Uncertainty", fontsize=11, fontweight='bold', color='#1e293b')
            ax.set_xlabel("Spatial X Coordinate (µm)", fontsize=10, color='#475569')
            ax.set_ylabel("Spatial Y Coordinate (µm)", fontsize=10, color='#475569')
        elif plot_data["type"] == "spatial_map":
            x = plot_data["x"]
            y = plot_data["y"]
            intensity = plot_data["intensity"]
            uncertainty = plot_data["uncertainty"]
            scat = ax.scatter(x, y, c=intensity, cmap="viridis", s=uncertainty*50, alpha=0.75, zorder=2, edgecolors='white', linewidths=0.3)
            cbar = fig.colorbar(scat, ax=ax)
            cbar.set_label("Predicted Esr1 Expression Intensity", color='#475569', fontsize=9.5)
            cbar.ax.tick_params(colors='#475569', labelsize=8.5)
            cbar.outline.set_visible(False)
            ax.set_title("Esr1 Expression Map (Circle Size = Interval Uncertainty)", fontsize=11, fontweight='bold', color='#1e293b')
            ax.set_xlabel("Spatial X Coordinate (µm)", fontsize=10, color='#475569')
            ax.set_ylabel("Spatial Y Coordinate (µm)", fontsize=10, color='#475569')
            
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#cbd5e1')
        ax.spines['bottom'].set_color('#cbd5e1')
        ax.tick_params(colors='#475569', labelsize=9)
        ax.set_facecolor('#f8fafc')
        fig.patch.set_facecolor('#ffffff')
        plt.tight_layout()
        return fig

    def make_scanpy_plot(plot_data):
        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        if plot_data["type"] == "umap":
            x = plot_data["x"]
            labels = plot_data["labels"]
            unique_labels = list(set(labels))
            color_map = {
                "T-cells": "#8b5cf6",
                "B-cells": "#06b6d4",
                "Monocytes": "#10b981",
                "NK cells": "#f97316",
                "Megakaryocytes": "#ec4899"
            }
            for label in unique_labels:
                mask = [l == label for l in labels]
                ax.scatter(x[mask, 0], x[mask, 1], c=color_map.get(label, "#94a3b8"), label=label, s=40, alpha=0.85, edgecolors='white', linewidths=0.3)
            ax.legend(frameon=True, facecolor='#f8fafc', edgecolor='#e2e8f0', fontsize=8.5, loc='best')
            ax.set_title("PBMC Single-Cell Leiden Clustering UMAP", fontsize=11, fontweight='bold', color='#1e293b')
            ax.set_xlabel("UMAP 1", fontsize=10, color='#475569')
            ax.set_ylabel("UMAP 2", fontsize=10, color='#475569')
        elif plot_data["type"] == "hvg":
            mean = plot_data["mean"]
            dispersion = plot_data["dispersion"]
            is_hvg = dispersion > 1.4
            ax.scatter(mean[~is_hvg], dispersion[~is_hvg], c='#94a3b8', s=20, alpha=0.5, label="Non-variable")
            ax.scatter(mean[is_hvg], dispersion[is_hvg], c='#ef4444', s=25, alpha=0.85, label="Highly Variable (HVG)")
            ax.axhline(y=1.4, color='#cbd5e1', linestyle='--', linewidth=1)
            ax.set_xlabel("Mean Expression (log)", fontsize=10, color='#475569')
            ax.set_ylabel("Dispersion score", fontsize=10, color='#475569')
            ax.set_title("Highly Variable Genes (HVG) Feature Selection", fontsize=11, fontweight='bold', color='#1e293b')
            ax.legend(frameon=True, facecolor='#f8fafc', edgecolor='#e2e8f0', fontsize=8.5)
            
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#cbd5e1')
        ax.spines['bottom'].set_color('#cbd5e1')
        ax.tick_params(colors='#475569', labelsize=9)
        ax.set_facecolor('#f8fafc')
        fig.patch.set_facecolor('#ffffff')
        plt.tight_layout()
        return fig

    return make_alphagenome_plot, make_scanpy_plot, make_tissue_plot


@app.cell(hide_code=True)
def _(np):
    # Scientific data mock registry for the three case studies
    CASE_STUDIES = {
        "alphagenome": {
            "tutorial_code": (
                "import alphagenome as ag\n"
                "# Load model weights for liver tissue\n"
                "model = ag.load_model('liver_model')\n"
                "# Define variant chr1:1092738 A>G\n"
                "variant = ag.Variant('chr1', 1092738, 'A', 'G')\n"
                "score = model.predict(variant)\n"
                "print(f'Prediction: {score}')"
            ),
            "naive_code": (
                "def predict_variant_effect(chrom, pos, ref, alt):\n"
                "    # Naive tool generation by LLM without testing\n"
                "    import alphagenome as ag\n"
                "    model = ag.load_model('liver_model')\n"
                "    variant = ag.Variant(chrom, pos, ref, alt)\n"
                "    return model.predict(variant)"
            ),
            "verified_code": (
                "def score_variant_mcp(chrom, pos, ref, alt, tissue='liver'):\n"
                "    # Verified and refined by Paper2Agent Test-Verifier-Improver\n"
                "    import alphagenome as ag\n"
                "    \n"
                "    # Download weights if not cached\n"
                "    ag.utils.download_model_weights(tissue)\n"
                "    \n"
                "    # Correct API: SequenceModel.from_pretrained\n"
                "    model = ag.models.SequenceModel.from_pretrained(tissue)\n"
                "    \n"
                "    # Correct API: Variant resides in models submodule\n"
                "    variant = ag.models.Variant(chrom, pos, ref, alt)\n"
                "    score = model.predict(variant)\n"
                "    return {'score': float(score), 'status': 'success'}"
            ),
            "verified_log": [
                ("INFO", "Initializing environment and installing dependencies (alphagenome>=1.0.0)..."),
                ("SCAN", "Tutorial Scanner: Detected tutorial file 'examples/predict_effect.py'. Running..."),
                ("LLM", "Tool Extractor: Extracting tool candidate 'predict_variant_effect(chrom, pos, ref, alt)'. Code generated."),
                ("TEST", "Test Verifier: Running generated test script on candidate tool..."),
                ("ERROR", "module 'alphagenome' has no attribute 'load_model' in line 5. (API version mismatch)"),
                ("REFINER", "Inspecting imports. Found 'from alphagenome.models import SequenceModel' is preferred. Modifying tool..."),
                ("TEST", "Test Verifier: Running test script again..."),
                ("ERROR", "FileNotFoundError: model weights 'liver_model' not found. (Weights need separate download)"),
                ("REFINER", "Adding 'ag.utils.download_model_weights(tissue)' utility function before loading model..."),
                ("TEST", "Test Verifier: Running test script again..."),
                ("SUCCESS", "Test passed successfully! Output value 0.85 matches expected value 0.85 from tutorial."),
                ("DEPLOY", "MCP server tool 'score_variant_mcp' successfully verified and packaged!")
            ],
            "prompts": [
                "Score the variant rs1626703 (chr1:1092738:A>G) in Liver tissue",
                "Identify the causal variant for LDL cholesterol at the chromosome 1p13 locus"
            ],
            "naive_chat": {
                "Score the variant rs1626703 (chr1:1092738:A>G) in Liver tissue": {
                    "thought": "The user wants to score a variant. I will write a script to load the AlphaGenome library and run prediction.",
                    "output": "I will run the AlphaGenome prediction pipeline. However, the execution failed with the following traceback:\n```\nAttributeError: module 'alphagenome' has no attribute 'load_model'\n```\nThis indicates that the paper's library API has changed. I am unable to resolve the correct method call from the repository structure alone.",
                    "success": False
                },
                "Identify the causal variant for LDL cholesterol at the chromosome 1p13 locus": {
                    "thought": "I need to parse the GWAS credible set for LDL cholesterol and score all variants.",
                    "output": "I ran the script but got a FileNotFoundError when downloading the training weights for AlphaGenome. The original codebase references a local model path that doesn't exist on my machine. I cannot proceed without resolving these model assets.",
                    "success": False
                }
            },
            "verified_chat": {
                "Score the variant rs1626703 (chr1:1092738:A>G) in Liver tissue": {
                    "thought": "The user wants to score rs1626703 in Liver tissue. I have a verified MCP tool 'score_variant_mcp' that handles this. I will call it.",
                    "call": "score_variant_mcp(chrom='chr1', pos=1092738, ref='A', alt='G', tissue='liver')",
                    "result": '{"score": 0.85, "status": "success"}',
                    "output": "Based on the AlphaGenome model, the variant **rs1626703 (chr1:1092738:A>G)** has a highly significant predicted functional score of **0.85** in Liver tissue (compared to a baseline average of 0.12). This variant is predicted to disrupt a splicing motif, leading to elevated gene expression of **MPHOSPH9**, suggesting a causal mechanism for altered lipid levels.",
                    "success": True,
                    "plot_data": {"type": "bar", "labels": ["Baseline", "rs1626703 (Liver)", "rs1626703 (Brain)", "rs1626703 (Heart)"], "values": [0.12, 0.85, 0.21, 0.08]}
                },
                "Identify the causal variant for LDL cholesterol at the chromosome 1p13 locus": {
                    "thought": "The user wants to identify the causal gene at the 1p13 locus. I can query the GWAS credible set and score variants.",
                    "call": "score_locus_variants_mcp(locus='1p13', trait='ldl')",
                    "result": '{"causal_variant": "rs629301", "prioritized_gene": "SORT1", "score": 0.94}',
                    "output": "The Paper Agent analyzed 15 variants in the 1p13 credible set. The variant **rs629301** scored **0.94**, far outperforming other variants. The model prioritizes **SORT1** as the causal gene, which is highly expressed in liver and key to LDL regulation. This matches clinical observations of SORT1 hepatocyte expression, resolving ambiguity from previous studies that focused on CELSR2.",
                    "success": True,
                    "plot_data": {"type": "scatter", "x": [109.1, 109.2, 109.25, 109.3, 109.4], "y": [0.15, 0.22, 0.94, 0.08, 0.11], "labels": ["rs111", "rs222", "rs629301 (SORT1)", "rs333", "rs444"]}
                }
            }
        },
        "tissue": {
            "tutorial_code": (
                "import tissue as ts\n"
                "adata = ts.load_dataset('merfish_hypothalamus')\n"
                "res = ts.run_pipeline(adata, method='conformal')\n"
                "print(res.prediction_intervals)"
            ),
            "naive_code": (
                "def run_tissue_analysis(dataset_name):\n"
                "    # Naive tool candidate by LLM\n"
                "    import tissue as ts\n"
                "    adata = ts.load_dataset(dataset_name)\n"
                "    return ts.run_pipeline(adata, method='conformal')"
            ),
            "verified_code": (
                "def compute_conformal_intervals(dataset='merfish_hypothalamus', alpha=0.1):\n"
                "    # Verified and refined by Paper2Agent\n"
                "    import tissue as ts\n"
                "    import scanpy as sc\n"
                "    \n"
                "    adata = ts.datasets.load_spatial_data(dataset)\n"
                "    # Conformal prediction requires a separate calibration split\n"
                "    cal_data, test_data = ts.utils.split_calibration(adata, test_size=0.2)\n"
                "    \n"
                "    # Correct API: wrap AnnData in TISSUEDataWrapper\n"
                "    wrapped_cal = ts.TISSUEDataWrapper(cal_data)\n"
                "    wrapped_test = ts.TISSUEDataWrapper(test_data)\n"
                "    \n"
                "    results = ts.run_pipeline(wrapped_test, calibration_data=wrapped_cal, alpha=alpha, method='conformal')\n"
                "    return {'intervals': results.prediction_intervals.tolist(), 'status': 'success'}"
            ),
            "verified_log": [
                ("INFO", "Initializing environment and installing dependencies (tissue-sc>=0.8.2)..."),
                ("SCAN", "Tutorial Scanner: Found tutorial notebook 'conformal_prediction.ipynb'. Running..."),
                ("LLM", "Tool Extractor: Extracting tool candidate 'run_tissue_pipeline(dataset_name)'. Code generated."),
                ("TEST", "Test Verifier: Running generated test script on candidate tool..."),
                ("ERROR", "TypeError: run_pipeline() missing 1 required positional argument: 'calibration_data' in line 12."),
                ("REFINER", "Conformal prediction requires a calibration dataset. Extracting split logic from tutorial..."),
                ("TEST", "Test Verifier: Running test script again..."),
                ("ERROR", "ValueError: data must be wrapped in TISSUEDataWrapper in line 18."),
                ("REFINER", "Wrapping AnnData in TISSUEDataWrapper before passing to the model. Modifying code..."),
                ("TEST", "Test Verifier: Running test script again..."),
                ("SUCCESS", "Test passed successfully! Output prediction intervals match expected results."),
                ("DEPLOY", "MCP server tool 'compute_conformal_intervals' successfully verified and packaged!")
            ],
            "prompts": [
                "Compute conformal prediction intervals for merfish_hypothalamus dataset",
                "Plot spatial gene expression uncertainty maps"
            ],
            "naive_chat": {
                "Compute conformal prediction intervals for merfish_hypothalamus dataset": {
                    "thought": "I will run TISSUE prediction intervals.",
                    "output": "Execution failed with the following traceback:\n```\nTypeError: run_pipeline() missing 1 required positional argument: 'calibration_data'\n```\nTISSUE's API requires calibration datasets. I am unsure how the calibration dataset is structured in this repository.",
                    "success": False
                },
                "Plot spatial gene expression uncertainty maps": {
                    "thought": "I will plot uncertainty maps.",
                    "output": "I got an error trying to extract prediction intervals: `ValueError: data must be wrapped in TISSUEDataWrapper`. I don't know how to initialize this wrapper class without a tutorial context.",
                    "success": False
                }
            },
            "verified_chat": {
                "Compute conformal prediction intervals for merfish_hypothalamus dataset": {
                    "thought": "I will call the verified tool.",
                    "call": "compute_conformal_intervals(dataset='merfish_hypothalamus', alpha=0.1)",
                    "result": '{"interval_width_mean": 2.45, "status": "success"}',
                    "output": "TISSUE successfully computed conformal prediction intervals (alpha=0.1) for the MERFISH hypothalamus dataset. The mean prediction interval width across cells is **2.45**, reflecting high model confidence in the preoptic area and lower confidence in the posterior regions.",
                    "success": True,
                    "plot_data": {"type": "tissue_interval", "x": np.random.uniform(10, 50, 100), "y": np.random.uniform(10, 50, 100), "width": np.random.exponential(1.5, 100)}
                },
                "Plot spatial gene expression uncertainty maps": {
                    "thought": "I will call the verified tool to get spatial plotting coordinates.",
                    "call": "plot_uncertainty_spatial(dataset='merfish_hypothalamus', gene='Esr1')",
                    "result": '{"plot_saved": true, "status": "success"}',
                    "output": "The uncertainty map for *Esr1* expression was successfully generated. You can see the spatial coordinate grid below, where node sizes represent the conformal prediction interval width (uncertainty). The preoptic area (upper-left) shows highly confident predictions, whereas the peripheral regions show wider intervals.",
                    "success": True,
                    "plot_data": {"type": "spatial_map", "x": np.random.normal(30, 8, 150), "y": np.random.normal(30, 8, 150), "intensity": np.random.uniform(0.1, 1.0, 150), "uncertainty": np.random.uniform(0.5, 3.0, 150)}
                }
            }
        },
        "scanpy": {
            "tutorial_code": (
                "import scanpy as sc\n"
                "adata = sc.read_10x_mtx('data/')\n"
                "sc.pp.filter_cells(adata, min_genes=200)\n"
                "sc.pp.normalize_total(adata, target_sum=1e4)\n"
                "sc.pp.log1p(adata)\n"
                "sc.pp.highly_variable_genes(adata)\n"
                "sc.tl.pca(adata)\n"
                "sc.tl.umap(adata)\n"
                "sc.tl.leiden(adata)"
            ),
            "naive_code": (
                "def preprocess_and_cluster(data_path):\n"
                "    # Naive tool candidate by LLM\n"
                "    import scanpy as sc\n"
                "    adata = sc.read_10x_mtx(data_path)\n"
                "    sc.pp.filter_cells(adata, min_genes=200)\n"
                "    sc.pp.normalize_total(adata, target_sum=1e4)\n"
                "    sc.pp.log1p(adata)\n"
                "    sc.tl.pca(adata)\n"
                "    sc.tl.umap(adata)\n"
                "    sc.tl.leiden(adata)\n"
                "    return adata"
            ),
            "verified_code": (
                "def run_scanpy_pipeline(data_path, cluster_resolution=0.5):\n"
                "    # Verified and refined by Paper2Agent\n"
                "    import scanpy as sc\n"
                "    \n"
                "    adata = sc.read_10x_mtx(data_path)\n"
                "    sc.pp.filter_cells(adata, min_genes=200)\n"
                "    sc.pp.normalize_total(adata, target_sum=1e4)\n"
                "    sc.pp.log1p(adata)\n"
                "    sc.pp.highly_variable_genes(adata)\n"
                "    sc.tl.pca(adata)\n"
                "    \n"
                "    # Correct pipeline requires computing neighborhood graph first\n"
                "    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=20)\n"
                "    sc.tl.umap(adata)\n"
                "    sc.tl.leiden(adata, resolution=cluster_resolution)\n"
                "    return {'clusters': adata.obs['leiden'].tolist(), 'umap': adata.obsm['X_umap'].tolist(), 'status': 'success'}"
            ),
            "verified_log": [
                ("INFO", "Initializing environment and installing dependencies (scanpy>=1.9.3)..."),
                ("SCAN", "Tutorial Scanner: Found tutorial script 'scanpy_pbmc3k.py'. Running..."),
                ("LLM", "Tool Extractor: Extracting tool candidate 'preprocess_and_cluster(data_path)'. Code generated."),
                ("TEST", "Test Verifier: Running generated test script on candidate tool..."),
                ("ERROR", "KeyError: 'neighbors' needed for leiden clustering in line 15."),
                ("REFINER", "Leiden clustering requires a prior neighborhood graph computation. Adding 'sc.pp.neighbors(adata)' before clustering..."),
                ("TEST", "Test Verifier: Running test script again..."),
                ("SUCCESS", "Test passed successfully! Output cell clusters match tutorial results."),
                ("DEPLOY", "MCP server tool 'run_scanpy_pipeline' successfully verified and packaged!")
            ],
            "prompts": [
                "Cluster the PBMC dataset and annotate cell types",
                "Plot highly variable genes in single-cell data"
            ],
            "naive_chat": {
                "Cluster the PBMC dataset and annotate cell types": {
                    "thought": "I will run the preprocessing code.",
                    "output": "Execution failed with the following traceback:\n```\nKeyError: 'neighbors' needed for leiden clustering\n```\nI omitted the neighborhood graph construction step, which is required by Scanpy before running the Leiden clustering algorithm.",
                    "success": False
                },
                "Plot highly variable genes in single-cell data": {
                    "thought": "I will call the highly variable genes plot.",
                    "output": "The data preprocessing failed because I ran log1p on raw counts without normalizing them first, resulting in skewed highly variable genes selection.",
                    "success": False
                }
            },
            "verified_chat": {
                "Cluster the PBMC dataset and annotate cell types": {
                    "thought": "I will call the verified tool.",
                    "call": "run_scanpy_pipeline(data_path='pbmc3k_raw.h5ad', cluster_resolution=0.5)",
                    "result": '{"n_clusters": 5, "status": "success"}',
                    "output": "Scanpy successfully preprocessed the dataset (QC filtering, normalization, log transformation, PCA, neighborhood graph, UMAP) and clustered the cells using Leiden. 5 clusters were identified: T-cells, B-cells, Monocytes, NK cells, and Megakaryocytes.",
                    "success": True,
                    "plot_data": {"type": "umap", "x": np.random.normal(size=(200, 2)), "labels": np.random.choice(["T-cells", "B-cells", "Monocytes", "NK cells", "Megakaryocytes"], 200)}
                },
                "Plot highly variable genes in single-cell data": {
                    "thought": "I will use the verified pipeline tool to extract variable genes.",
                    "call": "plot_variable_genes_mcp(data_path='pbmc3k_raw.h5ad')",
                    "result": '{"n_genes": 2000, "status": "success"}',
                    "output": "Highly variable genes were identified using the standardized Scanpy pipeline. The plot below displays the dispersion versus mean expression for all genes, highlighting the top 2,000 highly variable genes in red.",
                    "success": True,
                    "plot_data": {"type": "hvg", "mean": np.random.exponential(1.0, 500), "dispersion": np.random.normal(1.2, 0.6, 500)}
                }
            }
        }
    }
    return (CASE_STUDIES,)


if __name__ == "__main__":
    app.run()
