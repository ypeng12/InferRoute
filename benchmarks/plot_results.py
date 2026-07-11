"""
Plotting and AIQ calculation script for InferRoute.

This script parses evaluation results from benchmarks/results/eval_results.json,
computes average metrics, groups swept scenarios to trace the cost-quality
curves (Zero Router, KNN Router, MLP Router), computes the AIQ (Area Under the Curve)
metric, and plots the results.

Inspired by:
"ROUTERBENCH: A Benchmark for Multi-LLM Routing System" (withmartian/routerbench)
"""

import os
import json
from typing import List, Dict, Tuple

# Try to import pandas and matplotlib for visualization
HAS_PLOT_LIBS = False
try:
    import pandas as pd
    import matplotlib.pyplot as plt
    HAS_PLOT_LIBS = True
except ImportError:
    pass

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
RESULTS_PATH = os.path.join(RESULTS_DIR, "eval_results.json")
SUMMARY_PATH = os.path.join(RESULTS_DIR, "evaluation_summary.md")

def calculate_auc(points: List[Tuple[float, float]]) -> float:
    """
    Computes the Area Under the Curve (AUC) using the trapezoidal rule,
    normalized by the cost range (c_max - c_min) of the curve.
    This yields the average quality efficiency score (0.0 to 1.0) of the router.
    """
    sorted_pts = sorted(points, key=lambda x: x[0])
    if len(sorted_pts) < 2:
        return 0.0
    area = 0.0
    for i in range(len(sorted_pts) - 1):
        c1, q1 = sorted_pts[i]
        c2, q2 = sorted_pts[i+1]
        area += 0.5 * (q1 + q2) * (c2 - c1)
    
    cost_range = sorted_pts[-1][0] - sorted_pts[0][0]
    if cost_range > 0:
        return area / cost_range
    return sorted_pts[0][1]

def analyze_results():
    if not os.path.exists(RESULTS_PATH):
        print(f"Error: Results file not found at {RESULTS_PATH}. Please run run_router_eval.py first.")
        return

    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Group data by scenario
    scenarios: Dict[str, List[dict]] = {}
    for entry in data:
        sc = entry["scenario"]
        if sc not in scenarios:
            scenarios[sc] = []
        scenarios[sc].append(entry)

    summary_stats = {}
    
    # Average stats per scenario
    for name, entries in scenarios.items():
        total = len(entries)
        avg_cost = sum(e["cost_usd"] for e in entries) / total
        avg_quality = sum(e["quality_score"] for e in entries) / total
        avg_latency = sum(e["latency_ms"] for e in entries) / total
        avg_ttft = sum(e["ttft_ms"] for e in entries) / total
        slo_met_percent = (sum(1 for e in entries if e["slo_compliant"]) / total) * 100.0
        fallback_percent = (sum(1 for e in entries if e["fallback_triggered"]) / total) * 100.0
        
        summary_stats[name] = {
            "avg_cost": avg_cost,
            "avg_quality": avg_quality,
            "avg_latency": avg_latency,
            "avg_ttft": avg_ttft,
            "slo_compliance": slo_met_percent,
            "fallback_rate": fallback_percent
        }

    # Group curves
    curves = {
        "zero-router": [],
        "knn-router": [],
        "mlp-router": [],
        "cascade-router": []
    }
    
    for name, stats in summary_stats.items():
        for prefix in curves.keys():
            if name.startswith(prefix):
                curves[prefix].append((stats["avg_cost"], stats["avg_quality"]))

    # Sort curve points by cost
    for k in curves.keys():
        curves[k] = sorted(curves[k], key=lambda x: x[0])

    # Calculate AIQ (Area under the cost-quality curve)
    aiq_scores = {}
    for k, pts in curves.items():
        aiq_scores[k] = calculate_auc(pts)

    print("\n=============================================================")
    print("InferRoute RouterBench Policy Evaluation Summary Results")
    print("=============================================================")
    
    markdown_lines = [
        "# 📊 InferRoute RouterBench & FrugalGPT Evaluation Summary\n",
        "Inspired by the RouterBench framework (`withmartian/routerbench`) and FrugalGPT cascading LLMs, this report evaluates routing policies on cost, quality, and SLA compliance. We plot the Pareto curves by sweeping the willingness-to-pay ($\\lambda$), mixture ratio ($p$), and cascade threshold ($\\tau$).\n",
        "## 📈 Curve Efficiency: AIQ (Area Under the Trade-off Curve)",
        "AIQ measures the average quality efficiency score of a router across its swept cost range (normalized AUC, bounded between 0% and 100%). Higher is better.\n",
        "| Routing Curve | AIQ Score (Normalized AUC) | Description |",
        "| :--- | :--- | :--- |",
        f"| **Oracle Router Upper Bound** | *Theoretical Optimal* | Represents the perfect offline selection. |",
        f"| **Cascade Router (FrugalGPT)** | {aiq_scores['cascade-router'] * 100:.1f}% | Server-side cascading model escalation. |",
        f"| **KNN Router** | {aiq_scores['knn-router'] * 100:.1f}% | Jaccard similarity nearest-neighbor routing. |",
        f"| **MLP Router** | {aiq_scores['mlp-router'] * 100:.1f}% | Content-aware classifier routing. |",
        f"| **Zero Router Baseline** | {aiq_scores['zero-router'] * 100:.1f}% | Non-content-aware random model mixture. |\n",
        "## 📋 Comprehensive Performance Table",
        "| Scenario | Avg Cost ($ USD) | Avg Quality (0-1) | Avg Latency (ms) | Avg TTFT (ms) | SLO Compliance (%) | Fallback Rate (%) |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
    ]

    # Sort scenarios alphabetically for display, but keep sweeps grouped
    sorted_scenarios = sorted(summary_stats.keys())
    for name in sorted_scenarios:
        stats = summary_stats[name]
        row = (
            f"| **{name}** "
            f"| ${stats['avg_cost']:.6f} "
            f"| {stats['avg_quality']:.2f} "
            f"| {stats['avg_latency']:.1f}ms "
            f"| {stats['avg_ttft']:.1f}ms "
            f"| {stats['slo_compliance']:.1f}% "
            f"| {stats['fallback_rate']:.1f}% |"
        )
        print(f"Scenario: {name:<22} | Cost: ${stats['avg_cost']:.6f} | Quality: {stats['avg_quality']:.2f} | Latency: {stats['avg_latency']:.1f}ms | SLO: {stats['slo_compliance']:.1f}%")
        markdown_lines.append(row)

    # Save Markdown Summary Table
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(markdown_lines))
    print(f"\nMarkdown summary saved to {SUMMARY_PATH}")

    # Generate charts if libraries are present
    if HAS_PLOT_LIBS:
        print("\nPlotting libraries detected. Generating cost-quality trade-off curves...")
        
        # 1. Cost vs Quality Pareto Frontier Plot
        plt.figure(figsize=(10, 6))
        
        # Colors
        colors = {
            "zero-router": "#e74c3c",
            "knn-router": "#3498db",
            "mlp-router": "#9b59b6",
            "cascade-router": "#2ecc71"
        }
        
        # Plot curves
        for key, pts in curves.items():
            if not pts:
                continue
            costs, qualities = zip(*pts)
            # Scale cost to micro-USD
            scaled_costs = [c * 1_000_000 for c in costs]
            label = f"{key.upper()} (AIQ={aiq_scores[key]:.2f})"
            plt.plot(scaled_costs, qualities, 'o-', label=label, color=colors[key], linewidth=2.5, markersize=8)

        # Plot individual reference points
        ref_points = {
            "always-openai": "#27ae60",
            "always-gemini": "#1abc9c",
            "always-vllm": "#f1c40f",
            "always-ollama": "#e67e22",
            "rule-router": "#34495e",
            "oracle-router": "#d35400"
        }
        
        for name, color in ref_points.items():
            if name in summary_stats:
                st = summary_stats[name]
                sc_cost = st["avg_cost"] * 1_000_000
                marker = '*' if 'oracle' in name else 's'
                size = 180 if 'oracle' in name else 100
                plt.scatter(sc_cost, st["avg_quality"], color=color, marker=marker, s=size, zorder=5, label=name)
                plt.text(sc_cost + 0.5, st["avg_quality"] - 0.01, f" {name}", fontsize=8, weight="bold")
            
        plt.xlabel("Average Cost per Request (USD per 1 Million Prompts)", fontsize=11)
        plt.ylabel("Average Response Quality Score (0 - 1.0)", fontsize=11)
        plt.title("Cost-Quality Pareto Trade-off Curves (RouterBench Framework)", fontsize=13, weight="bold")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(loc="lower right", fontsize=9)
        plt.tight_layout()
        
        plot_path = os.path.join(RESULTS_DIR, "cost_quality_frontier.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Saved Cost-Quality trade-off chart to {plot_path}")
        
        # 2. Latency TTFT vs Total Latency Bar Plot (Only main representative scenarios)
        main_scenarios = [
            "always-openai", "always-gemini", "always-vllm", "always-ollama",
            "rule-router", "oracle-router", "cascade-router_t0.60",
            "knn-router_l1.00", "mlp-router_l1.00"
        ]
        
        plt.figure(figsize=(10, 6))
        plot_data = []
        for name in main_scenarios:
            if name in summary_stats:
                st = summary_stats[name]
                st["name"] = name
                plot_data.append(st)
                
        if plot_data:
            df_plot = pd.DataFrame(plot_data)
            x = range(len(df_plot))
            width = 0.35
            
            plt.bar([i - width/2 for i in x], df_plot["avg_ttft"], width, label="Avg TTFT (ms)", color="#3498db")
            plt.bar([i + width/2 for i in x], df_plot["avg_latency"], width, label="Avg Total Latency (ms)", color="#2ecc71")
            
            plt.xticks(x, df_plot["name"], rotation=20, ha="right", fontsize=9, weight="bold")
            plt.ylabel("Duration (milliseconds)", fontsize=11)
            plt.title("Latency Performance Comparison (TTFT vs. Total Latency)", fontsize=13, weight="bold")
            plt.legend()
            plt.grid(True, axis="y", linestyle="--", alpha=0.5)
            plt.tight_layout()
            
            latency_plot_path = os.path.join(RESULTS_DIR, "latency_comparison.png")
            plt.savefig(latency_plot_path, dpi=150)
            plt.close()
            print(f"Saved Latency chart to {latency_plot_path}")
    else:
        print("\nWarning: pandas/matplotlib not found. Skipping PNG chart generation.")
        print("To generate cost-quality plots, please run: pip install pandas matplotlib")

if __name__ == "__main__":
    analyze_results()
