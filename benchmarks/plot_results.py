import os
import json
from typing import List

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

def analyze_results():
    if not os.path.exists(RESULTS_PATH):
        print(f"Error: Results file not found at {RESULTS_PATH}. Please run run_router_eval.py first.")
        return

    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Group data by scenario manually (pure Python fallback)
    scenarios: dict[str, List[dict]] = {}
    for entry in data:
        sc = entry["scenario"]
        if sc not in scenarios:
            scenarios[sc] = []
        scenarios[sc].append(entry)

    summary_stats = {}
    
    print("\n=============================================================")
    print("InferRoute Router Evaluation Summary Results")
    print("=============================================================")
    
    markdown_lines = [
        "# 📊 InferRoute Router Benchmark Evaluation Summary\n",
        "| Router Scenario | Avg Cost ($ USD) | Avg Quality Score (0-1) | Avg Latency (ms) | Avg TTFT (ms) | SLO Compliance (%) | Fallback Rate (%) |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
    ]

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
        
        row = (
            f"| **{name}** "
            f"| ${avg_cost:.6f} "
            f"| {avg_quality:.2f} "
            f"| {avg_latency:.1f}ms "
            f"| {avg_ttft:.1f}ms "
            f"| {slo_met_percent:.1f}% "
            f"| {fallback_percent:.1f}% |"
        )
        print(f"Scenario: {name:<22} | Cost: ${avg_cost:.6f} | Quality: {avg_quality:.2f} | Latency: {avg_latency:.1f}ms | TTFT: {avg_ttft:.1f}ms | SLO: {slo_met_percent:.1f}% | Fallback: {fallback_percent:.1f}%")
        markdown_lines.append(row)

    # Save Markdown Summary Table
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(markdown_lines))
    print(f"\nMarkdown summary saved to {SUMMARY_PATH}")

    # Generate charts if libraries are present
    if HAS_PLOT_LIBS:
        print("\nPlotting libraries detected. Generating Pareto frontier charts...")
        df = pd.DataFrame(data)
        
        # 1. Cost vs Quality Pareto Frontier Plot
        plt.figure(figsize=(10, 6))
        grouped = df.groupby("scenario").mean(numeric_only=True).reset_index()
        
        for idx, row in grouped.iterrows():
            plt.scatter(row["cost_usd"], row["quality_score"], s=150, label=row["scenario"], alpha=0.8)
            plt.text(row["cost_usd"] + (row["cost_usd"]*0.01), row["quality_score"], f"  {row['scenario']}", fontsize=9, weight="bold")
            
        plt.xlabel("Average Cost per Request ($ USD)", fontsize=12)
        plt.ylabel("Average Response Quality Score (0 - 1.0)", fontsize=12)
        plt.title("Cost-Quality Pareto Frontier Curve for LLM Routers", fontsize=14, weight="bold")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        
        plot_path = os.path.join(RESULTS_DIR, "cost_quality_frontier.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Saved Cost-Quality chart to {plot_path}")
        
        # 2. Latency TTFT vs Total Latency Bar Plot
        plt.figure(figsize=(10, 6))
        x = range(len(grouped))
        width = 0.35
        
        plt.bar([i - width/2 for i in x], grouped["ttft_ms"], width, label="Avg TTFT (ms)", color="#3498db")
        plt.bar([i + width/2 for i in x], grouped["latency_ms"], width, label="Avg Total Latency (ms)", color="#2ecc71")
        
        plt.xticks(x, grouped["scenario"], rotation=15, ha="right", fontsize=9, weight="bold")
        plt.ylabel("Duration (milliseconds)", fontsize=12)
        plt.title("Latency Performance Comparison (TTFT vs. Total Latency)", fontsize=14, weight="bold")
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
