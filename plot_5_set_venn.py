import pandas as pd
import matplotlib.pyplot as plt
from venn import venn  # Requires the `venn` package: pip install venn

def read_genes(file_path):
    """Read gene names from a CSV file and return as a set."""
    try:
        df = pd.read_csv(file_path)  # Ensure correct delimiter if needed
        return set(df["Gene"])  # Extract unique gene names
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return set()

def plot_venn_diagram():
    """Plot a large 5-set Venn diagram with optimized margins and layout."""
    # Define file paths
    file_paths = {
        "EGCN": "prediction/results/EGCN_CPDB_predicted_driver_genes_epo200_2048.csv",
        "ChebNet": "prediction/results/ChebNet_CPDB_predicted_driver_genes_epo200_2048.csv",
        "ChebNetII": "prediction/results/ChebNetII_CPDB_predicted_driver_genes_epo200_2048.csv",
        "EMOGI": "prediction/results/EMOGI_CPDB_predicted_driver_genes_epo1027_2048.csv",
        "GCN": "prediction/results/GCN_CPDB_predicted_driver_genes_epo1027_2048.csv"
    }

    # Read data from files
    gene_sets = {model: read_genes(path) for model, path in file_paths.items()}

    # Create the Venn diagram
    fig, ax = plt.subplots(figsize=(12, 10))  # Increased figure size for a larger plot
    venn_plot = venn(gene_sets, ax=ax)

    # Create a legend bar at the bottom
    colors = ["purple", "blue", "cyan", "lightgreen", "yellow"]
    labels = list(gene_sets.keys())
    handles = [plt.Line2D([0], [0], color=col, lw=8) for col in colors]

    legend = ax.legend(
        handles, labels,
        loc="lower center",
        #bbox_to_anchor=(0.5, -0.15),  # place below the plot
        bbox_to_anchor=(0.5, -0.05),
        ncol=len(labels),
        frameon=False,
        fontsize=14,
        handlelength=1.5,
        columnspacing=1.0
    )

    # Minimize margins
    plt.tight_layout(pad=0.2)
    plt.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.18)  # leave space for bottom legend

    # Save and show the plot
    plt.savefig("prediction/results/venn_diagram.png", bbox_inches="tight", dpi=300)
    plt.show()

# Call the function
plot_venn_diagram()
