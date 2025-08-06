import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# Define file path
# file_path = "prediction/results/predictive_avg_scores_5_models.csv"
file_path = "results/gene_prediction/predictive_avg_scores_7_models.csv"

# Read the CSV file
df = pd.read_csv(file_path, index_col=0)

# Define colors for each gene group
colors = ["green", "red", "blue", "orange"]
group_labels = df.columns.tolist()  # Extract column names
models = df.index.tolist()  # Extract row names

# Set bar width and positions
bar_width = 0.22
x = np.arange(len(models))  # The x locations for the models

# Create the plot
plt.figure(figsize=(14, 7))

# Plot bars for each gene group
for i, (label, color) in enumerate(zip(group_labels, colors)):
    plt.bar(x + i * bar_width, df[label], width=bar_width, label=label, color=color, edgecolor="black")

# Set labels and title
plt.ylabel("Average Score", fontsize=16)
plt.xticks(x + bar_width * 1.5, models, fontsize=16) 

# **Increase distance of x-tick labels from the x-axis**
plt.gca().tick_params(axis='x', pad=10)  # Moves x-tick labels further down
plt.subplots_adjust(bottom=0.18)  # Adjust bottom margin

# Add value labels on top of bars
for i in range(len(models)):
    for j in range(len(group_labels)):
        plt.text(
            x[i] + j * bar_width, 
            df.iloc[i, j] + 0.02, 
            f"{df.iloc[i, j]:.4f}", 
            ha="center", 
            fontsize=6, 
            # fontweight="bold"
        )

# Adjust bottom margin to create space for the legend
plt.subplots_adjust(bottom=0.3)  # Increase bottom margin

# Show legend at the bottom with more space
plt.legend(
    loc="lower center", bbox_to_anchor=(0.5, -0.25),  
    fontsize=14, ncol=len(group_labels), frameon=False
)

plt.ylim(0, 1.2)  # Set y-axis limit from 0 to 1.2

# Save and show the plot
plt.savefig("results/predictive_avg_scores_7_models_plot.png", dpi=300, bbox_inches="tight")
plt.show()
