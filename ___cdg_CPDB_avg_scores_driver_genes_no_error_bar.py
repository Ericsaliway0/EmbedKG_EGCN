import matplotlib.pyplot as plt
import numpy as np
import os

# Data for the plot
networks = ['CPDB', 'STRING', 'HIPPIE']
categories = ['Canonical driver genes', 'Potential driver genes', 'Nondriver genes', 'Rest of genes']
predicted_scores = [
    [0.84709316, 0.840757, 0.038428705, 0.13210054],  # CPDB scores
    [0.9904424426744190, 0.9981801071284640, 0.030308282997256600, 0.44687549206073200],   # STRING scores
    [0.9849851791129030, 0.9974297007829980, 0.08101516925777780, 0.5473833623830690],   # HIPPIE scores
]
errors = [
    [0.029387883, 0.01634637, 0.01912501, 0.013932662],  # CPDB error bars
    [0.021346225, 0.02605839, 0.01055604, 0.01839576],   # STRING error bars
    [0.022136463, 0.02836824, 0.01134865, 0.01707403],  # HIPPIE error bars
]
p_values = [
    ['4.15E-05', '7.70E-26', '1.61E-43', '6.44E-201'],  # CPDB
    ['1.021E-15', '6.643E-193', '5.93E-272', '2.36E-276'],  # STRING
    ['2.31E-16', '1.21E-160', '7.94E-220', '3.28E-228'],  # HIPPIE
]

# Bar width and spacing
bar_width = 0.8
group_spacing = 0.8
bar_spacing = 0.3
x = np.arange(len(networks)) * (len(categories) * (bar_width + bar_spacing) + group_spacing)

# Colors for the bars
colors = ['#a3c4f3', '#8eecf5', '#f1c0e8', '#b9fbc0']

# Create the figure and axes
fig, ax = plt.subplots(figsize=(8, 5))

# Scale predicted scores to match max height of ~0.3
scaling_factor = 0.3 / 0.5
scaled_scores = [[score * scaling_factor for score in network] for network in predicted_scores]
scaled_errors = [[error * scaling_factor for error in network] for network in errors]

# Plot the bars with error bars
error_bar_settings = {
    'elinewidth': 1.0,
    'capsize': 2,
    'capthick': 0.8
}

for i, category in enumerate(categories):
    category_scores = [scaled_scores[j][i] for j in range(len(networks))]
    category_errors = [scaled_errors[j][i] for j in range(len(networks))]
    positions = x + i * (bar_width + bar_spacing)
    ax.bar(positions, category_scores, bar_width, label=category, color=colors[i], yerr=category_errors, error_kw=error_bar_settings)

# Add brackets (lines) with p-values for each network group
general_offset = 0.05
line_height_increment = 0.01
p_value_text_offset = line_height_increment + 0.005  # Consistent text offset for all p-values

# Adjust the spacing between the bottom line and the error bars
error_bar_margin = 0.02  # Space between the error bars and the bottom line
p_value_text_offset = line_height_increment + 0.01  # Increased offset for more space between lines and numbers

for i, network in enumerate(networks):
    x_group = x[i]
    bar_positions = [x_group + j * (bar_width + bar_spacing) for j in range(len(categories))]
    heights = [scaled_scores[i][j] + scaled_errors[i][j] for j in range(len(categories))]

    # Bottom line: 2nd to 4th bar
    y = max(heights[1], heights[3]) + error_bar_margin  # Add margin to separate from error bars
    ax.plot([bar_positions[1], bar_positions[1], bar_positions[3], bar_positions[3]], 
            [y, y + line_height_increment, y + line_height_increment, y], color='black', linewidth=1)
    ax.text((bar_positions[1] + bar_positions[3]) / 2, y + p_value_text_offset, p_values[i][3], ha='center', fontsize=8)

    # Third line: 1st to 2nd bar
    y += general_offset
    ax.plot([bar_positions[0], bar_positions[0], bar_positions[1], bar_positions[1]], 
            [y, y + line_height_increment, y + line_height_increment, y], color='black', linewidth=1)
    ax.text((bar_positions[0] + bar_positions[1]) / 2, y + p_value_text_offset, p_values[i][2], ha='center', fontsize=8)

    # Second line: 1st to 3rd bar
    y += general_offset
    ax.plot([bar_positions[0], bar_positions[0], bar_positions[2], bar_positions[2]], 
            [y, y + line_height_increment, y + line_height_increment, y], color='black', linewidth=1)
    ax.text((bar_positions[0] + bar_positions[2]) / 2, y + p_value_text_offset, p_values[i][1], ha='center', fontsize=8)

    # Topmost line: 1st to 4th bar
    y += general_offset
    ax.plot([bar_positions[0], bar_positions[0], bar_positions[3], bar_positions[3]], 
            [y, y + line_height_increment, y + line_height_increment, y], color='black', linewidth=1)
    ax.text((bar_positions[0] + bar_positions[3]) / 2, y + p_value_text_offset, p_values[i][0], ha='center', fontsize=8)

# Set axis labels and title
ax.set_ylabel('Average predicted score', fontsize=18)
ax.set_xticks(x + 1.5 * (bar_width + bar_spacing))
ax.set_xticklabels(networks, fontsize=16)
ax.set_ylim(0, 1.0)

# Add legend
ax.legend(title=None, loc='upper left', fontsize=12)

# Adjust layout
plt.tight_layout()

# Save the plot
file_path = os.path.join('results/', 'average_predicted_scores.png')
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.show()
