import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# List of elements and their states
elements = ['Al', 'N', 'W']
states = ['0+', '1+', '2+']

fig, axes = plt.subplots(3, 3, figsize=(16, 8))  # Create a 3x3 grid of subplots

for i, element in enumerate(elements):
    for j, state in enumerate(states):
        # Construct the file path for each matrix data file
        file_path = f"solps/mat{element}{state}.dat"
        
        # Read the data from the file
        data_df = pd.read_csv(file_path, header=None)
        data_array = data_df.values

        # Plot the data in the corresponding subplot
        ax = axes[i][j]
        # Here's where the colorbar range is set using vmin and vmax
        cax = ax.imshow(data_array, cmap='viridis', aspect='auto', vmin=0, vmax=10000)
        fig.colorbar(cax, ax=ax, label="Magnitude")
        ax.set_title(f"Transfer Matrix: {element}{state}")
        ax.set_xlabel("Surface Number")
        ax.set_ylabel("Surface Number")

# Adjust layout to prevent overlap
plt.tight_layout()
plt.savefig("transferMatrices.png")
plt.show()

