import matplotlib.pyplot as plt

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# Define layers
layers = [
    ('Input Layer', 784),
    ('Hidden Layer 1', 512),
    ('Hidden Layer 2', 256),
    ('Hidden Layer 3', 128),
    ('Output Layer', 10)
]

# Positions for the boxes
y_positions = list(range(len(layers)))[::-1]

# Plot each layer as a box
for i, (layer_name, neurons) in enumerate(layers):
    ax.text(0.5, y_positions[i], f'{layer_name}\n{neurons} neurons',
            bbox=dict(boxstyle='round,pad=0.5', edgecolor='black', facecolor='lightblue'),
            ha='center', va='center', fontsize=12)

# Add Dropout annotations between layers
for i in range(1, len(layers)):
    ax.annotate('Dropout(0.2)', xy=(0.5, y_positions[i-1]-0.4), xytext=(0.5, y_positions[i]-0.6),
                arrowprops=dict(arrowstyle='->'), ha='center', fontsize=10, color='darkred')

# Hide axes
ax.axis('off')

# Set limits
ax.set_xlim(0, 1)
ax.set_ylim(-1, len(layers))

# Title
plt.title('Model Architecture', fontsize=14)
plt.show()
