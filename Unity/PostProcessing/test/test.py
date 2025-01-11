import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from matplotlib.widgets import CheckButtons

file_path = 'C:/Users/Thoma/Code/TheVRGameDemoLocal/TheVRGameDemoLocal/PostProcessing/test/OL_2023_08_16_17_27_01.txt'

# Read the tab-separated file into lines
with open(file_path, 'r') as file:
    lines = file.readlines()

# Store the last line in a list variable
last_line = lines[-1].strip()

# Remove the last line from the lines list
lines = lines[:-1]

# Convert the lines list to a Pandas DataFrame
data = [line.strip().split('\t') for line in lines]
columns = data[0]  # Assuming the first line contains column names
data = data[1:]  # Removing the header line
df = pd.DataFrame(data, columns=columns)

# Extract the timestamp from the first column and convert it to datetime
df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%Y_%m_%d_%H_%M_%S')

# Create a 3D plot
fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
ax.view_init(elev=107, azim=-90)

ax.set_title('Click on legend point to toggle it on/off')

# Set plot title and axis labels
ax.set_title('Objects position through time', fontsize=16)
ax.set_xlabel('X', fontsize=14)
ax.set_ylabel('Y', fontsize=14)
ax.set_zlabel('Z', fontsize=14)

# Define a list of colors and markers
colors = ['blue', 'red', 'green', 'black', 'orange', 'purple']
markers = ['o', 'o', 'o', 'o', 'o', 'o', '^', '^', '^', '^', 's', 's', 's', 's', 'X', 'X', 'X', 'X']
objects = list()

# Plot the positions for each object with a combination of unique color and marker
for i, col in enumerate(df.columns):
    excluded_columns = ['Timestamp', 'Gaze', 'Grab', 'CamPosition', 'CamRotation', 'Hand']

    if col not in excluded_columns:
        object_name = col
        x = df[col].str.split(',', expand=True).astype(float).iloc[:, 0]
        y = df[col].str.split(',', expand=True).astype(float).iloc[:, 1]
        z = df[col].str.split(',', expand=True).astype(float).iloc[:, 2]

        color = colors[i % len(colors)]

        marker = markers[i % len(markers)]

        # Check if the position of the object has changed between consecutive timestamps
        x_diff = x.diff()
        y_diff = y.diff()
        z_diff = z.diff()
        moved = (x_diff != 0) | (y_diff != 0) | (z_diff != 0)

        # Calculate the sizes array based on the number of points that moved
        num_moved = np.sum(moved)
        sizes = np.linspace(40, 3, num_moved)
        # Interpolate the sizes to match the length of x, y, and z arrays
        sizes = np.interp(np.arange(len(x)), np.where(moved)[0], sizes)
        object = ax.scatter(x[moved], y[moved], z[moved], label=object_name, c=color, marker=marker, s=sizes[moved],
                            alpha=0)
        objects.append(object)

# Add legend for the custom legend lines
legend_lines = []
for i, col in enumerate(df.columns):
    if col != 'Timestamp' or col != 'Gaze' or col != 'Grab':
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        legend_line = Line2D([0], [0], color=color, marker=marker, linestyle='None',
                             markersize=10)  # Increase markersize
        legend_lines.append(legend_line)

# Add legend with interactive toggling
leg = ax.legend(legend_lines, df.columns[1:], bbox_to_anchor=(1.04, 1), loc='upper left')

lined = {}  # Will map legend points to original scatter points.
for legpoint, origpoint in zip(leg.get_lines(), objects):
    legpoint.set_picker(True)  # Enable picking on the legend point.
    lined[legpoint] = origpoint
    legpoint.set_alpha(0.2)  # Initially set the legend point alpha to 0.7 (partially visible)

def on_pick(event):
    # On the pick event, find the original scatter point corresponding to the legend
    # proxy point, and toggle its visibility.
    legpoint = event.artist
    origpoint = lined[legpoint]

    # Check the alpha value of the original scatter point
    current_alpha = origpoint.get_alpha()

    # Toggle the alpha value between 0.7 and 0
    new_alpha = 0.0 if current_alpha == 0.7 else 0.7

    # Set the new alpha value for the original scatter point
    origpoint.set_alpha(new_alpha)

    # Change the alpha on the point in the legend, so we can see what points
    # have been toggled.
    legpoint.set_alpha(0.7 if new_alpha == 0.7 else 0.2)

    fig.canvas.draw()

fig.canvas.mpl_connect('pick_event', on_pick)

# Add checkboxes for each item in the legend
rax = plt.axes([0.8, 0.1, 0.1, 0.5])
labels = [str(line.get_label()) for line in leg.get_lines()]
visibility = [line.get_alpha() != 0 for line in leg.get_lines()]
check = CheckButtons(rax, labels, visibility)

def func(label):
    index = labels.index(label)
    line = leg.get_lines()[index]
    current_alpha = line.get_alpha()
    new_alpha = 1.0 if current_alpha == 0 else 0
    line.set_alpha(new_alpha)
    fig.canvas.draw_idle()

check.on_clicked(func)

plt.show()
