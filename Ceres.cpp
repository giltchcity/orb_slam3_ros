import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
from mpl_toolkits.mplot3d import Axes3D

def read_tum_trajectory(file_path):
    """Read trajectory file in TUM format"""
    try:
        print(f"Attempting to read file: {file_path}")
        data = np.loadtxt(file_path)
        print(f"Successfully loaded file: {file_path}, shape: {data.shape}")
        # Extract position information (x, y, z), corresponding to columns 1, 2, 3
        positions = data[:, 1:4]
        return positions
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def plot_trajectories():
    # File paths
    files = [
        "ceres_optimized_trajectory.txt",
        "standard_trajectory_no_loop.txt",
        "standard_trajectory_sim3_transformed.txt",
        "standard_trajectory_with_loop.txt"
    ]
    
    # Legend labels
    labels = [
        "optimization2",
        "No Loop",
        "Sim3 Transformed",
        "With Loop"
    ]
    
    # Colors for trajectories
    colors = ['blue', 'green', 'red', 'purple']
    
    print("Creating figure...")
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 12))
    
    # 3D plot
    ax1 = fig.add_subplot(221, projection='3d')
    # 2D plots
    ax2 = fig.add_subplot(222)  # X-Y plane
    ax3 = fig.add_subplot(223)  # X-Z plane
    ax4 = fig.add_subplot(224)  # Y-Z plane
    
    # Store all plot objects
    plot_lines = []
    scatter_points = []
    
    # Read and plot each trajectory
    for file_path, label, color in zip(files, labels, colors):
        positions = read_tum_trajectory(file_path)
        if positions is None:
            print(f"Skipping {file_path} due to read error")
            continue
            
        print(f"Plotting {label} with {len(positions)} points")
        
        # 3D plot
        line1 = ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                label=label, color=color)[0]
        scatter1 = ax1.scatter(positions[0, 0], positions[0, 1], positions[0, 2], 
                    color=color, marker='o', s=50)  # Mark starting point
        
        # 2D plots
        line2 = ax2.plot(positions[:, 0], positions[:, 1], label=label, color=color)[0]
        scatter2 = ax2.scatter(positions[0, 0], positions[0, 1], color=color, marker='o', s=50)
        
        line3 = ax3.plot(positions[:, 0], positions[:, 2], color=color)[0]
        scatter3 = ax3.scatter(positions[0, 0], positions[0, 2], color=color, marker='o', s=50)
        
        line4 = ax4.plot(positions[:, 1], positions[:, 2], color=color)[0]
        scatter4 = ax4.scatter(positions[0, 1], positions[0, 2], color=color, marker='o', s=50)
        
        # Save plot objects for visibility control
        plot_lines.append([line1, line2, line3, line4])
        scatter_points.append([scatter1, scatter2, scatter3, scatter4])
    
    # Set labels and titles
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Trajectory Comparison')
    
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('X-Y Plane')
    ax2.grid(True)
    ax2.legend()
    
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Z (m)')
    ax3.set_title('X-Z Plane')
    ax3.grid(True)
    
    ax4.set_xlabel('Y (m)')
    ax4.set_ylabel('Z (m)')
    ax4.set_title('Y-Z Plane')
    ax4.grid(True)
    
    # Add checkboxes for visibility control
    print("Setting up checkboxes...")
    # Create a small axis for checkboxes
    plt.subplots_adjust(bottom=0.2)  # Make space for checkboxes
    checkbox_ax = plt.axes([0.01, 0.01, 0.15, 0.15])
    # Initial state: all checked
    visibility = [True, True, True, True]
    # Create checkboxes
    checkbox = CheckButtons(checkbox_ax, labels, visibility)
    
    # Checkbox callback function
    def update_visibility(label):
        index = labels.index(label)
        # Toggle visibility of corresponding trajectory
        for line in plot_lines[index]:
            line.set_visible(not line.get_visible())
        for scatter in scatter_points[index]:
            scatter.set_visible(not scatter.get_visible())
        # Redraw figure
        fig.canvas.draw_idle()
    
    # Register callback function
    checkbox.on_clicked(update_visibility)
    
    # Adjust layout
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    
    print("Ready to display plot...")
    # Show figure - BLOCKING MODE
    plt.show(block=True)
    print("Plot window closed.")

if __name__ == "__main__":
    # Remove ion() - we don't want interactive mode here
    print("Starting trajectory visualization...")
    plot_trajectories()
