import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R

def load_trajectory(filename):
    """Load trajectory data (timestamp x y z qx qy qz qw)"""
    trajectory = []
    timestamps = []
    
    try:
        with open(filename, 'r') as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    parts = line.strip().split()
                    if len(parts) >= 8:
                        timestamp = float(parts[0])
                        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                        qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
                        
                        timestamps.append(timestamp)
                        trajectory.append([x, y, z])
    except FileNotFoundError:
        print(f"Warning: File not found {filename}")
        return np.array([]), np.array([])
    
    return np.array(trajectory), np.array(timestamps)

def load_optimization_points(filename):
    """Load point correspondences before and after optimization"""
    points_before = []
    points_after = []
    point_ids = []
    movements = []
    
    try:
        with open(filename, 'r') as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    parts = line.strip().split()
                    if len(parts) >= 8:  # point_id x_orig y_orig z_orig x_opt y_opt z_opt movement_mm
                        point_id = int(parts[0])
                        x_orig, y_orig, z_orig = float(parts[1]), float(parts[2]), float(parts[3])
                        x_opt, y_opt, z_opt = float(parts[4]), float(parts[5]), float(parts[6])
                        movement_mm = float(parts[7])
                        
                        point_ids.append(point_id)
                        points_before.append([x_orig, y_orig, z_orig])
                        points_after.append([x_opt, y_opt, z_opt])
                        movements.append(movement_mm / 1000.0)  # Convert to meters
    except FileNotFoundError:
        print(f"Warning: File not found {filename}")
        return np.array([]), np.array([]), np.array([]), np.array([])
    
    return np.array(points_before), np.array(points_after), np.array(point_ids), np.array(movements)

def create_trajectory_line(trajectory, color):
    """Create trajectory line"""
    if len(trajectory) < 2:
        return o3d.geometry.LineSet()
    
    lines = []
    for i in range(len(trajectory) - 1):
        lines.append([i, i + 1])
    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(trajectory)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.paint_uniform_color(color)
    
    return line_set

def create_trajectory_spheres(trajectory, color, radius=0.02):
    """Create small spheres at each trajectory position"""
    spheres = []
    # Downsample, show one sphere every few points
    step = max(1, len(trajectory) // 50)  # Show at most 50 spheres
    
    for i in range(0, len(trajectory), step):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        sphere.translate(trajectory[i])
        sphere.paint_uniform_color(color)
        spheres.append(sphere)
    
    return spheres

def create_movement_lines(points_before, points_after, movements):
    """Create line segments representing point movements, colored by movement distance"""
    if len(points_before) == 0:
        return o3d.geometry.LineSet()
    
    lines = []
    points = []
    colors = []
    
    # Calculate color mapping
    max_movement = movements.max() if len(movements) > 0 else 1.0
    
    for i in range(len(points_before)):
        # Add two endpoints of the line segment
        points.append(points_before[i])
        points.append(points_after[i])
        
        # Add line segment index
        lines.append([i*2, i*2+1])
        
        # Set color based on movement distance (green=small movement, red=large movement)
        normalized = min(movements[i] / max_movement, 1.0)
        color = [normalized, 1-normalized, 0]  # From green to red
        colors.append(color)
    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    
    return line_set

def main():
    # File paths
    base_path = "/home/jixian/Desktop/orbslam3_docker/Datasets/Voxmap/"
    
    # 1. Load points before and after optimization
    print("Loading point correspondences before and after optimization...")
    optimization_file = base_path + "output/optimized_points_final.txt"
    
    points_before, points_after, point_ids, movements = load_optimization_points(optimization_file)
    
    if len(points_before) > 0:
        print(f"  Successfully loaded {len(points_before)} point pairs")
    else:
        print("  Warning: Failed to load optimization point data")
        points_before = np.array([])
        points_after = np.array([])
        movements = np.array([])
    
    # 2. Load trajectory data (before and after loop closure)
    print("\nLoading trajectory data...")
    
    # Correct trajectory file paths
    traj_before_file = base_path + "standard_trajectory_no_loop.txt"
    traj_after_file = base_path + "standard_trajectory_with_loop.txt"
    
    traj_before, timestamps_before = load_trajectory(traj_before_file)
    traj_after, timestamps_after = load_trajectory(traj_after_file)
    
    if len(traj_before) > 0:
        print(f"  Trajectory before loop closure (no_loop): {len(traj_before)} poses")
    else:
        print(f"  Warning: File not found {traj_before_file}")
        
    if len(traj_after) > 0:
        print(f"  Trajectory after loop closure (with_loop): {len(traj_after)} poses")
    else:
        print(f"  Warning: File not found {traj_after_file}")
    
    # 3. Create visualization elements
    geometries = []
    
    # 3.1 Create point clouds
    if len(points_before) > 0:
        # Points before optimization (blue)
        pcd_before = o3d.geometry.PointCloud()
        pcd_before.points = o3d.utility.Vector3dVector(points_before)
        pcd_before.paint_uniform_color([0.2, 0.2, 0.8])  # Blue
        geometries.append(pcd_before)
        
        # Points after optimization (green)
        pcd_after = o3d.geometry.PointCloud()
        pcd_after.points = o3d.utility.Vector3dVector(points_after)
        pcd_after.paint_uniform_color([0.2, 0.8, 0.2])  # Green
        geometries.append(pcd_after)
        
        # Create movement line segments
        movement_lines = create_movement_lines(points_before, points_after, movements)
        geometries.append(movement_lines)
        
        # Mark points with excessive movement (red spheres)
        large_movement_threshold = 0.5  # 0.5 meters
        large_movements = movements > large_movement_threshold
        if np.any(large_movements):
            print(f"\nWarning: {np.sum(large_movements)} points moved more than {large_movement_threshold} meters")
            outlier_indices = np.where(large_movements)[0]
            for idx in outlier_indices[:20]:  # Show at most 20 outlier points
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
                sphere.translate(points_after[idx])
                sphere.paint_uniform_color([1.0, 0.0, 0.0])  # Red
                geometries.append(sphere)
    
    # 3.2 Create trajectory visualization
    if len(traj_before) > 0:
        # Trajectory before loop closure (cyan)
        traj_line_before = create_trajectory_line(traj_before, [0, 0.8, 0.8])
        geometries.append(traj_line_before)
        
        # Add keyframe markers on trajectory
        spheres_before = create_trajectory_spheres(traj_before, [0, 0.6, 0.6], radius=0.03)
        geometries.extend(spheres_before)
        
        # Mark start point (large green sphere)
        start_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
        start_sphere.translate(traj_before[0])
        start_sphere.paint_uniform_color([0.0, 1.0, 0.0])
        geometries.append(start_sphere)
    
    if len(traj_after) > 0:
        # Trajectory after loop closure (orange)
        traj_line_after = create_trajectory_line(traj_after, [1.0, 0.5, 0])
        geometries.append(traj_line_after)
        
        # Add keyframe markers on trajectory
        spheres_after = create_trajectory_spheres(traj_after, [0.8, 0.4, 0], radius=0.03)
        geometries.extend(spheres_after)
        
        # Mark end point (large red sphere)
        end_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
        end_sphere.translate(traj_after[-1])
        end_sphere.paint_uniform_color([1.0, 0.0, 0.0])
        geometries.append(end_sphere)
    
    # 3.3 If both trajectories have the same number of points, show correspondences
    if len(traj_before) == len(traj_after) and len(traj_before) > 0:
        # Create sparse correspondence lines (connect every few points)
        step = max(1, len(traj_before) // 20)
        correspondence_lines = []
        correspondence_points = []
        
        for i in range(0, len(traj_before), step):
            correspondence_points.append(traj_before[i])
            correspondence_points.append(traj_after[i])
            correspondence_lines.append([len(correspondence_points)-2, len(correspondence_points)-1])
        
        if len(correspondence_lines) > 0:
            corr_line_set = o3d.geometry.LineSet()
            corr_line_set.points = o3d.utility.Vector3dVector(correspondence_points)
            corr_line_set.lines = o3d.utility.Vector2iVector(correspondence_lines)
            corr_line_set.paint_uniform_color([0.5, 0.5, 0.5])  # Gray
            geometries.append(corr_line_set)
    
    # 4. Add coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    geometries.append(coord_frame)
    
    # 5. Statistics
    if len(movements) > 0:
        print("\n=== Point Optimization Statistics ===")
        print(f"Number of optimized points: {len(points_before)}")
        print(f"Average movement: {movements.mean():.3f} m")
        print(f"Median movement: {np.median(movements):.3f} m")
        print(f"Maximum movement: {movements.max():.3f} m")
        print(f"Minimum movement: {movements.min():.3f} m")
    
    if len(traj_before) > 0 and len(traj_after) > 0 and len(traj_before) == len(traj_after):
        traj_movements = np.linalg.norm(traj_after - traj_before, axis=1)
        print("\n=== Trajectory Optimization Statistics ===")
        print(f"Average trajectory offset: {traj_movements.mean():.3f} m")
        print(f"Maximum trajectory offset: {traj_movements.max():.3f} m")
    
    # 6. Visualization instructions
    print("\n=== Visualization Legend ===")
    print("[3D Points]")
    print("  Blue points: Position before optimization")
    print("  Green points: Position after optimization")
    print("  Lines: Point movements (green→yellow→red indicates increasing movement distance)")
    print("  Red spheres: Points with abnormally large movements")
    print("\n[Trajectories]")
    print("  Cyan line+spheres: Trajectory before loop closure")
    print("  Orange line+spheres: Trajectory after loop closure")
    print("  Large green sphere: Trajectory start point")
    print("  Large red sphere: Trajectory end point")
    print("  Gray lines: Correspondence lines between trajectories")
    print("\nControls: Drag mouse to rotate, scroll wheel to zoom, Ctrl+drag to pan")
    
    # 7. Create visualization window
    if len(geometries) > 1:  # Only show if there's more than just the coordinate frame
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Loop Closure Optimization Visualization", width=1600, height=900)
        
        for geometry in geometries:
            vis.add_geometry(geometry)
        
        # Set rendering options
        render_option = vis.get_render_option()
        render_option.point_size = 8.0
        render_option.line_width = 3.0
        render_option.background_color = np.array([0.95, 0.95, 0.95])
        render_option.show_coordinate_frame = True
        
        # Set viewpoint
        ctr = vis.get_view_control()
        ctr.set_zoom(0.5)
        
        vis.run()
        vis.destroy_window()
    else:
        print("\nInsufficient data for visualization, please check file paths and formats")

if __name__ == "__main__":
    main()
