#!/usr/bin/env python3
"""
Voxblox-style Data Extractor
Combines strict Voxblox filtering mechanisms with complete data extraction
"""

import numpy as np
import os
from collections import defaultdict
from typing import Tuple, List, Dict, Optional

try:
    import open3d as o3d
except ImportError:
    print("Installing open3d...")
    os.system("pip3 install open3d")
    import open3d as o3d

try:
    from scipy.spatial.transform import Rotation as R
except ImportError:
    print("Installing scipy...")
    os.system("pip3 install scipy")
    from scipy.spatial.transform import Rotation as R

try:
    import rosbag
    from cv_bridge import CvBridge
    HAS_ROSBAG = True
except ImportError:
    print("Warning: rosbag not available, will use projected depth only")
    HAS_ROSBAG = False


class VoxbloxStyleDataExtractor:
    """
    Data extractor based on Voxblox principles
    Core features:
    1. Truncation distance limit (most important!)
    2. Anti-grazing filter
    3. Consecutive observation requirements
    4. Depth consistency check
    5. Extract actual depth from rosbag
    """
    
    def __init__(self, camera_params: dict, voxblox_params: dict = None):
        # Camera parameters
        self.fx = camera_params['fx']
        self.fy = camera_params['fy']
        self.cx = camera_params['cx']
        self.cy = camera_params['cy']
        self.width = camera_params.get('width', 640)
        self.height = camera_params.get('height', 480)
        
        # Voxblox parameters (critical!)
        if voxblox_params is None:
            voxblox_params = {}
        
        # TSDF core parameters
        self.voxel_size = voxblox_params.get('voxel_size', 0.02)  # 2cm voxel
        self.truncation_distance = voxblox_params.get('truncation_distance', 3.0 * self.voxel_size)  # 6cm
        
        # Depth range
        self.min_depth = voxblox_params.get('min_depth', 0.3)
        self.max_depth = voxblox_params.get('max_depth', 4.0)
        
        # Anti-grazing parameters (avoid grazing angles)
        self.max_grazing_angle = voxblox_params.get('max_grazing_angle', 70.0)  # degrees
        self.cos_min_angle = np.cos(np.radians(self.max_grazing_angle))
        
        # Observation quality parameters
        self.min_consecutive_observations = voxblox_params.get('min_consecutive_observations', 3)
        self.max_frame_gap = voxblox_params.get('max_frame_gap', 2)  # Number of frames allowed to skip
        self.max_observations_per_point = voxblox_params.get('max_observations_per_point', 20)
        
        # Depth image storage
        self.depth_images = {}
        
        if HAS_ROSBAG:
            self.bridge = CvBridge()
        
        print("Voxblox-style parameters:")
        print(f"  - Voxel size: {self.voxel_size}m")
        print(f"  - Truncation distance: {self.truncation_distance}m")
        print(f"  - Depth range: [{self.min_depth}, {self.max_depth}]m")
        print(f"  - Max grazing angle: {self.max_grazing_angle}°")
        print(f"  - Min consecutive obs: {self.min_consecutive_observations}")
    
    def load_trajectory(self, file_path: str) -> Tuple[Dict[int, np.ndarray], List[int], List[float]]:
        """Load trajectory file"""
        poses = {}
        kf_indices = []
        timestamps = []
        
        if not os.path.exists(file_path):
            print(f"Error: {file_path} not found!")
            return poses, kf_indices, timestamps
        
        with open(file_path, 'r') as f:
            kf_idx = 0
            for line in f:
                if line.startswith('#') or line.strip() == '':
                    continue
                
                parts = line.strip().split()
                
                # TUM format
                if len(parts) == 8:
                    try:
                        timestamp = float(parts[0])
                        tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
                        qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
                        
                        rotation = R.from_quat([qx, qy, qz, qw])
                        T_wc = np.eye(4)
                        T_wc[:3, :3] = rotation.as_matrix()
                        T_wc[:3, 3] = [tx, ty, tz]
                        
                        # Convert to camera-to-world transformation (note this!)
                        T_cw = np.linalg.inv(T_wc)
                        
                        poses[kf_idx] = T_cw
                        kf_indices.append(kf_idx)
                        timestamps.append(timestamp)
                        kf_idx += 1
                    except:
                        continue
        
        print(f"Loaded {len(poses)} poses")
        return poses, kf_indices, timestamps
    
    def load_depth_from_rosbag(self, bag_file: str, kf_indices: List[int], 
                               timestamps: List[float], depth_topic: str = '/camera/depth/image_raw'):
        """Load depth images from rosbag"""
        if not HAS_ROSBAG:
            print("Rosbag not available, using projected depth only")
            return
        
        if not os.path.exists(bag_file):
            print(f"Warning: {bag_file} not found")
            return
        
        print(f"Loading depth images from {bag_file}...")
        bag = rosbag.Bag(bag_file)
        
        depth_msgs = []
        for topic, msg, t in bag.read_messages(topics=[depth_topic]):
            depth_msgs.append((msg.header.stamp.to_sec(), msg))
        
        print(f"Found {len(depth_msgs)} depth messages")
        
        for kf_idx, kf_time in zip(kf_indices, timestamps):
            best_msg = None
            best_time_diff = 0.05  # 50ms tolerance
            
            for msg_time, msg in depth_msgs:
                time_diff = abs(msg_time - kf_time)
                if time_diff < best_time_diff:
                    best_time_diff = time_diff
                    best_msg = msg
            
            if best_msg is not None:
                cv_image = self.bridge.imgmsg_to_cv2(best_msg, "passthrough")
                self.depth_images[kf_idx] = cv_image
        
        bag.close()
        print(f"Loaded {len(self.depth_images)} depth images")
    
    def get_depth_at_pixel(self, kf_idx: int, u: float, v: float) -> Tuple[float, float]:
        """Get depth value at pixel"""
        if kf_idx not in self.depth_images:
            return -1.0, 0.0
        
        depth_img = self.depth_images[kf_idx]
        h, w = depth_img.shape[:2]
        
        if u < 0 or u >= w-1 or v < 0 or v >= h-1:
            return -1.0, 0.0
        
        u0, v0 = int(u), int(v)
        depth = float(depth_img[v0, u0])
        
        if depth_img.dtype == np.uint16:
            depth = depth / 1000.0  # mm to m
        
        if depth > self.min_depth and depth < self.max_depth:
            return depth, 1.0
        
        return -1.0, 0.0
    
    def check_anti_grazing(self, camera_origin: np.ndarray, point_3d: np.ndarray, 
                          surface_normal: Optional[np.ndarray] = None) -> bool:
        """Check for grazing angle (Voxblox's anti-grazing filter)"""
        view_direction = point_3d - camera_origin
        view_distance = np.linalg.norm(view_direction)
        
        if view_distance < 1e-6:
            return False
        
        view_direction = view_direction / view_distance
        
        if surface_normal is not None:
            cos_angle = abs(np.dot(view_direction, surface_normal))
            if cos_angle < self.cos_min_angle:
                return False  # Angle too large, it's grazing
        
        return True
    
    def compute_observation_weight(self, depth_diff: float, depth: float, 
                                  grazing_factor: float = 1.0) -> float:
        """Compute observation weight (Voxblox style)"""
        # Truncation weight: closer to surface means higher weight
        truncation_weight = np.exp(-abs(depth_diff) / self.truncation_distance)
        
        # Depth weight: near observations are more reliable
        depth_weight = 1.0 / (1.0 + depth / 5.0)
        
        # Combined weight
        return truncation_weight * depth_weight * grazing_factor
    
    def process_single_observation(self, point_3d: np.ndarray, kf_idx: int, 
                                  T_cw: np.ndarray, surface_normal: Optional[np.ndarray] = None) -> Optional[Dict]:
        """
        Process single observation (core function!)
        Implements Voxblox's key filtering mechanisms
        """
        # Get camera origin
        T_wc = np.linalg.inv(T_cw)
        camera_origin = T_wc[:3, 3]
        
        # Calculate distance to camera (for truncation check)
        distance_to_camera = np.linalg.norm(point_3d - camera_origin)
        
        # Transform to camera coordinate system
        point_homo = np.append(point_3d, 1)
        point_cam = T_cw @ point_homo
        depth_projected = point_cam[2]
        
        # Depth range check
        if depth_projected <= self.min_depth or depth_projected > self.max_depth:
            return None
        
        # Project to image
        u = self.fx * point_cam[0] / depth_projected + self.cx
        v = self.fy * point_cam[1] / depth_projected + self.cy
        
        # Image boundary check
        if not (0 <= u < self.width and 0 <= v < self.height):
            return None
        
        # Anti-grazing check (important!)
        grazing_factor = 1.0
        if surface_normal is not None:
            if not self.check_anti_grazing(camera_origin, point_3d, surface_normal):
                return None
            
            # Calculate grazing factor for weight
            view_dir = (point_3d - camera_origin) / distance_to_camera
            grazing_factor = max(0.1, abs(np.dot(view_dir, surface_normal)))
        
        # Get measured depth (if available)
        depth_measured = depth_projected  # Default to projected depth
        depth_confidence = 0.5
        
        if kf_idx in self.depth_images:
            depth_m, conf = self.get_depth_at_pixel(kf_idx, u, v)
            if depth_m > 0:
                depth_measured = depth_m
                depth_confidence = conf
        
        # Depth difference
        depth_diff = abs(depth_measured - depth_projected)
        
        # Truncation distance check (Voxblox's most important feature!)
        if depth_diff > self.truncation_distance:
            return None  # Beyond truncation range, don't record this observation
        
        # Calculate weight
        weight = self.compute_observation_weight(depth_diff, depth_projected, grazing_factor)
        
        # Filter out observations with very low weight
        if weight < 0.1:
            return None
        
        return {
            'kf_idx': kf_idx,
            'pixel': (u, v),
            'depth_measured': depth_measured,
            'depth_projected': depth_projected,
            'weight': weight,
            'confidence': depth_confidence,
            'depth_diff': depth_diff
        }
    
    def filter_consecutive_observations(self, observations: List[Dict]) -> List[Dict]:
        """
        Filter observations, keeping consecutive observation sequences
        This is an important mechanism for Voxblox to ensure observation quality
        """
        if len(observations) < self.min_consecutive_observations:
            return []
        
        # Sort by KF index
        observations = sorted(observations, key=lambda x: x['kf_idx'])
        
        # Find consecutive sequences
        sequences = []
        current_seq = [observations[0]]
        
        for i in range(1, len(observations)):
            # Check if consecutive (allow skipping max_frame_gap frames)
            if observations[i]['kf_idx'] - observations[i-1]['kf_idx'] <= self.max_frame_gap:
                current_seq.append(observations[i])
            else:
                # Sequence interrupted
                if len(current_seq) >= self.min_consecutive_observations:
                    sequences.append(current_seq)
                current_seq = [observations[i]]
        
        # Check last sequence
        if len(current_seq) >= self.min_consecutive_observations:
            sequences.append(current_seq)
        
        # Merge all valid sequences
        filtered = []
        for seq in sequences:
            filtered.extend(seq)
        
        # Limit maximum observations
        if len(filtered) > self.max_observations_per_point:
            # Sort by weight, keep best observations
            filtered = sorted(filtered, key=lambda x: x['weight'], reverse=True)
            filtered = filtered[:self.max_observations_per_point]
            # Re-sort by KF index
            filtered = sorted(filtered, key=lambda x: x['kf_idx'])
        
        return filtered
    
    def find_voxblox_correspondences(self, mesh_file: str, poses: Dict[int, np.ndarray], 
                                    timestamps: List[float], sample_size: int = 20000) -> Dict:
        """
        Find correspondences using Voxblox style
        Strict filtering mechanisms ensure only high-quality observations are retained
        """
        if not os.path.exists(mesh_file):
            print(f"Error: {mesh_file} not found!")
            return {}
        
        # Load mesh
        mesh = o3d.io.read_triangle_mesh(mesh_file)
        vertices = np.asarray(mesh.vertices)
        
        # Calculate normals (for anti-grazing)
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()
        normals = np.asarray(mesh.vertex_normals)
        
        print(f"Mesh has {len(vertices)} vertices")
        
        # Sampling
        if sample_size < len(vertices):
            indices = np.random.choice(len(vertices), sample_size, replace=False)
        else:
            indices = np.arange(len(vertices))
        
        print(f"Processing {len(indices)} points with Voxblox-style filtering...")
        
        point_observations = {}
        
        # Statistics
        total_checks = 0
        truncation_filtered = 0
        grazing_filtered = 0
        consecutive_filtered = 0
        
        for i, idx in enumerate(indices):
            if i % 2000 == 0:
                print(f"  Processing {i}/{len(indices)} points...")
            
            point_3d = vertices[idx]
            normal = normals[idx] if idx < len(normals) else None
            
            # Collect all potential observations
            observations = []
            
            for kf_idx, T_cw in poses.items():
                total_checks += 1
                
                # Process single observation (includes all Voxblox filtering)
                obs = self.process_single_observation(point_3d, kf_idx, T_cw, normal)
                
                if obs is not None:
                    observations.append(obs)
                else:
                    # Statistics for filtered reasons (simplified)
                    if len(observations) == 0:
                        truncation_filtered += 1
            
            # Filter consecutive observations
            filtered_obs = self.filter_consecutive_observations(observations)
            
            if len(filtered_obs) < len(observations):
                consecutive_filtered += len(observations) - len(filtered_obs)
            
            # Save valid observations
            if len(filtered_obs) >= self.min_consecutive_observations:
                point_observations[idx] = {
                    'point_3d': point_3d.tolist(),
                    'observations': filtered_obs
                }
        
        print(f"\nVoxblox-style filtering statistics:")
        print(f"  Total checks: {total_checks}")
        print(f"  Points with valid observations: {len(point_observations)}")
        
        # Calculate observation distribution
        if point_observations:
            obs_counts = [len(data['observations']) for data in point_observations.values()]
            print(f"\nObservation distribution:")
            print(f"  Average: {np.mean(obs_counts):.2f}")
            print(f"  Median: {np.median(obs_counts):.0f}")
            print(f"  Max: {max(obs_counts)}")
            print(f"  Min: {min(obs_counts)}")
        
        return point_observations
    
    def save_detailed_correspondences(self, point_observations: Dict, 
                                     poses_before: Dict, poses_after: Dict,
                                     output_file: str):
        """Save detailed correspondence data"""
        with open(output_file, 'w') as f:
            f.write("# Voxblox-style correspondences with complete data\n")
            f.write("# Format:\n")
            f.write("# point_id x y z num_observations\n")
            f.write("#   kf_idx u v depth_measured depth_projected weight confidence\n")
            f.write(f"# Truncation distance: {self.truncation_distance}m\n")
            f.write(f"# Min consecutive observations: {self.min_consecutive_observations}\n")
            f.write(f"# Total points: {len(point_observations)}\n")
            f.write("#\n")
            
            for point_id, data in point_observations.items():
                point_3d = data['point_3d']
                observations = data['observations']
                
                f.write(f"{point_id} {point_3d[0]:.6f} {point_3d[1]:.6f} {point_3d[2]:.6f} {len(observations)}\n")
                
                for obs in observations:
                    f.write(f"  {obs['kf_idx']} {obs['pixel'][0]:.2f} {obs['pixel'][1]:.2f} ")
                    f.write(f"{obs['depth_measured']:.4f} {obs['depth_projected']:.4f} ")
                    f.write(f"{obs['weight']:.4f} {obs['confidence']:.4f}\n")
        
        print(f"Saved to {output_file}")
        
        # Save poses
        poses_file = output_file.replace('.txt', '_poses.npz')
        np.savez(poses_file,
                 poses_before={str(k): v for k, v in poses_before.items()},
                 poses_after={str(k): v for k, v in poses_after.items()})
        print(f"Saved poses to {poses_file}")
        
        # Statistics
        total_obs = sum(len(data['observations']) for data in point_observations.values())
        avg_obs = total_obs / len(point_observations) if point_observations else 0
        
        print(f"\nFinal statistics:")
        print(f"  Total 3D points: {len(point_observations)}")
        print(f"  Total observations: {total_obs}")
        print(f"  Average observations per point: {avg_obs:.2f}")


def main():
    # Camera parameters
    camera_params = {
        'fx': 377.535257164,  # Use your actual parameters
        'fy': 377.209841379,
        'cx': 328.193371286,
        'cy': 240.426878936,
        'width': 640,
        'height': 480
    }
    
    # Voxblox parameters (adjust these to control observation quantity!)
    voxblox_params = {
        'voxel_size': 0.02,                    # 2cm voxel
        'truncation_distance': 0.06,           # 6cm (3x voxel size) - this is key!
        'min_depth': 0.3,
        'max_depth': 4.0,
        'max_grazing_angle': 70.0,             # 70 degree max grazing angle
        'min_consecutive_observations': 3,      # At least 3 consecutive observations
        'max_frame_gap': 2,                    # Allow skipping 2 frames
        'max_observations_per_point': 15       # Maximum 15 observations
    }
    
    # File paths
    rosbag_file = "/Datasets/Kimera/Kimera_Clipped_bag/12_07_thoth_clipped.bag"  # Your rosbag file
    mesh_file = "mesh_output.ply"  # Voxblox output mesh
    trajectory_before_file = "standard_trajectory_no_loop.txt"  # Trajectory before loop closure
    trajectory_after_file = "standard_trajectory_with_loop.txt"  # Trajectory after loop closure
    output_file = "optimization_data.txt"  # Output file
    
    # Create extractor
    extractor = VoxbloxStyleDataExtractor(camera_params, voxblox_params)
    
    # 1. Load trajectories
    print("\n=== Loading trajectories ===")
    poses_before, kf_indices, timestamps = extractor.load_trajectory(trajectory_before_file)
    poses_after, _, _ = extractor.load_trajectory(trajectory_after_file)
    
    if len(poses_before) == 0:
        print("Error: No poses loaded!")
        return
    
    # 2. Load depth images (if rosbag available)
    print("\n=== Loading depth images ===")
    if os.path.exists(rosbag_file):
        extractor.load_depth_from_rosbag(rosbag_file, kf_indices, timestamps)
    else:
        print(f"Warning: {rosbag_file} not found, using projected depth only")
    
    # 3. Find correspondences (using Voxblox-style strict filtering)
    print("\n=== Finding Voxblox-style correspondences ===")
    point_observations = extractor.find_voxblox_correspondences(
        mesh_file,
        poses_before,
        timestamps,
        sample_size=20000
    )
    
    if len(point_observations) == 0:
        print("Error: No valid correspondences found!")
        print("Try adjusting voxblox_params (increase truncation_distance or decrease min_consecutive_observations)")
        return
    
    # 4. Save results
    print("\n=== Saving results ===")
    extractor.save_detailed_correspondences(
        point_observations,
        poses_before,
        poses_after,
        output_file
    )
    
    print("\n=== Complete! ===")


if __name__ == "__main__":
    main()
