#!/usr/bin/env python3
"""
Voxblox风格的数据提取器
结合了严格的Voxblox过滤机制和完整的数据提取
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
    基于Voxblox原理的数据提取器
    核心特性：
    1. Truncation distance限制（最重要！）
    2. Anti-grazing过滤
    3. 连续观测要求
    4. 深度一致性检查
    5. 从rosbag提取实际深度
    """
    
    def __init__(self, camera_params: dict, voxblox_params: dict = None):
        # 相机参数
        self.fx = camera_params['fx']
        self.fy = camera_params['fy']
        self.cx = camera_params['cx']
        self.cy = camera_params['cy']
        self.width = camera_params.get('width', 640)
        self.height = camera_params.get('height', 480)
        
        # Voxblox参数（关键！）
        if voxblox_params is None:
            voxblox_params = {}
        
        # TSDF核心参数
        self.voxel_size = voxblox_params.get('voxel_size', 0.02)  # 2cm voxel
        self.truncation_distance = voxblox_params.get('truncation_distance', 3.0 * self.voxel_size)  # 6cm
        
        # 深度范围
        self.min_depth = voxblox_params.get('min_depth', 0.3)
        self.max_depth = voxblox_params.get('max_depth', 4.0)
        
        # Anti-grazing参数（避免掠射角）
        self.max_grazing_angle = voxblox_params.get('max_grazing_angle', 70.0)  # degrees
        self.cos_min_angle = np.cos(np.radians(self.max_grazing_angle))
        
        # 观测质量参数
        self.min_consecutive_observations = voxblox_params.get('min_consecutive_observations', 3)
        self.max_frame_gap = voxblox_params.get('max_frame_gap', 2)  # 允许跳过的帧数
        self.max_observations_per_point = voxblox_params.get('max_observations_per_point', 20)
        
        # 深度图像存储
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
        """加载轨迹文件"""
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
                
                # TUM格式
                if len(parts) == 8:
                    try:
                        timestamp = float(parts[0])
                        tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
                        qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
                        
                        rotation = R.from_quat([qx, qy, qz, qw])
                        T_wc = np.eye(4)
                        T_wc[:3, :3] = rotation.as_matrix()
                        T_wc[:3, 3] = [tx, ty, tz]
                        
                        # 转换为相机到世界的变换（注意这里！）
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
        """从rosbag加载深度图像"""
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
            best_time_diff = 0.05  # 50ms容差
            
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
        """获取像素处的深度值"""
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
        """检查是否为掠射角（Voxblox的anti-grazing filter）"""
        view_direction = point_3d - camera_origin
        view_distance = np.linalg.norm(view_direction)
        
        if view_distance < 1e-6:
            return False
        
        view_direction = view_direction / view_distance
        
        if surface_normal is not None:
            cos_angle = abs(np.dot(view_direction, surface_normal))
            if cos_angle < self.cos_min_angle:
                return False  # 角度太大，是掠射
        
        return True
    
    def compute_observation_weight(self, depth_diff: float, depth: float, 
                                  grazing_factor: float = 1.0) -> float:
        """计算观测权重（Voxblox风格）"""
        # Truncation权重：距离表面越近权重越高
        truncation_weight = np.exp(-abs(depth_diff) / self.truncation_distance)
        
        # 深度权重：近处观测更可靠
        depth_weight = 1.0 / (1.0 + depth / 5.0)
        
        # 综合权重
        return truncation_weight * depth_weight * grazing_factor
    
    def process_single_observation(self, point_3d: np.ndarray, kf_idx: int, 
                                  T_cw: np.ndarray, surface_normal: Optional[np.ndarray] = None) -> Optional[Dict]:
        """
        处理单个观测（核心函数！）
        实现Voxblox的关键过滤机制
        """
        # 获取相机原点
        T_wc = np.linalg.inv(T_cw)
        camera_origin = T_wc[:3, 3]
        
        # 计算到相机的距离（用于truncation检查）
        distance_to_camera = np.linalg.norm(point_3d - camera_origin)
        
        # 变换到相机坐标系
        point_homo = np.append(point_3d, 1)
        point_cam = T_cw @ point_homo
        depth_projected = point_cam[2]
        
        # 深度范围检查
        if depth_projected <= self.min_depth or depth_projected > self.max_depth:
            return None
        
        # 投影到图像
        u = self.fx * point_cam[0] / depth_projected + self.cx
        v = self.fy * point_cam[1] / depth_projected + self.cy
        
        # 图像边界检查
        if not (0 <= u < self.width and 0 <= v < self.height):
            return None
        
        # Anti-grazing检查（重要！）
        grazing_factor = 1.0
        if surface_normal is not None:
            if not self.check_anti_grazing(camera_origin, point_3d, surface_normal):
                return None
            
            # 计算grazing factor用于权重
            view_dir = (point_3d - camera_origin) / distance_to_camera
            grazing_factor = max(0.1, abs(np.dot(view_dir, surface_normal)))
        
        # 获取测量深度（如果有）
        depth_measured = depth_projected  # 默认使用投影深度
        depth_confidence = 0.5
        
        if kf_idx in self.depth_images:
            depth_m, conf = self.get_depth_at_pixel(kf_idx, u, v)
            if depth_m > 0:
                depth_measured = depth_m
                depth_confidence = conf
        
        # 深度差异
        depth_diff = abs(depth_measured - depth_projected)
        
        # Truncation distance检查（Voxblox最重要的特性！）
        if depth_diff > self.truncation_distance:
            return None  # 超出truncation范围，不记录这个观测
        
        # 计算权重
        weight = self.compute_observation_weight(depth_diff, depth_projected, grazing_factor)
        
        # 权重太低的观测也过滤掉
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
        过滤观测，保留连续的观测序列
        这是Voxblox保证观测质量的重要机制
        """
        if len(observations) < self.min_consecutive_observations:
            return []
        
        # 按KF索引排序
        observations = sorted(observations, key=lambda x: x['kf_idx'])
        
        # 找连续序列
        sequences = []
        current_seq = [observations[0]]
        
        for i in range(1, len(observations)):
            # 检查是否连续（允许跳过max_frame_gap帧）
            if observations[i]['kf_idx'] - observations[i-1]['kf_idx'] <= self.max_frame_gap:
                current_seq.append(observations[i])
            else:
                # 序列中断
                if len(current_seq) >= self.min_consecutive_observations:
                    sequences.append(current_seq)
                current_seq = [observations[i]]
        
        # 检查最后一个序列
        if len(current_seq) >= self.min_consecutive_observations:
            sequences.append(current_seq)
        
        # 合并所有有效序列
        filtered = []
        for seq in sequences:
            filtered.extend(seq)
        
        # 限制最大观测数
        if len(filtered) > self.max_observations_per_point:
            # 按权重排序，保留最好的观测
            filtered = sorted(filtered, key=lambda x: x['weight'], reverse=True)
            filtered = filtered[:self.max_observations_per_point]
            # 重新按KF索引排序
            filtered = sorted(filtered, key=lambda x: x['kf_idx'])
        
        return filtered
    
    def find_voxblox_correspondences(self, mesh_file: str, poses: Dict[int, np.ndarray], 
                                    timestamps: List[float], sample_size: int = 20000) -> Dict:
        """
        使用Voxblox风格找对应关系
        严格的过滤机制确保只有高质量的观测被保留
        """
        if not os.path.exists(mesh_file):
            print(f"Error: {mesh_file} not found!")
            return {}
        
        # 加载mesh
        mesh = o3d.io.read_triangle_mesh(mesh_file)
        vertices = np.asarray(mesh.vertices)
        
        # 计算法线（用于anti-grazing）
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()
        normals = np.asarray(mesh.vertex_normals)
        
        print(f"Mesh has {len(vertices)} vertices")
        
        # 采样
        if sample_size < len(vertices):
            indices = np.random.choice(len(vertices), sample_size, replace=False)
        else:
            indices = np.arange(len(vertices))
        
        print(f"Processing {len(indices)} points with Voxblox-style filtering...")
        
        point_observations = {}
        
        # 统计
        total_checks = 0
        truncation_filtered = 0
        grazing_filtered = 0
        consecutive_filtered = 0
        
        for i, idx in enumerate(indices):
            if i % 2000 == 0:
                print(f"  Processing {i}/{len(indices)} points...")
            
            point_3d = vertices[idx]
            normal = normals[idx] if idx < len(normals) else None
            
            # 收集所有潜在观测
            observations = []
            
            for kf_idx, T_cw in poses.items():
                total_checks += 1
                
                # 处理单个观测（包含所有Voxblox过滤）
                obs = self.process_single_observation(point_3d, kf_idx, T_cw, normal)
                
                if obs is not None:
                    observations.append(obs)
                else:
                    # 统计被过滤的原因（简化）
                    if len(observations) == 0:
                        truncation_filtered += 1
            
            # 过滤连续观测
            filtered_obs = self.filter_consecutive_observations(observations)
            
            if len(filtered_obs) < len(observations):
                consecutive_filtered += len(observations) - len(filtered_obs)
            
            # 保存有效观测
            if len(filtered_obs) >= self.min_consecutive_observations:
                point_observations[idx] = {
                    'point_3d': point_3d.tolist(),
                    'observations': filtered_obs
                }
        
        print(f"\nVoxblox-style filtering statistics:")
        print(f"  Total checks: {total_checks}")
        print(f"  Points with valid observations: {len(point_observations)}")
        
        # 计算观测分布
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
        """保存详细的对应关系数据"""
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
        
        # 保存位姿
        poses_file = output_file.replace('.txt', '_poses.npz')
        np.savez(poses_file,
                 poses_before={str(k): v for k, v in poses_before.items()},
                 poses_after={str(k): v for k, v in poses_after.items()})
        print(f"Saved poses to {poses_file}")
        
        # 统计信息
        total_obs = sum(len(data['observations']) for data in point_observations.values())
        avg_obs = total_obs / len(point_observations) if point_observations else 0
        
        print(f"\nFinal statistics:")
        print(f"  Total 3D points: {len(point_observations)}")
        print(f"  Total observations: {total_obs}")
        print(f"  Average observations per point: {avg_obs:.2f}")


def main():
    # 相机参数
    camera_params = {
        'fx': 377.535257164,  # 使用你的实际参数
        'fy': 377.209841379,
        'cx': 328.193371286,
        'cy': 240.426878936,
        'width': 640,
        'height': 480
    }
    
    # Voxblox参数（调整这些以控制观测数量！）
    voxblox_params = {
        'voxel_size': 0.02,                    # 2cm voxel
        'truncation_distance': 0.06,           # 6cm (3倍voxel size) - 这是关键！
        'min_depth': 0.3,
        'max_depth': 4.0,
        'max_grazing_angle': 70.0,             # 70度最大掠射角
        'min_consecutive_observations': 3,      # 至少3个连续观测
        'max_frame_gap': 2,                    # 允许跳过2帧
        'max_observations_per_point': 15       # 最多15个观测
    }
    
    # 文件路径
    # 文件路径

    rosbag_file = "/Datasets/Kimera/Kimera_Clipped_bag/12_07_thoth_clipped.bag"  # 你的rosbag文件

    

    mesh_file = "mesh_output.ply"  # voxblox输出的mesh

    trajectory_before_file = "standard_trajectory_no_loop.txt"  # loop closure前的轨迹

    trajectory_after_file = "standard_trajectory_with_loop.txt"  # loop closure后的轨迹

    output_file = "optimization_data.txt"  # 输出文件
    # 创建提取器
    extractor = VoxbloxStyleDataExtractor(camera_params, voxblox_params)
    
    # 1. 加载轨迹
    print("\n=== Loading trajectories ===")
    poses_before, kf_indices, timestamps = extractor.load_trajectory(trajectory_before_file)
    poses_after, _, _ = extractor.load_trajectory(trajectory_after_file)
    
    if len(poses_before) == 0:
        print("Error: No poses loaded!")
        return
    
    # 2. 加载深度图像（如果有rosbag）
    print("\n=== Loading depth images ===")
    if os.path.exists(rosbag_file):
        extractor.load_depth_from_rosbag(rosbag_file, kf_indices, timestamps)
    else:
        print(f"Warning: {rosbag_file} not found, using projected depth only")
    
    # 3. 找对应关系（使用Voxblox风格的严格过滤）
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
    
    # 4. 保存结果
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
