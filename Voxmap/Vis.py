import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

def load_trajectory(filename):
    """加载轨迹数据 (timestamp x y z qx qy qz qw)"""
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
        print(f"警告: 未找到文件 {filename}")
        return np.array([]), np.array([])
    
    return np.array(trajectory), np.array(timestamps)

def load_optimization_points(filename):
    """加载优化前后的点对应关系"""
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
                        movements.append(movement_mm / 1000.0)  # 转换为米
    except FileNotFoundError:
        print(f"警告: 未找到文件 {filename}")
        return np.array([]), np.array([]), np.array([]), np.array([])
    
    return np.array(points_before), np.array(points_after), np.array(point_ids), np.array(movements)

def create_trajectory_line(trajectory, color):
    """创建轨迹线"""
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
    """在轨迹的每个位置创建小球"""
    spheres = []
    # 降采样，每隔几个点显示一个球
    step = max(1, len(trajectory) // 50)  # 最多显示50个球
    
    for i in range(0, len(trajectory), step):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        sphere.translate(trajectory[i])
        sphere.paint_uniform_color(color)
        spheres.append(sphere)
    
    return spheres

def create_movement_lines(points_before, points_after, movements):
    """创建表示点移动的线段，根据移动距离着色"""
    if len(points_before) == 0:
        return o3d.geometry.LineSet()
    
    lines = []
    points = []
    colors = []
    
    # 计算颜色映射
    max_movement = movements.max() if len(movements) > 0 else 1.0
    
    for i in range(len(points_before)):
        # 添加线段的两个端点
        points.append(points_before[i])
        points.append(points_after[i])
        
        # 添加线段索引
        lines.append([i*2, i*2+1])
        
        # 根据移动距离设置颜色（绿色=小移动，红色=大移动）
        normalized = min(movements[i] / max_movement, 1.0)
        color = [normalized, 1-normalized, 0]  # 从绿到红
        colors.append(color)
    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    
    return line_set

def main():
    # 文件路径
    base_path = "/home/jixian/Desktop/orbslam3_docker/Datasets/Voxmap/"
    
    # 1. 加载优化前后的点
    print("加载优化前后的点对应关系...")
    optimization_file = base_path + "output/optimized_points_final.txt"
    
    points_before, points_after, point_ids, movements = load_optimization_points(optimization_file)
    
    if len(points_before) > 0:
        print(f"  成功加载了 {len(points_before)} 个点对")
    else:
        print("  警告: 未能加载优化点数据")
        points_before = np.array([])
        points_after = np.array([])
        movements = np.array([])
    
    # 2. 加载轨迹数据（loop closure前后）
    print("\n加载轨迹数据...")
    
    # 正确的轨迹文件路径
    traj_before_file = base_path + "standard_trajectory_no_loop.txt"
    traj_after_file = base_path + "standard_trajectory_with_loop.txt"
    
    traj_before, timestamps_before = load_trajectory(traj_before_file)
    traj_after, timestamps_after = load_trajectory(traj_after_file)
    
    if len(traj_before) > 0:
        print(f"  Loop closure前轨迹(no_loop): {len(traj_before)} 个位姿")
    else:
        print(f"  警告: 未找到文件 {traj_before_file}")
        
    if len(traj_after) > 0:
        print(f"  Loop closure后轨迹(with_loop): {len(traj_after)} 个位姿")
    else:
        print(f"  警告: 未找到文件 {traj_after_file}")
    
    # 3. 创建可视化元素
    geometries = []
    
    # 3.1 创建点云
    if len(points_before) > 0:
        # 优化前的点（蓝色）
        pcd_before = o3d.geometry.PointCloud()
        pcd_before.points = o3d.utility.Vector3dVector(points_before)
        pcd_before.paint_uniform_color([0.2, 0.2, 0.8])  # 蓝色
        geometries.append(pcd_before)
        
        # 优化后的点（绿色）
        pcd_after = o3d.geometry.PointCloud()
        pcd_after.points = o3d.utility.Vector3dVector(points_after)
        pcd_after.paint_uniform_color([0.2, 0.8, 0.2])  # 绿色
        geometries.append(pcd_after)
        
        # 创建移动线段
        movement_lines = create_movement_lines(points_before, points_after, movements)
        geometries.append(movement_lines)
        
        # 标记移动过大的点（红色球）
        large_movement_threshold = 0.5  # 0.5米
        large_movements = movements > large_movement_threshold
        if np.any(large_movements):
            print(f"\n警告: {np.sum(large_movements)} 个点移动超过{large_movement_threshold}米")
            outlier_indices = np.where(large_movements)[0]
            for idx in outlier_indices[:20]:  # 最多显示20个异常点
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
                sphere.translate(points_after[idx])
                sphere.paint_uniform_color([1.0, 0.0, 0.0])  # 红色
                geometries.append(sphere)
    
    # 3.2 创建轨迹可视化
    if len(traj_before) > 0:
        # Loop closure前的轨迹（青色）
        traj_line_before = create_trajectory_line(traj_before, [0, 0.8, 0.8])
        geometries.append(traj_line_before)
        
        # 在轨迹上添加关键帧标记
        spheres_before = create_trajectory_spheres(traj_before, [0, 0.6, 0.6], radius=0.03)
        geometries.extend(spheres_before)
        
        # 标记起点（大绿球）
        start_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
        start_sphere.translate(traj_before[0])
        start_sphere.paint_uniform_color([0.0, 1.0, 0.0])
        geometries.append(start_sphere)
    
    if len(traj_after) > 0:
        # Loop closure后的轨迹（橙色）
        traj_line_after = create_trajectory_line(traj_after, [1.0, 0.5, 0])
        geometries.append(traj_line_after)
        
        # 在轨迹上添加关键帧标记
        spheres_after = create_trajectory_spheres(traj_after, [0.8, 0.4, 0], radius=0.03)
        geometries.extend(spheres_after)
        
        # 标记终点（大红球）
        end_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
        end_sphere.translate(traj_after[-1])
        end_sphere.paint_uniform_color([1.0, 0.0, 0.0])
        geometries.append(end_sphere)
    
    # 3.3 如果两条轨迹点数相同，显示对应关系
    if len(traj_before) == len(traj_after) and len(traj_before) > 0:
        # 创建稀疏的对应线（每隔几个点连一条）
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
            corr_line_set.paint_uniform_color([0.5, 0.5, 0.5])  # 灰色
            geometries.append(corr_line_set)
    
    # 4. 添加坐标轴
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    geometries.append(coord_frame)
    
    # 5. 统计信息
    if len(movements) > 0:
        print("\n=== 点优化统计 ===")
        print(f"优化点数: {len(points_before)}")
        print(f"平均移动: {movements.mean():.3f} m")
        print(f"中位数移动: {np.median(movements):.3f} m")
        print(f"最大移动: {movements.max():.3f} m")
        print(f"最小移动: {movements.min():.3f} m")
    
    if len(traj_before) > 0 and len(traj_after) > 0 and len(traj_before) == len(traj_after):
        traj_movements = np.linalg.norm(traj_after - traj_before, axis=1)
        print("\n=== 轨迹优化统计 ===")
        print(f"平均轨迹偏移: {traj_movements.mean():.3f} m")
        print(f"最大轨迹偏移: {traj_movements.max():.3f} m")
    
    # 6. 可视化说明
    print("\n=== 可视化说明 ===")
    print("【3D点】")
    print("  蓝色点: 优化前位置")
    print("  绿色点: 优化后位置")
    print("  连线: 点的移动（绿→黄→红 表示移动距离增大）")
    print("  红色球: 移动异常大的点")
    print("\n【轨迹】")
    print("  青色线+球: Loop closure前的轨迹")
    print("  橙色线+球: Loop closure后的轨迹")
    print("  绿色大球: 轨迹起点")
    print("  红色大球: 轨迹终点")
    print("  灰色线: 轨迹对应点连线")
    print("\n控制: 鼠标拖动旋转，滚轮缩放，Ctrl+拖动平移")
    
    # 7. 创建可视化窗口
    if len(geometries) > 1:  # 只有坐标轴的话就不显示了
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Loop Closure优化可视化", width=1600, height=900)
        
        for geometry in geometries:
            vis.add_geometry(geometry)
        
        # 设置渲染选项
        render_option = vis.get_render_option()
        render_option.point_size = 8.0
        render_option.line_width = 3.0
        render_option.background_color = np.array([0.95, 0.95, 0.95])
        render_option.show_coordinate_frame = True
        
        # 设置视角
        ctr = vis.get_view_control()
        ctr.set_zoom(0.5)
        
        vis.run()
        vis.destroy_window()
    else:
        print("\n没有足够的数据进行可视化，请检查文件路径和格式")
    
    # 8. 生成统计图
    if len(movements) > 0:
        plt.figure(figsize=(12, 5))
        
        # 点移动距离分布
        plt.subplot(1, 2, 1)
        plt.hist(movements, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        plt.axvline(x=movements.mean(), color='r', linestyle='--', label=f'均值: {movements.mean():.3f}m')
        plt.axvline(x=np.median(movements), color='g', linestyle='--', label=f'中位数: {np.median(movements):.3f}m')
        plt.xlabel('移动距离 (m)')
        plt.ylabel('点数')
        plt.title('3D点优化移动距离分布')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 轨迹偏移分布（如果有）
        if len(traj_before) > 0 and len(traj_after) > 0 and len(traj_before) == len(traj_after):
            plt.subplot(1, 2, 2)
            traj_movements = np.linalg.norm(traj_after - traj_before, axis=1)
            plt.plot(range(len(traj_movements)), traj_movements, 'o-', markersize=3)
            plt.xlabel('轨迹点索引')
            plt.ylabel('位置偏移 (m)')
            plt.title('Loop Closure轨迹优化偏移')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(base_path + 'loop_closure_statistics.png', dpi=150)
        print(f"\n统计图已保存到: {base_path}loop_closure_statistics.png")
        plt.show()

if __name__ == "__main__":
    main()
