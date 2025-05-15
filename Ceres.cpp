#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <set>
#include <string>
#include <sstream>
#include <algorithm>
#include <iomanip>
#include <ceres/ceres.h>
#include <ceres/autodiff_cost_function.h>
#include <ceres/product_manifold.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <sophus/se3.hpp>

// 相对位姿约束的自动求导代价函数 - 使用更稳定的实现
struct SE3RelativePoseFunctor {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    SE3RelativePoseFunctor(const Eigen::Matrix4d& T_ij, double weight = 1.0)
        : m_T_ij(T_ij), m_weight(weight) {
        // 从T_ij中提取旋转和平移
        m_R_ij = m_T_ij.block<3, 3>(0, 0);
        m_t_ij = m_T_ij.block<3, 1>(0, 3);
    }

    template <typename T>
    bool operator()(const T* pose_i, const T* pose_j, T* residuals) const {
        // 提取四元数 (注意: Eigen四元数顺序为 w,x,y,z)
        const Eigen::Quaternion<T> q_i(pose_i[3], pose_i[0], pose_i[1], pose_i[2]);
        const Eigen::Map<const Eigen::Matrix<T, 3, 1>> t_i(pose_i + 4);
        
        const Eigen::Quaternion<T> q_j(pose_j[3], pose_j[0], pose_j[1], pose_j[2]);
        const Eigen::Map<const Eigen::Matrix<T, 3, 1>> t_j(pose_j + 4);
        
        // 转换测量约束为T类型
        const Eigen::Matrix<T, 3, 3> R_ij = m_R_ij.template cast<T>();
        const Eigen::Matrix<T, 3, 1> t_ij = m_t_ij.template cast<T>();
        
        // 计算相对旋转: q_i^-1 * q_j
        const Eigen::Quaternion<T> q_i_inverse = q_i.conjugate();
        const Eigen::Quaternion<T> q_ij_expected = q_i_inverse * q_j;
        
        // 计算相对平移: R_i^-1 * (t_j - t_i)
        const Eigen::Matrix<T, 3, 3> R_i_inverse = q_i_inverse.toRotationMatrix();
        const Eigen::Matrix<T, 3, 1> t_ij_expected = R_i_inverse * (t_j - t_i);
        
        // 计算旋转误差
        Eigen::Matrix<T, 3, 1> rotation_error;
        {
            // 转换为旋转矩阵
            const Eigen::Matrix<T, 3, 3> R_expected = q_ij_expected.toRotationMatrix();
            const Eigen::Matrix<T, 3, 3> R_error = R_ij.transpose() * R_expected;
            
            // 转换为轴角表示(对数映射)
            const Eigen::Quaternion<T> q_error(R_error);
            
            // 确保有效的acos输入
            T angle_scalar = q_error.w();
            angle_scalar = angle_scalar < T(-1.0) ? T(-1.0) : (angle_scalar > T(1.0) ? T(1.0) : angle_scalar);
            
            // 计算角度
            const T angle = T(2.0) * acos(angle_scalar);
            
            if (angle < T(1e-10)) {
                rotation_error.setZero();
            } else {
                const T s = sqrt(T(1.0) - angle_scalar * angle_scalar);
                if (s < T(1e-10)) {
                    rotation_error = Eigen::Matrix<T, 3, 1>(T(0), T(0), T(0));
                } else {
                    rotation_error = Eigen::Matrix<T, 3, 1>(
                        q_error.x() / s,
                        q_error.y() / s,
                        q_error.z() / s
                    ) * angle;
                }
            }
        }
        
        // 计算平移误差
        const Eigen::Matrix<T, 3, 1> translation_error = t_ij_expected - t_ij;
        
        // 应用权重
        const T weight_t = T(m_weight);
        for (int i = 0; i < 3; ++i) {
            residuals[i] = rotation_error[i] * weight_t;
            residuals[i + 3] = translation_error[i] * weight_t;
        }
        
        return true;
    }

private:
    Eigen::Matrix4d m_T_ij;
    Eigen::Matrix3d m_R_ij;
    Eigen::Vector3d m_t_ij;
    double m_weight;
};

// 解析 4x4 矩阵字符串到 Eigen::Matrix4d
Eigen::Matrix4d parseMatrix4d(const std::string& matrix_str) {
    std::istringstream iss(matrix_str);
    std::vector<double> values;
    double value;
    
    while (iss >> value) {
        values.push_back(value);
    }
    
    Eigen::Matrix4d mat = Eigen::Matrix4d::Identity();
    if (values.size() == 16) {
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                mat(i, j) = values[i * 4 + j];
            }
        }
    }
    
    return mat;
}

// 从位姿数据创建四元数
Eigen::Quaterniond getQuaternionFromPose(const std::vector<double>& pose) {
    return Eigen::Quaterniond(pose[3], pose[0], pose[1], pose[2]); // w, x, y, z
}

// 从位姿数据创建平移向量
Eigen::Vector3d getTranslationFromPose(const std::vector<double>& pose) {
    return Eigen::Vector3d(pose[4], pose[5], pose[6]);
}

// 使用 Ceres 优化位姿图
void OptimizeEssentialGraph(const std::string& data_dir, bool bFixScale = true) {
    std::cout << "使用 Ceres 2.2 优化回环闭合的位姿图..." << std::endl;
    
    // 1. 读取关键帧位姿
    std::map<int, std::vector<double>> keyframe_poses;
    std::ifstream kf_poses_file(data_dir + "keyframe_poses.txt");
    if (!kf_poses_file.is_open()) {
        std::cerr << "错误：无法打开 keyframe_poses.txt" << std::endl;
        return;
    }
    
    std::string line;
    std::getline(kf_poses_file, line); // 跳过标题行
    int kf_id;
    double timestamp, tx, ty, tz, qx, qy, qz, qw;
    
    while (kf_poses_file >> kf_id >> timestamp >> tx >> ty >> tz >> qx >> qy >> qz >> qw) {
        std::vector<double> pose = {qx, qy, qz, qw, tx, ty, tz}; // qx, qy, qz, qw, tx, ty, tz 格式
        keyframe_poses[kf_id] = pose;
    }
    
    std::cout << "读取了 " << keyframe_poses.size() << " 个关键帧的位姿" << std::endl;

    // 2. 读取回环匹配信息
    std::ifstream loop_match_file(data_dir + "loop_match.txt");
    if (!loop_match_file.is_open()) {
        std::cerr << "错误：无法打开 loop_match.txt" << std::endl;
        return;
    }
    
    int current_kf_id, loop_kf_id;
    
    // 跳过标题行
    std::getline(loop_match_file, line); // 跳过第一行（标题）
    std::getline(loop_match_file, line); // 跳过第二行（注释）
    
    // 读取当前关键帧和回环关键帧的 ID
    loop_match_file >> current_kf_id >> loop_kf_id;
    
    // 读取变换矩阵
    std::string matrix_line;
    std::getline(loop_match_file, matrix_line); // 清除读取 ID 后的换行符
    std::getline(loop_match_file, matrix_line); // 读取实际的矩阵行
    Eigen::Matrix4d loop_constraint = parseMatrix4d(matrix_line);
    
    std::cout << "关键帧 " << current_kf_id << " 和关键帧 " << loop_kf_id << " 之间的回环约束" << std::endl;
    
    // 3. 读取关键帧信息，包括最大关键帧 ID
    int init_kf_id = 0;
    int max_kf_id = 0;
    std::ifstream map_info_file(data_dir + "map_info.txt");
    if (map_info_file.is_open()) {
        std::string key;
        map_info_file >> key >> key; // MAP_ID
        map_info_file >> key >> init_kf_id; // INIT_KF_ID
        map_info_file >> key >> max_kf_id; // MAX_KF_ID
        std::cout << "初始关键帧: " << init_kf_id << ", 最大关键帧 ID: " << max_kf_id << std::endl;
    } else {
        // 如果无法读取文件，则从 keyframe_poses 估计 max_kf_id
        for (const auto& kf_pair : keyframe_poses) {
            max_kf_id = std::max(max_kf_id, kf_pair.first);
        }
        std::cout << "无法读取 map_info.txt，估计最大关键帧 ID: " << max_kf_id << std::endl;
    }
    
    // 4. 读取已校正的 Sim3 位姿（对于 RGBD/双目系统，这些是尺度为 1 的 SE3）
    std::map<int, std::vector<double>> corrected_poses;
    std::ifstream corrected_file(data_dir + "corrected_sim3.txt");
    if (corrected_file.is_open()) {
        std::getline(corrected_file, line); // 跳过标题行
        int kf_id;
        double scale, tx, ty, tz, qx, qy, qz, qw;
        
        while (corrected_file >> kf_id >> scale >> tx >> ty >> tz >> qx >> qy >> qz >> qw) {
            std::vector<double> pose = {qx, qy, qz, qw, tx, ty, tz}; // qx, qy, qz, qw, tx, ty, tz 格式
            corrected_poses[kf_id] = pose;
        }
        
        std::cout << "读取了 " << corrected_poses.size() << " 个关键帧的已校正位姿" << std::endl;
    }
    
    // 5. 读取未校正的 Sim3 位姿（已校正关键帧的原始位姿）
    std::map<int, std::vector<double>> non_corrected_poses;
    std::ifstream non_corrected_file(data_dir + "non_corrected_sim3.txt");
    if (non_corrected_file.is_open()) {
        std::getline(non_corrected_file, line); // 跳过标题行
        int kf_id;
        double scale, tx, ty, tz, qx, qy, qz, qw;
        
        while (non_corrected_file >> kf_id >> scale >> tx >> ty >> tz >> qx >> qy >> qz >> qw) {
            std::vector<double> pose = {qx, qy, qz, qw, tx, ty, tz}; // qx, qy, qz, qw, tx, ty, tz 格式
            non_corrected_poses[kf_id] = pose;
        }
        
        std::cout << "读取了 " << non_corrected_poses.size() << " 个关键帧的未校正位姿" << std::endl;
    }
    
    // 6. 读取生成树信息（子帧->父帧映射）
    std::map<int, int> spanning_tree; // 子帧 -> 父帧
    std::ifstream spanning_tree_file(data_dir + "spanning_tree.txt");
    if (!spanning_tree_file.is_open()) {
        std::cerr << "错误：无法打开 spanning_tree.txt" << std::endl;
        return;
    }
    
    std::getline(spanning_tree_file, line); // 跳过标题行
    int child_id, parent_id;
    
    while (spanning_tree_file >> child_id >> parent_id) {
        spanning_tree[child_id] = parent_id;
    }
    
    std::cout << "读取了 " << spanning_tree.size() << " 条生成树边" << std::endl;
    
    // 7. 读取共视图（关键帧之间的强边）
    const int minFeat = 100; // 强边的最小特征数
    std::map<std::pair<int, int>, int> covisibility_weights; // (kf1_id, kf2_id) -> 权重
    std::ifstream covisibility_file(data_dir + "covisibility.txt");
    if (!covisibility_file.is_open()) {
        std::cerr << "错误：无法打开 covisibility.txt" << std::endl;
        return;
    }
    
    std::getline(covisibility_file, line); // 跳过标题行
    int kf1_id, kf2_id, weight;
    
    while (covisibility_file >> kf1_id >> kf2_id >> weight) {
        covisibility_weights[std::make_pair(kf1_id, kf2_id)] = weight;
    }
    
    std::cout << "读取了 " << covisibility_weights.size() << " 条边的共视信息" << std::endl;

    // 8. 读取回环连接（如果可用）
    std::map<int, std::set<int>> loop_connections;
    std::ifstream loop_conn_file(data_dir + "loop_connections.txt");
    if (loop_conn_file.is_open()) {
        std::getline(loop_conn_file, line); // 跳过标题行
        
        while (std::getline(loop_conn_file, line)) {
            if (line.empty() || line[0] == '#') continue;
            
            std::istringstream iss(line);
            int kf_id;
            if (!(iss >> kf_id)) continue;
            
            std::set<int> connections;
            int connected_kf;
            while (iss >> connected_kf) {
                connections.insert(connected_kf);
            }
            
            if (!connections.empty()) {
                loop_connections[kf_id] = connections;
            }
        }
        
        std::cout << "读取了 " << loop_connections.size() << " 个关键帧的回环连接" << std::endl;
    } else {
        // 如果 loop_connections.txt 不存在，创建一个只包含回环闭合对的默认连接
        std::set<int> loop_kfs;
        loop_kfs.insert(loop_kf_id);
        loop_connections[current_kf_id] = loop_kfs;
        std::cout << "创建了关键帧 " << current_kf_id << " 和关键帧 " << loop_kf_id << " 之间的默认回环连接" << std::endl;
    }

    // 获取两个关键帧之间的共视权重的辅助函数
    auto getCovisibilityWeight = [&](int kf1_id, int kf2_id) -> int {
        auto it = covisibility_weights.find(std::make_pair(kf1_id, kf2_id));
        if (it != covisibility_weights.end()) {
            return it->second;
        }
        
        // 检查反向顺序
        it = covisibility_weights.find(std::make_pair(kf2_id, kf1_id));
        if (it != covisibility_weights.end()) {
            return it->second;
        }
        
        return 0; // 未找到共视关系
    };

    // 9. 设置优化问题
    ceres::Problem problem;
    
    // 创建 SE3 流形 - 使用 Ceres 内置的 ProductManifold
    // 将四元数流形和欧几里得流形组合成 SE3
// 创建 SE3 流形 - 使用单独的实例化而不是模板参数列表
    auto* quaternion_manifold = new ceres::EigenQuaternionManifold();
    auto* translation_manifold = new ceres::EuclideanManifold<3>();
    ceres::Manifold* se3_manifold = new ceres::ProductManifold(quaternion_manifold, translation_manifold);
    
    // 存储优化后的位姿
    std::map<int, double*> optimized_poses;
    
    // 用于跟踪已分配内存的向量
    std::vector<double*> allocated_memory;
    
    // 10. 将关键帧位姿作为变量添加到优化中
    for (auto& kf_pose : keyframe_poses) {
        int kf_id = kf_pose.first;
        
        // 为位姿分配内存
        double* pose = new double[7];
        allocated_memory.push_back(pose);
        
        // 检查这个关键帧是否有校正后的位姿
        if (corrected_poses.find(kf_id) != corrected_poses.end()) {
            // 使用校正后的位姿作为初始值
            for (int i = 0; i < 7; i++) {
                pose[i] = corrected_poses[kf_id][i];
            }
        } else {
            // 使用原始位姿
            for (int i = 0; i < 7; i++) {
                pose[i] = kf_pose.second[i];
            }
        }
        
        optimized_poses[kf_id] = pose;
        
        // 将变量添加到优化问题中
        problem.AddParameterBlock(optimized_poses[kf_id], 7, se3_manifold);
        
        // 固定初始关键帧
        if (kf_id == init_kf_id) {
            problem.SetParameterBlockConstant(optimized_poses[kf_id]);
            std::cout << "固定关键帧 " << kf_id << "（初始关键帧）" << std::endl;
        }
    }
    
    // 创建损失函数
    // 对于回环闭合和边界约束使用强度不同的鲁棒核函数
    ceres::LossFunction* loop_huber_loss = new ceres::HuberLoss(1.0);
    ceres::LossFunction* boundary_huber_loss = new ceres::HuberLoss(1.0);
    ceres::LossFunction* normal_huber_loss = new ceres::HuberLoss(1.0);
    
    // 调整权重 - 降低权重的相对差异
    const double loop_weight = 1000.0;      // 降低从10000到1000
    const double boundary_weight = 500.0;   // 降低从1000到500
    const double normal_weight = 100.0;     // 保持不变
    
    // 11. 添加回环闭合约束
    // 11. 添加回环闭合约束
    std::cout << "开始添加回环约束..." << std::endl;
    std::set<std::pair<int, int>> sInsertedEdges;
    int count_loop = 0;
    int skipped_by_weight = 0;
    int skipped_by_missing_pose = 0;
    
    // 首先添加主回环闭合约束 (pCurKF到pLoopKF)
    if (optimized_poses.find(current_kf_id) != optimized_poses.end() && 
        optimized_poses.find(loop_kf_id) != optimized_poses.end()) {
        
        // 使用自动求导代价函数创建回环约束
        ceres::CostFunction* loop_cost_function = 
            new ceres::AutoDiffCostFunction<SE3RelativePoseFunctor, 6, 7, 7>(
                new SE3RelativePoseFunctor(loop_constraint, loop_weight)
            );
        
        problem.AddResidualBlock(
            loop_cost_function,
            loop_huber_loss, // 使用鲁棒损失函数
            optimized_poses[current_kf_id],
            optimized_poses[loop_kf_id]
        );
        
        // 标记这条边已插入
        sInsertedEdges.insert(std::make_pair(std::min(current_kf_id, loop_kf_id), 
                                            std::max(current_kf_id, loop_kf_id)));
        count_loop++;
        
        std::cout << "添加了关键帧 " << current_kf_id 
                << " 和关键帧 " << loop_kf_id << " 之间的回环闭合约束，权重为 " << loop_weight << std::endl;
    }
    
    // 然后，按照ORB-SLAM3的方式处理其他回环连接
    for (const auto& kf_pair : loop_connections) {
        int kf1_id = kf_pair.first;
        
        // 如果这个关键帧不在我们的 optimized_poses 中，则跳过
        if (optimized_poses.find(kf1_id) == optimized_poses.end()) {
            skipped_by_missing_pose++;
            continue;
        }
        
        // 获取并准备当前关键帧的位姿变换
        Eigen::Quaterniond q1;
        Eigen::Vector3d t1;
        bool kf1_corrected = corrected_poses.find(kf1_id) != corrected_poses.end();
        
        if (kf1_corrected) {
            q1 = getQuaternionFromPose(corrected_poses[kf1_id]);
            t1 = getTranslationFromPose(corrected_poses[kf1_id]);
        } else {
            q1 = getQuaternionFromPose(keyframe_poses[kf1_id]);
            t1 = getTranslationFromPose(keyframe_poses[kf1_id]);
        }
        
        Eigen::Matrix3d R1 = q1.toRotationMatrix();
        
        const std::set<int>& connected_kfs = kf_pair.second;
        
        for (int kf2_id : connected_kfs) {
            // 如果连接的关键帧不在我们的 optimized_poses 中，则跳过
            if (optimized_poses.find(kf2_id) == optimized_poses.end()) {
                skipped_by_missing_pose++;
                continue;
            }
            
            // 检查是否为主回环连接，或具有足够的特征点
            bool isMainLoopEdge = (kf1_id == current_kf_id && kf2_id == loop_kf_id) || 
                                 (kf1_id == loop_kf_id && kf2_id == current_kf_id);
                                 
            // 如果不是主回环边且权重低，则跳过 - 仿照ORB-SLAM3的逻辑
            if (!isMainLoopEdge) {
                int covis_weight = getCovisibilityWeight(kf1_id, kf2_id);
                if (covis_weight < minFeat) {
                    skipped_by_weight++;
                    continue;
                }
            }
            
            // 如果已添加过此边，则跳过
            if (sInsertedEdges.count(std::make_pair(std::min(kf1_id, kf2_id), 
                                                  std::max(kf1_id, kf2_id)))) {
                continue;
            }
            
            // 获取并准备连接关键帧的位姿变换
            Eigen::Quaterniond q2;
            Eigen::Vector3d t2;
            bool kf2_corrected = corrected_poses.find(kf2_id) != corrected_poses.end();
            
            if (kf2_corrected) {
                q2 = getQuaternionFromPose(corrected_poses[kf2_id]);
                t2 = getTranslationFromPose(corrected_poses[kf2_id]);
            } else {
                q2 = getQuaternionFromPose(keyframe_poses[kf2_id]);
                t2 = getTranslationFromPose(keyframe_poses[kf2_id]);
            }
            
            Eigen::Matrix3d R2 = q2.toRotationMatrix();
            
            // 计算相对位姿变换 - 这里采用SE3的形式而不是Sim3，因为尺度都为1
            Eigen::Matrix3d R12 = R1.transpose() * R2;
            Eigen::Vector3d t12 = R1.transpose() * (t2 - t1);
            
            Eigen::Matrix4d T12 = Eigen::Matrix4d::Identity();
            T12.block<3, 3>(0, 0) = R12;
            T12.block<3, 1>(0, 3) = t12;
            
            // 使用合适的权重 - 对回环连接使用较高权重
            double edge_weight = normal_weight;
            if (isMainLoopEdge) {
                edge_weight = loop_weight; // 主回环边使用最高权重
            } else {
                // 可以根据共视点数量动态调整权重
                int covis_weight = getCovisibilityWeight(kf1_id, kf2_id);
                edge_weight = std::max(normal_weight, std::min(loop_weight/2.0, normal_weight * covis_weight / 100.0));
            }
            
            // 创建并添加回环连接约束
            ceres::CostFunction* loop_conn_cost = 
                new ceres::AutoDiffCostFunction<SE3RelativePoseFunctor, 6, 7, 7>(
                    new SE3RelativePoseFunctor(T12, edge_weight)
                );
            
            ceres::LossFunction* loss_function = (isMainLoopEdge) ? loop_huber_loss : normal_huber_loss;
            
            problem.AddResidualBlock(
                loop_conn_cost,
                loss_function,
                optimized_poses[kf1_id],
                optimized_poses[kf2_id]
            );
            
            // 标记此边已添加
            sInsertedEdges.insert(std::make_pair(std::min(kf1_id, kf2_id), 
                                                std::max(kf1_id, kf2_id)));
            count_loop++;
            
            // 打印调试信息
            std::string edge_type = isMainLoopEdge ? "主回环" : "回环连接";
            std::cout << "添加了关键帧 " << kf1_id << " 和关键帧 " << kf2_id 
                    << " 之间的" << edge_type << "约束，权重为 " << edge_weight << std::endl;
        }
    }
    
    std::cout << "成功添加了 " << count_loop << " 条回环约束" << std::endl;
    std::cout << "因共视权重不足跳过了 " << skipped_by_weight << " 条边" << std::endl;
    std::cout << "因关键帧缺失跳过了 " << skipped_by_missing_pose << " 条边" << std::endl;


        
    // 12. 添加生成树约束
    for (auto& edge : spanning_tree) {
        int child_id = edge.first;
        int parent_id = edge.second;
        
        // 如果任一关键帧不在我们的 optimized_poses 中，则跳过
        if (optimized_poses.find(child_id) == optimized_poses.end() || 
            optimized_poses.find(parent_id) == optimized_poses.end()) {
            continue;
        }
        
        // 如果我们已经添加了这条边，则跳过
        if (inserted_edges.count(std::make_pair(std::min(child_id, parent_id), std::max(child_id, parent_id)))) {
            continue;
        }
        
        // 检查这是否是校正区域和未校正区域之间的边界
        bool child_corrected = corrected_poses.find(child_id) != corrected_poses.end();
        bool parent_corrected = corrected_poses.find(parent_id) != corrected_poses.end();
        bool is_boundary = (child_corrected != parent_corrected);
        
        // 根据区域设置正确的约束
        Eigen::Matrix4d T_parent_child = Eigen::Matrix4d::Identity();
        
        if (is_boundary) {
            // 对于边界约束，使用校正前和原始位姿
            int corrected_id = parent_corrected ? parent_id : child_id;
            int uncorrected_id = parent_corrected ? child_id : parent_id;
            
            if (non_corrected_poses.find(corrected_id) != non_corrected_poses.end()) {
                // 获取校正关键帧的校正前位姿
                Eigen::Quaterniond q_corrected_pre = getQuaternionFromPose(non_corrected_poses.at(corrected_id));
                Eigen::Vector3d t_corrected_pre = getTranslationFromPose(non_corrected_poses.at(corrected_id));
                
                // 获取未校正关键帧的原始位姿
                Eigen::Quaterniond q_uncorrected = getQuaternionFromPose(keyframe_poses.at(uncorrected_id));
                Eigen::Vector3d t_uncorrected = getTranslationFromPose(keyframe_poses.at(uncorrected_id));
                
                // 计算应该保持的真实相对变换
                Eigen::Matrix3d R_corrected_pre = q_corrected_pre.toRotationMatrix();
                Eigen::Matrix3d R_uncorrected = q_uncorrected.toRotationMatrix();
                
                if (corrected_id == parent_id) {
                    // 父节点已校正，子节点未校正
                    Eigen::Matrix3d R_rel = R_corrected_pre.transpose() * R_uncorrected;
                    Eigen::Vector3d t_rel = R_corrected_pre.transpose() * (t_uncorrected - t_corrected_pre);
                    
                    T_parent_child.block<3, 3>(0, 0) = R_rel;
                    T_parent_child.block<3, 1>(0, 3) = t_rel;
                } else {
                    // 子节点已校正，父节点未校正
                    Eigen::Matrix3d R_rel = R_uncorrected.transpose() * R_corrected_pre;
                    Eigen::Vector3d t_rel = R_uncorrected.transpose() * (t_corrected_pre - t_uncorrected);
                    
                    T_parent_child.block<3, 3>(0, 0) = R_rel;
                    T_parent_child.block<3, 1>(0, 3) = t_rel;
                }
                
                std::cout << "使用真实相对位姿作为关键帧 " 
                          << parent_id << " 和关键帧 " << child_id << " 之间的边界约束" << std::endl;
            } else {
                // 如果没有校正前数据，则回退到标准方法
                std::cerr << "警告：边界关键帧 " << corrected_id 
                          << " 没有校正前数据，使用可能不正确的约束！" << std::endl;
                
                // 使用当前位姿（可能导致不一致的约束）
                Eigen::Quaterniond q_parent, q_child;
                Eigen::Vector3d t_parent, t_child;
                
                if (parent_corrected) {
                    q_parent = getQuaternionFromPose(corrected_poses[parent_id]);
                    t_parent = getTranslationFromPose(corrected_poses[parent_id]);
                    q_child = getQuaternionFromPose(keyframe_poses[child_id]);
                    t_child = getTranslationFromPose(keyframe_poses[child_id]);
                } else {
                    q_parent = getQuaternionFromPose(keyframe_poses[parent_id]);
                    t_parent = getTranslationFromPose(keyframe_poses[parent_id]);
                    q_child = getQuaternionFromPose(corrected_poses[child_id]);
                    t_child = getTranslationFromPose(corrected_poses[child_id]);
                }
                
                Eigen::Matrix3d R_parent = q_parent.toRotationMatrix();
                Eigen::Matrix3d R_child = q_child.toRotationMatrix();
                
                Eigen::Matrix3d R_parent_child = R_parent.transpose() * R_child;
                Eigen::Vector3d t_parent_child = R_parent.transpose() * (t_child - t_parent);
                
                T_parent_child.block<3, 3>(0, 0) = R_parent_child;
                T_parent_child.block<3, 1>(0, 3) = t_parent_child;
            }
        } else if (parent_corrected && child_corrected) {
            // 两个都在校正区域 - 使用校正后的位姿
            Eigen::Quaterniond q_parent = getQuaternionFromPose(corrected_poses[parent_id]);
            Eigen::Vector3d t_parent = getTranslationFromPose(corrected_poses[parent_id]);
            
            Eigen::Quaterniond q_child = getQuaternionFromPose(corrected_poses[child_id]);
            Eigen::Vector3d t_child = getTranslationFromPose(corrected_poses[child_id]);
            
            Eigen::Matrix3d R_parent = q_parent.toRotationMatrix();
            Eigen::Matrix3d R_child = q_child.toRotationMatrix();
            
            Eigen::Matrix3d R_parent_child = R_parent.transpose() * R_child;
            Eigen::Vector3d t_parent_child = R_parent.transpose() * (t_child - t_parent);
            
            T_parent_child.block<3, 3>(0, 0) = R_parent_child;
            T_parent_child.block<3, 1>(0, 3) = t_parent_child;
        } else {
            // 两个都在未校正区域 - 使用原始位姿
            Eigen::Quaterniond q_parent = getQuaternionFromPose(keyframe_poses[parent_id]);
            Eigen::Vector3d t_parent = getTranslationFromPose(keyframe_poses[parent_id]);
            
            Eigen::Quaterniond q_child = getQuaternionFromPose(keyframe_poses[child_id]);
            Eigen::Vector3d t_child = getTranslationFromPose(keyframe_poses[child_id]);
            
            Eigen::Matrix3d R_parent = q_parent.toRotationMatrix();
            Eigen::Matrix3d R_child = q_child.toRotationMatrix();
            
            Eigen::Matrix3d R_parent_child = R_parent.transpose() * R_child;
            Eigen::Vector3d t_parent_child = R_parent.transpose() * (t_child - t_parent);
            
            T_parent_child.block<3, 3>(0, 0) = R_parent_child;
            T_parent_child.block<3, 1>(0, 3) = t_parent_child;
        }
        
        // 确定权重 - 边界约束的权重更高
        double constraint_weight = is_boundary ? boundary_weight : normal_weight;
        
        // 添加残差
        ceres::CostFunction* spanning_tree_cost_function = 
            new ceres::AutoDiffCostFunction<SE3RelativePoseFunctor, 6, 7, 7>(
                new SE3RelativePoseFunctor(T_parent_child, constraint_weight)
            );
        
        problem.AddResidualBlock(
            spanning_tree_cost_function,
            is_boundary ? boundary_huber_loss : normal_huber_loss, // 使用与约束类型匹配的鲁棒损失函数
            optimized_poses[parent_id],
            optimized_poses[child_id]
        );
        
        // 如果这是边界约束，打印一条消息
        if (is_boundary) {
            std::cout << "添加了关键帧 " << parent_id 
                      << " 和关键帧 " << child_id << " 之间的边界约束，权重为 " << constraint_weight << std::endl;
        }
        
        // 标记这条边已插入
        inserted_edges.insert(std::make_pair(std::min(child_id, parent_id), std::max(child_id, parent_id)));
    }

    // 13. 添加共视图约束
    for (const auto& weight_entry : covisibility_weights) {
        int kf1_id = weight_entry.first.first;
        int kf2_id = weight_entry.first.second;
        int weight = weight_entry.second;
        
        // 跳过低权重边
        if (weight < minFeat) {
            continue;
        }
        
        // 如果任一关键帧不在我们的 optimized_poses 中，则跳过
        if (optimized_poses.find(kf1_id) == optimized_poses.end() || 
            optimized_poses.find(kf2_id) == optimized_poses.end()) {
            continue;
        }
        
        // 如果我们已经添加了这条边，则跳过
        if (inserted_edges.count(std::make_pair(std::min(kf1_id, kf2_id), std::max(kf1_id, kf2_id)))) {
            continue;
        }
        
        // 跳过生成树中的父子关系
        if (spanning_tree[kf1_id] == kf2_id || spanning_tree[kf2_id] == kf1_id) {
            continue;
        }
        
        // 检查这些关键帧是否都在校正区域或都在未校正区域
        bool kf1_corrected = corrected_poses.find(kf1_id) != corrected_poses.end();
        bool kf2_corrected = corrected_poses.find(kf2_id) != corrected_poses.end();
        
        // 跳过跨越边界的边（一个已校正，一个未校正）
        if (kf1_corrected != kf2_corrected) {
            continue;
        }
        
        // 设置变换约束
        Eigen::Matrix4d T_kf1_kf2 = Eigen::Matrix4d::Identity();
        
        if (kf1_corrected && kf2_corrected) {
            // 两个都在校正区域 - 使用校正后的位姿
            Eigen::Quaterniond q1 = getQuaternionFromPose(corrected_poses[kf1_id]);
            Eigen::Vector3d t1 = getTranslationFromPose(corrected_poses[kf1_id]);
            
            Eigen::Quaterniond q2 = getQuaternionFromPose(corrected_poses[kf2_id]);
            Eigen::Vector3d t2 = getTranslationFromPose(corrected_poses[kf2_id]);
            
            Eigen::Matrix3d R1 = q1.toRotationMatrix();
            Eigen::Matrix3d R2 = q2.toRotationMatrix();
            
            Eigen::Matrix3d R12 = R1.transpose() * R2;
            Eigen::Vector3d t12 = R1.transpose() * (t2 - t1);
            
            T_kf1_kf2.block<3, 3>(0, 0) = R12;
            T_kf1_kf2.block<3, 1>(0, 3) = t12;
        } else {
            // 两个都在未校正区域 - 使用原始位姿
            Eigen::Quaterniond q1 = getQuaternionFromPose(keyframe_poses[kf1_id]);
            Eigen::Vector3d t1 = getTranslationFromPose(keyframe_poses[kf1_id]);
            
            Eigen::Quaterniond q2 = getQuaternionFromPose(keyframe_poses[kf2_id]);
            Eigen::Vector3d t2 = getTranslationFromPose(keyframe_poses[kf2_id]);
            
            Eigen::Matrix3d R1 = q1.toRotationMatrix();
            Eigen::Matrix3d R2 = q2.toRotationMatrix();
            
            Eigen::Matrix3d R12 = R1.transpose() * R2;
            Eigen::Vector3d t12 = R1.transpose() * (t2 - t1);
            
            T_kf1_kf2.block<3, 3>(0, 0) = R12;
            T_kf1_kf2.block<3, 1>(0, 3) = t12;
        }
        
        // 根据共视性缩放权重
        double weight_factor = std::min(2.0, weight / 100.0);
        double constraint_weight = 10.0 * weight_factor;
        
        ceres::CostFunction* covisibility_cost = 
            new ceres::AutoDiffCostFunction<SE3RelativePoseFunctor, 6, 7, 7>(
                new SE3RelativePoseFunctor(T_kf1_kf2, constraint_weight)
            );
        
        problem.AddResidualBlock(
            covisibility_cost,
            normal_huber_loss, // 使用鲁棒损失函数
            optimized_poses[kf1_id],
            optimized_poses[kf2_id]
        );
        
        // 标记这条边已插入
        inserted_edges.insert(std::make_pair(std::min(kf1_id, kf2_id), std::max(kf1_id, kf2_id)));
    }

    // 14. 配置求解器 - 改进的优化器配置
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.use_nonmonotonic_steps = true;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 100;  // 增加迭代次数
    options.function_tolerance = 1e-6;
    options.gradient_tolerance = 1e-10;
    options.parameter_tolerance = 1e-8;
    options.check_gradients = false;  // 禁用梯度检查以提高性能
    options.gradient_check_relative_precision = 1e-4;  // 梯度检查的相对精度
    
    // 15. 运行求解器
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    
    std::cout << summary.BriefReport() << std::endl;
    
    // 16. 将优化后的位姿写入文件（从 Tcw 转换为 Twc 格式）
    // 如果输出目录不存在，则创建它
    std::string output_dir = "/Datasets/CERES_Work/output/";
    system(("mkdir -p " + output_dir).c_str());
    
    std::string output_file = output_dir + "optimized_poses.txt";
    std::ofstream out_file(output_file);
    
    if (!out_file.is_open()) {
        std::cerr << "错误：无法打开输出文件: " << output_file << std::endl;
        return;
    }
    
    // 转换从 Tcw 到 Twc 格式的函数
    auto convertTcwToTwc = [](const double* tcw_pose, double* twc_pose) {
        // 提取四元数和平移
        Eigen::Quaterniond q_cw(tcw_pose[3], tcw_pose[0], tcw_pose[1], tcw_pose[2]); // w, x, y, z
        Eigen::Vector3d t_cw(tcw_pose[4], tcw_pose[5], tcw_pose[6]);
        
        // 转换为 Sophus::SE3d
        Sophus::SE3d Tcw(q_cw, t_cw);
        
        // 计算逆（Twc = Tcw^(-1)）
        Sophus::SE3d Twc = Tcw.inverse();
        
        // 从 Twc 中提取四元数和平移
        Eigen::Quaterniond q_wc = Twc.unit_quaternion();
        Eigen::Vector3d t_wc = Twc.translation();
        
        // 存储为数组格式 [qx, qy, qz, qw, tx, ty, tz]
        twc_pose[0] = q_wc.x();
        twc_pose[1] = q_wc.y();
        twc_pose[2] = q_wc.z();
        twc_pose[3] = q_wc.w();
        twc_pose[4] = t_wc.x();
        twc_pose[5] = t_wc.y();
        twc_pose[6] = t_wc.z();
    };
    
    // 以 TUM 格式写入：timestamp tx ty tz qx qy qz qw（Twc 格式）
    out_file << "# timestamp tx ty tz qx qy qz qw" << std::endl;
    
    // 用于存储所有时间戳和关键帧 ID 的向量，用于排序
    std::vector<std::pair<double, int>> timestamps_kfids;
    
    // 收集所有关键帧的时间戳
    for (const auto& kf_pair : keyframe_poses) {
        int kf_id = kf_pair.first;
        double timestamp = 0.0;
        
        // 尝试从原始 keyframe_poses 文件中获取时间戳
        std::ifstream ts_file(data_dir + "keyframe_poses.txt");
        std::string ts_line;
        std::getline(ts_file, ts_line); // 跳过标题行
        
        int id;
        double ts;
        double dummy1, dummy2, dummy3, dummy4, dummy5, dummy6, dummy7;
        while (ts_file >> id >> ts >> dummy1 >> dummy2 >> dummy3 >> dummy4 >> dummy5 >> dummy6 >> dummy7) {
            if (id == kf_id) {
                timestamp = ts;
                timestamps_kfids.push_back(std::make_pair(timestamp, kf_id));
                break;
            }
        }
    }
    
    // 按时间戳排序
    std::sort(timestamps_kfids.begin(), timestamps_kfids.end());
    
    // 按时间顺序写入（对于可视化和评估很重要）
    for (const auto& ts_kf : timestamps_kfids) {
        double timestamp = ts_kf.first;
        int kf_id = ts_kf.second;
        
        if (optimized_poses.find(kf_id) != optimized_poses.end()) {
            const double* tcw_pose = optimized_poses[kf_id];
            double twc_pose[7]; // Twc 位姿的临时存储
            
            // 从 Tcw 转换为 Twc 格式
            convertTcwToTwc(tcw_pose, twc_pose);
            
            // 以 TUM 格式写入：timestamp tx ty tz qx qy qz qw
            out_file << std::fixed << std::setprecision(9) << timestamp << " "
                     << twc_pose[4] << " " << twc_pose[5] << " " << twc_pose[6] << " "
                     << twc_pose[0] << " " << twc_pose[1] << " " << twc_pose[2] << " " << twc_pose[3] << std::endl;
        }
    }
    
    out_file.close();
    
    std::cout << "优化后的位姿已保存到: " << output_file << "（以 Twc 格式用于可视化）" << std::endl;


    std::cout << "位姿图优化完成！" << std::endl;
}

int main(int argc, char** argv) {
    std::string data_dir = "/Datasets/CERES_Work/input/optimization_data/";
    
    // 检查是否提供了数据目录
    if (argc > 1) {
        data_dir = argv[1];
        // 确保目录以斜杠结尾
        if (data_dir.back() != '/') {
            data_dir += '/';
        }
    }
    
    // 对于 RGBD/双目系统，bFixScale 应为 true
    bool bFixScale = true;
    
    // 运行优化
    OptimizeEssentialGraph(data_dir, bFixScale);
    
    return 0;
}
