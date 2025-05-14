#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <unordered_map>
#include <algorithm>
#include <iomanip>  // For std::setprecision
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <sophus/se3.hpp>
#include <ceres/ceres.h>
#include <sys/stat.h>

// 定义关键帧结构
struct KeyFrame {
    unsigned long id;
    unsigned long parent_id;
    double timestamp;
    bool is_fixed;
    Sophus::SE3d pose;  // 世界坐标系到相机坐标系的变换
};

// SE3相对位姿代价函数
struct SE3RelativePoseCostFunctor {
    SE3RelativePoseCostFunctor(const Sophus::SE3d& T_ab) : T_ab_(T_ab) {}
    
    template <typename T>
    bool operator()(
        const T* const pose_a,  // 7参数：[tx,ty,tz,qx,qy,qz,qw]
        const T* const pose_b,
        T* residuals) const {
        
        // 提取位姿A（世界坐标系到A相机坐标系的变换）
        Eigen::Map<const Eigen::Matrix<T,3,1>> t_a(pose_a);
        Eigen::Map<const Eigen::Quaternion<T>> q_a(pose_a + 3);
        
        // 提取位姿B（世界坐标系到B相机坐标系的变换）
        Eigen::Map<const Eigen::Matrix<T,3,1>> t_b(pose_b);
        Eigen::Map<const Eigen::Quaternion<T>> q_b(pose_b + 3);
        
        // 计算相对位姿：从B相机坐标系到A相机坐标系的变换
        Sophus::SE3<T> T_a_w(q_a, t_a);
        Sophus::SE3<T> T_b_w(q_b, t_b);
        Sophus::SE3<T> T_a_b = T_a_w * T_b_w.inverse();
        
        // 计算与测量值的差异
        Sophus::SE3<T> T_a_b_measured(
            Eigen::Quaternion<T>(T_ab_.unit_quaternion().cast<T>()),
            T_ab_.translation().cast<T>());
        
        // 计算误差（SE3的李代数表示）
        Eigen::Matrix<T,6,1> error = (T_a_b_measured.inverse() * T_a_b).log();
        
        // 填充残差
        for (int i = 0; i < 6; ++i) {
            residuals[i] = error[i];
        }
        
        return true;
    }
    
private:
    Sophus::SE3d T_ab_; // 测量的相对位姿
};

// 处理关键帧位姿
void processKeyframePose(std::stringstream& ss, std::map<unsigned long, KeyFrame>& keyframes) {
    unsigned long kf_id;
    double timestamp, tx, ty, tz, qx, qy, qz, qw;
    
    ss >> kf_id >> timestamp >> tx >> ty >> tz >> qx >> qy >> qz >> qw;
    
    KeyFrame& kf = keyframes[kf_id];
    kf.id = kf_id;
    kf.timestamp = timestamp;
    kf.pose = Sophus::SE3d(Eigen::Quaterniond(qw, qx, qy, qz), Eigen::Vector3d(tx, ty, tz));
}

// 处理关键帧信息
void processKeyframeInfo(std::stringstream& ss, std::map<unsigned long, KeyFrame>& keyframes) {
    unsigned long kf_id, parent_id;
    int has_velocity, is_fixed, is_bad, is_inertial, is_virtual;
    
    ss >> kf_id >> parent_id >> has_velocity >> is_fixed >> is_bad >> is_inertial >> is_virtual;
    
    // 确保关键帧存在
    if (keyframes.find(kf_id) == keyframes.end()) {
        std::cerr << "Warning: keyframe " << kf_id << " not found in poses file" << std::endl;
        return;
    }
    
    KeyFrame& kf = keyframes[kf_id];
    kf.parent_id = parent_id;
    kf.is_fixed = (is_fixed == 1);
}

// 保存轨迹为TUM格式
void saveTUMTrajectory(const std::string& filename, const std::map<unsigned long, KeyFrame>& keyframes) {
    std::ofstream file(filename);
    file << "# timestamp tx ty tz qx qy qz qw" << std::endl;
    
    // 创建按时间戳排序的关键帧向量
    std::vector<std::pair<double, unsigned long>> sorted_timestamps;
    for (const auto& kf_pair : keyframes) {
        sorted_timestamps.push_back({kf_pair.second.timestamp, kf_pair.first});
    }
    
    // 按时间戳排序
    std::sort(sorted_timestamps.begin(), sorted_timestamps.end());
    
    // 保存排序后的轨迹
    for (const auto& ts_pair : sorted_timestamps) {
        const KeyFrame& kf = keyframes.at(ts_pair.second);
        Eigen::Vector3d t = kf.pose.translation();
        Eigen::Quaterniond q = kf.pose.unit_quaternion();
        
        file << std::fixed << std::setprecision(9) 
             << kf.timestamp << " "
             << t.x() << " " << t.y() << " " << t.z() << " "
             << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
    }
    
    file.close();
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <data_dir>" << std::endl;
        return 1;
    }
    
    std::string data_dir = argv[1];
    
    // 1. 读取输入文件
    
    // 1.1 读取map_info.txt
    unsigned long init_kf_id = 0;
    unsigned long max_kf_id = 0;
    
    std::ifstream map_info_file(data_dir + "/optimization_data/map_info.txt");
    if (!map_info_file.is_open()) {
        std::cerr << "Failed to open map_info.txt" << std::endl;
        return 1;
    }
    
    std::string line;
    while (std::getline(map_info_file, line)) {
        if (line.empty() || line[0] == '#') continue;
        
        std::stringstream ss(line);
        std::string key;
        ss >> key;
        
        if (key == "INIT_KF_ID") ss >> init_kf_id;
        else if (key == "MAX_KF_ID") ss >> max_kf_id;
    }
    map_info_file.close();
    
    // 1.2 读取keyframe_ids.txt
    unsigned long loop_kf_id = 0;
    unsigned long current_kf_id = 0;
    bool fixed_scale = true;
    
    std::ifstream kf_ids_file(data_dir + "/optimization_data/keyframe_ids.txt");
    if (!kf_ids_file.is_open()) {
        std::cerr << "Failed to open keyframe_ids.txt" << std::endl;
        return 1;
    }
    
    while (std::getline(kf_ids_file, line)) {
        if (line.empty() || line[0] == '#') continue;
        
        std::stringstream ss(line);
        std::string key;
        unsigned long value;
        ss >> key >> value;
        
        if (key == "LOOP_KF_ID") loop_kf_id = value;
        else if (key == "CURRENT_KF_ID") current_kf_id = value;
        else if (key == "FIXED_SCALE") fixed_scale = (value == 1);
    }
    kf_ids_file.close();
    
    // 1.3 读取loop_match.txt
    // 1.3 读取loop_match.txt
    Sophus::SE3d loop_relative_pose;  // 从当前KF到回环KF的变换
    
    std::ifstream loop_match_file(data_dir + "/optimization_data/loop_match.txt");
    if (!loop_match_file.is_open()) {
        std::cerr << "Failed to open loop_match.txt" << std::endl;
        return 1;
    }
    
    // 跳过注释行
    while (std::getline(loop_match_file, line)) {
        if (line.empty() || line[0] == '#') continue;
        break;
    }
    
    // 读取KF ID行
    std::stringstream ss(line);
    unsigned long cur_id, loop_id;
    ss >> cur_id >> loop_id;
    
    // 确保ID匹配
    if (cur_id != current_kf_id || loop_id != loop_kf_id) {
        std::cerr << "Warning: KF IDs in loop_match.txt don't match keyframe_ids.txt" << std::endl;
    }
    
    // 读取相对变换矩阵
    if (std::getline(loop_match_file, line)) {
        Eigen::Matrix4d mat;
        std::stringstream mat_ss(line);
        
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                mat_ss >> mat(i, j);
            }
        }
        
        // 提取3x3旋转矩阵
        Eigen::Matrix3d R = mat.block<3,3>(0,0);
        
        // 确保旋转矩阵正交
        Eigen::JacobiSVD<Eigen::Matrix3d> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix3d orthogonal_R = svd.matrixU() * svd.matrixV().transpose();
        
        // 确保是右手坐标系（行列式为正）
        if (orthogonal_R.determinant() < 0) {
            orthogonal_R = -orthogonal_R;
        }
        
        // 使用正交化后的旋转矩阵和原始平移向量创建SE3
        Eigen::Vector3d t = mat.block<3,1>(0,3);
        loop_relative_pose = Sophus::SE3d(orthogonal_R, t);
    }
    loop_match_file.close();
    
    // 1.4 读取keyframe_poses.txt
    std::map<unsigned long, KeyFrame> keyframes;
    
    std::ifstream poses_file(data_dir + "/optimization_data/keyframe_poses.txt");
    if (!poses_file.is_open()) {
        std::cerr << "Failed to open keyframe_poses.txt" << std::endl;
        return 1;
    }
    
    // 跳过注释行
    while (std::getline(poses_file, line)) {
        if (line.empty() || line[0] == '#') continue;
        break;
    }
    
    // 处理第一行
    std::stringstream first_ss(line);
    processKeyframePose(first_ss, keyframes);
    
    // 处理剩余行
    while (std::getline(poses_file, line)) {
        if (line.empty() || line[0] == '#') continue;
        
        std::stringstream ss(line);
        processKeyframePose(ss, keyframes);
    }
    poses_file.close();
    
    // 1.5 读取keyframes.txt，设置父关键帧和固定标志
    std::ifstream kfs_file(data_dir + "/optimization_data/keyframes.txt");
    if (!kfs_file.is_open()) {
        std::cerr << "Failed to open keyframes.txt" << std::endl;
        return 1;
    }
    
    // 跳过注释行
    while (std::getline(kfs_file, line)) {
        if (line.empty() || line[0] == '#') continue;
        break;
    }
    
    // 处理第一行
    std::stringstream first_kf_ss(line);
    processKeyframeInfo(first_kf_ss, keyframes);
    
    // 处理剩余行
    while (std::getline(kfs_file, line)) {
        if (line.empty() || line[0] == '#') continue;
        
        std::stringstream ss(line);
        processKeyframeInfo(ss, keyframes);
    }
    kfs_file.close();
    
    // 1.6 读取spanning_tree.txt
    std::map<unsigned long, unsigned long> spanning_tree;  // child -> parent
    
    std::ifstream tree_file(data_dir + "/optimization_data/spanning_tree.txt");
    if (!tree_file.is_open()) {
        std::cerr << "Failed to open spanning_tree.txt" << std::endl;
        return 1;
    }
    
    // 跳过注释行
    while (std::getline(tree_file, line)) {
        if (line.empty() || line[0] == '#') continue;
        break;
    }
    
    // 处理第一行
    {
        std::stringstream ss(line);
        unsigned long child_id, parent_id;
        ss >> child_id >> parent_id;
        spanning_tree[child_id] = parent_id;
    }
    
    // 处理剩余行
    while (std::getline(tree_file, line)) {
        if (line.empty() || line[0] == '#') continue;
        
        std::stringstream ss(line);
        unsigned long child_id, parent_id;
        ss >> child_id >> parent_id;
        spanning_tree[child_id] = parent_id;
    }
    tree_file.close();
    
    // 1.7 读取covisibility.txt
    std::map<std::pair<unsigned long, unsigned long>, int> covisibility_weights;
    
    std::ifstream covis_file(data_dir + "/optimization_data/covisibility.txt");
    if (!covis_file.is_open()) {
        std::cerr << "Failed to open covisibility.txt" << std::endl;
        return 1;
    }
    
    // 跳过注释行
    while (std::getline(covis_file, line)) {
        if (line.empty() || line[0] == '#') continue;
        break;
    }
    
    // 处理第一行
    {
        std::stringstream ss(line);
        unsigned long kf_id, conn_id;
        int weight;
        ss >> kf_id >> conn_id >> weight;
        
        std::pair<unsigned long, unsigned long> edge = 
            (kf_id < conn_id) ? std::make_pair(kf_id, conn_id) : std::make_pair(conn_id, kf_id);
        covisibility_weights[edge] = weight;
    }
    
    // 处理剩余行
    while (std::getline(covis_file, line)) {
        if (line.empty() || line[0] == '#') continue;
        
        std::stringstream ss(line);
        unsigned long kf_id, conn_id;
        int weight;
        ss >> kf_id >> conn_id >> weight;
        
        std::pair<unsigned long, unsigned long> edge = 
            (kf_id < conn_id) ? std::make_pair(kf_id, conn_id) : std::make_pair(conn_id, kf_id);
        covisibility_weights[edge] = weight;
    }
    covis_file.close();
    
    // 1.8 读取loop_connections.txt
    std::map<unsigned long, std::set<unsigned long>> loop_connections;
    
    std::ifstream loop_conn_file(data_dir + "/optimization_data/loop_connections.txt");
    if (!loop_conn_file.is_open()) {
        std::cerr << "Failed to open loop_connections.txt" << std::endl;
        return 1;
    }
    
    // 跳过注释行
    while (std::getline(loop_conn_file, line)) {
        if (line.empty() || line[0] == '#') continue;
        break;
    }
    
    // 处理第一行
    {
        std::stringstream ss(line);
        unsigned long kf_id;
        ss >> kf_id;
        
        unsigned long conn_id;
        while (ss >> conn_id) {
            loop_connections[kf_id].insert(conn_id);
        }
    }
    
    // 处理剩余行
    while (std::getline(loop_conn_file, line)) {
        if (line.empty() || line[0] == '#') continue;
        
        std::stringstream ss(line);
        unsigned long kf_id;
        ss >> kf_id;
        
        unsigned long conn_id;
        while (ss >> conn_id) {
            loop_connections[kf_id].insert(conn_id);
        }
    }
    loop_conn_file.close();
    
    // 2. 设置CERES优化问题
    
    // 2.1 备份原始关键帧位姿
    std::map<unsigned long, Sophus::SE3d> original_poses;
    for (const auto& kf_pair : keyframes) {
        original_poses[kf_pair.first] = kf_pair.second.pose;
    }
    
    // 2.2 配置求解器选项
    ceres::Problem problem;
    ceres::LossFunction* loss_function = new ceres::HuberLoss(1.0);
    
    // 2.3 设置参数块
    std::map<unsigned long, double*> pose_parameters;
    
    // 使用Ceres内置的欧氏空间和四元数流形组合 - 更适合SE3
    // 更简洁的方式，使用类型推导
    auto* product_manifold = new ceres::ProductManifold(
        ceres::EuclideanManifold<3>{}, 
        ceres::EigenQuaternionManifold{});
    
    const int min_covisibility_weight = 100;  // 最小共视权重阈值
    
    // 对每个关键帧，添加SE3位姿变量
    for (const auto& kf_pair : keyframes) {
        unsigned long kf_id = kf_pair.first;
        const KeyFrame& kf = kf_pair.second;
        
        // 每个SE3变量包含平移(3)和四元数旋转(4)
        double* param = new double[7];
        
        // 平移部分
        param[0] = kf.pose.translation()(0);
        param[1] = kf.pose.translation()(1);
        param[2] = kf.pose.translation()(2);
        
        // 旋转部分（四元数）
        Eigen::Quaterniond q = kf.pose.unit_quaternion();
        param[3] = q.x();
        param[4] = q.y();
        param[5] = q.z();
        param[6] = q.w();
        
        pose_parameters[kf_id] = param;
        
        // 添加到优化问题中 - 使用Ceres 2.2的Manifold API
        problem.AddParameterBlock(param, 7, product_manifold);
        
        // 固定初始关键帧或指定为固定的关键帧
        if (kf.is_fixed) {
            problem.SetParameterBlockConstant(param);
        }
    }
    
    // 3. 添加约束
    
    // 3.1 添加生成树约束
    for (const auto& tree_edge : spanning_tree) {
        unsigned long child_id = tree_edge.first;
        unsigned long parent_id = tree_edge.second;
        
        double* child_param = pose_parameters[child_id];
        double* parent_param = pose_parameters[parent_id];
        
        // 计算相对变换
        const KeyFrame& child_kf = keyframes[child_id];
        const KeyFrame& parent_kf = keyframes[parent_id];
        
        Sophus::SE3d T_parent_world = parent_kf.pose;
        Sophus::SE3d T_child_world = child_kf.pose;
        Sophus::SE3d T_parent_child = T_parent_world * T_child_world.inverse();
        
        // 添加相对位姿约束
        ceres::CostFunction* cost_function = 
            new ceres::AutoDiffCostFunction<SE3RelativePoseCostFunctor, 6, 7, 7>(
                new SE3RelativePoseCostFunctor(T_parent_child));
        
        problem.AddResidualBlock(
            cost_function, 
            loss_function, 
            parent_param,
            child_param);
    }
    
    // 3.2 添加共视图约束
    std::set<std::pair<unsigned long, unsigned long>> inserted_edges;
    
    for (const auto& covis_edge : covisibility_weights) {
        const std::pair<unsigned long, unsigned long>& edge = covis_edge.first;
        int weight = covis_edge.second;
        
        // 跳过低权重的共视关系
        if (weight < min_covisibility_weight) 
            continue;
        
        unsigned long id1 = edge.first;
        unsigned long id2 = edge.second;
        
        // 避免已经添加过的边（如生成树边）
        if (spanning_tree[id1] == id2 || spanning_tree[id2] == id1)
            continue;
        
        // 标记已插入边
        inserted_edges.insert(edge);
        
        double* param1 = pose_parameters[id1];
        double* param2 = pose_parameters[id2];
        
        // 计算相对变换
        const KeyFrame& kf1 = keyframes[id1];
        const KeyFrame& kf2 = keyframes[id2];
        
        Sophus::SE3d T_kf1_world = kf1.pose;
        Sophus::SE3d T_kf2_world = kf2.pose;
        Sophus::SE3d T_kf1_kf2 = T_kf1_world * T_kf2_world.inverse();
        
        // 添加相对位姿约束
        ceres::CostFunction* cost_function = 
            new ceres::AutoDiffCostFunction<SE3RelativePoseCostFunctor, 6, 7, 7>(
                new SE3RelativePoseCostFunctor(T_kf1_kf2));
        
        // 根据共视权重调整信息矩阵
        double information_scale = weight / static_cast<double>(min_covisibility_weight);
        if (information_scale > 1.0) information_scale = 1.0;
        
        ceres::LossFunction* scaled_loss = 
            new ceres::ScaledLoss(loss_function, information_scale, ceres::TAKE_OWNERSHIP);
        
        problem.AddResidualBlock(
            cost_function, 
            scaled_loss, 
            param1,
            param2);
    }
    
    // 3.3 添加回环约束
    
    // 3.3.1 首先添加来自loop_connections的约束
    for (const auto& lc_pair : loop_connections) {
        unsigned long kf_id = lc_pair.first;
        const std::set<unsigned long>& connections = lc_pair.second;
        
        double* kf_param = pose_parameters[kf_id];
        
        for (unsigned long conn_id : connections) {
            // 避免重复添加边
            std::pair<unsigned long, unsigned long> edge = 
                (kf_id < conn_id) ? std::make_pair(kf_id, conn_id) : std::make_pair(conn_id, kf_id);
                
            if (inserted_edges.count(edge))
                continue;
                
            // 标记已插入边
            inserted_edges.insert(edge);
            
            double* conn_param = pose_parameters[conn_id];
            
            // 计算相对变换
            const KeyFrame& kf = keyframes[kf_id];
            const KeyFrame& conn_kf = keyframes[conn_id];
            
            Sophus::SE3d T_kf_world = kf.pose;
            Sophus::SE3d T_conn_world = conn_kf.pose;
            Sophus::SE3d T_kf_conn = T_kf_world * T_conn_world.inverse();
            
            // 添加相对位姿约束，高权重
            ceres::CostFunction* cost_function = 
                new ceres::AutoDiffCostFunction<SE3RelativePoseCostFunctor, 6, 7, 7>(
                    new SE3RelativePoseCostFunctor(T_kf_conn));
            
            // 回环约束使用更高的权重
            ceres::LossFunction* loop_loss = 
                new ceres::ScaledLoss(loss_function, 10.0, ceres::TAKE_OWNERSHIP);
            
            problem.AddResidualBlock(
                cost_function, 
                loop_loss, 
                kf_param,
                conn_param);
        }
    }
    
    // 3.3.2 添加当前回环约束
    double* current_param = pose_parameters[current_kf_id];
    double* loop_param = pose_parameters[loop_kf_id];
    
    // 添加回环约束，这是最重要的约束，使用较高的权重
    ceres::CostFunction* loop_cost_function = 
        new ceres::AutoDiffCostFunction<SE3RelativePoseCostFunctor, 6, 7, 7>(
            new SE3RelativePoseCostFunctor(loop_relative_pose));
    
    // 回环约束使用更高的权重
    ceres::LossFunction* direct_loop_loss = 
        new ceres::ScaledLoss(loss_function, 10.0, ceres::TAKE_OWNERSHIP);
    
    problem.AddResidualBlock(
        loop_cost_function, 
        direct_loop_loss, 
        loop_param,      // 回环KF参数
        current_param);  // 当前KF参数
    
    // 4. 执行优化
    
    // 4.1 配置求解器选项
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.max_num_iterations = 20;  // 与原始ORB-SLAM3保持一致
    options.minimizer_progress_to_stdout = true;
    options.num_threads = 8;  // 多线程优化
    
    ceres::Solver::Summary summary;
    std::cout << "Starting optimization..." << std::endl;
    ceres::Solve(options, &problem, &summary);
    
    std::cout << summary.FullReport() << std::endl;
    
    // 5. 更新关键帧位姿
    
    // 5.1 获取优化后的位姿
    for (const auto& param_pair : pose_parameters) {
        unsigned long kf_id = param_pair.first;
        const double* param = param_pair.second;
        
        Eigen::Vector3d t(param[0], param[1], param[2]);
        Eigen::Quaterniond q(param[6], param[3], param[4], param[5]); // w,x,y,z
        q.normalize();
        
        // 更新关键帧位姿
        keyframes[kf_id].pose = Sophus::SE3d(q, t);
    }
    
    // 6. 保存优化结果
    
    std::string output_dir = data_dir + "/../output";
    
    // 创建输出目录
    struct stat st = {0};
    if (stat(output_dir.c_str(), &st) == -1) {
#ifdef _WIN32
        _mkdir(output_dir.c_str());
#else
        mkdir(output_dir.c_str(), 0755);
#endif
    }
    
    // 保存TUM格式轨迹
    saveTUMTrajectory(output_dir + "/trajectory.txt", keyframes);
    
    std::cout << "Optimization completed successfully!" << std::endl;
    std::cout << "Results saved to: " << output_dir << "/trajectory.txt (TUM format)" << std::endl;
    
    // 释放内存
    for (auto& param_pair : pose_parameters) {
        delete[] param_pair.second;
    }
    
    return 0;
}
