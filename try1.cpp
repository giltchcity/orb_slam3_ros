#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <set>
#include <queue>
#include <string>
#include <sstream>
#include <algorithm>
#include <iomanip>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/SVD>

// 从TUM格式行解析时间戳和位姿
bool parseTUMPoseLine(const std::string& line, double& timestamp, Eigen::Vector3d& translation, Eigen::Quaterniond& rotation) {
    std::istringstream iss(line);
    double tx, ty, tz, qx, qy, qz, qw;
    
    // 跳过注释行
    if (line.empty() || line[0] == '#') return false;
    
    // 读取时间戳和位姿
    if (!(iss >> timestamp >> tx >> ty >> tz >> qx >> qy >> qz >> qw)) return false;
    
    translation = Eigen::Vector3d(tx, ty, tz);
    rotation = Eigen::Quaterniond(qw, qx, qy, qz); // 注意Eigen的四元数构造顺序是(w,x,y,z)
    rotation.normalize(); // 确保四元数单位化
    
    return true;
}

// 实现优化轨迹，保持前段和后段良好对齐，中间平滑过渡
void PropagateAndRotateBack(const std::string& original_file, const std::string& partially_corrected_file, const std::string& output_file) {
    std::cout << "开始轨迹优化，保持前段和后段良好对齐，中间平滑过渡..." << std::endl;
    
    // 1. 读取原始轨迹
    std::map<double, std::pair<Eigen::Vector3d, Eigen::Quaterniond>> original_poses;
    std::map<int, double> kf_to_timestamp; // KF ID到时间戳的映射
    std::map<double, int> timestamp_to_kf; // 时间戳到KF ID的映射
    
    std::ifstream original_file_stream(original_file);
    if (!original_file_stream.is_open()) {
        std::cerr << "错误：无法打开原始轨迹文件: " << original_file << std::endl;
        return;
    }
    
    std::string line;
    double timestamp;
    Eigen::Vector3d translation;
    Eigen::Quaterniond rotation;
    
    int kf_count = 0;
    while (std::getline(original_file_stream, line)) {
        if (parseTUMPoseLine(line, timestamp, translation, rotation)) {
            original_poses[timestamp] = std::make_pair(translation, rotation);
            kf_to_timestamp[kf_count] = timestamp;
            timestamp_to_kf[timestamp] = kf_count;
            kf_count++;
        }
    }
    
    std::cout << "读取了 " << original_poses.size() << " 个关键帧的原始位姿" << std::endl;
    
    // 2. 读取部分校正轨迹，并区分已校正和未校正
    std::map<double, std::pair<Eigen::Vector3d, Eigen::Quaterniond>> corrected_poses;
    std::map<double, std::pair<Eigen::Vector3d, Eigen::Quaterniond>> uncorrected_poses;
    
    std::ifstream partially_corrected_file_stream(partially_corrected_file);
    if (!partially_corrected_file_stream.is_open()) {
        std::cerr << "错误：无法打开部分校正轨迹文件: " << partially_corrected_file << std::endl;
        return;
    }
    
    // 用于区分已校正和未校正帧的时间戳
    double boundary_timestamp = 1317384588915486208.000000000; // 对应KF441
    
    // 回环关键帧ID
    const int loop_kf_id = 2;     // 回环关键帧
    const int current_kf_id = 464; // 当前关键帧
    
    // 获取回环关键帧和当前关键帧时间戳
    double loop_timestamp = 0;
    double current_timestamp = 0;
    
    if (kf_to_timestamp.find(loop_kf_id) != kf_to_timestamp.end()) {
        loop_timestamp = kf_to_timestamp[loop_kf_id];
        std::cout << "回环关键帧 " << loop_kf_id << " 对应时间戳: " << std::fixed << loop_timestamp << std::endl;
    } else {
        std::cerr << "警告: 找不到回环关键帧 " << loop_kf_id << " 的时间戳" << std::endl;
    }
    
    if (kf_to_timestamp.find(current_kf_id) != kf_to_timestamp.end()) {
        current_timestamp = kf_to_timestamp[current_kf_id];
        std::cout << "当前关键帧 " << current_kf_id << " 对应时间戳: " << std::fixed << current_timestamp << std::endl;
    } else {
        std::cerr << "警告: 找不到当前关键帧 " << current_kf_id << " 的时间戳" << std::endl;
    }
    
    while (std::getline(partially_corrected_file_stream, line)) {
        if (parseTUMPoseLine(line, timestamp, translation, rotation)) {
            if (timestamp >= boundary_timestamp) { // 大于等于边界时间戳的是校正过的
                corrected_poses[timestamp] = std::make_pair(translation, rotation);
            } else { // 小于边界时间戳的是未校正的
                uncorrected_poses[timestamp] = std::make_pair(translation, rotation);
            }
        }
    }
    
    std::cout << "读取了 " << corrected_poses.size() << " 个校正后的关键帧位姿" << std::endl;
    std::cout << "读取了 " << uncorrected_poses.size() << " 个未校正的关键帧位姿" << std::endl;
    
    // 3. 构建关键帧连接关系 - 按时间戳顺序
    std::vector<double> timestamps;
    for (const auto& pose_pair : original_poses) {
        timestamps.push_back(pose_pair.first);
    }
    
    // 如果原始文件为空，则使用部分校正文件中的所有时间戳
    if (timestamps.empty()) {
        for (const auto& pose_pair : uncorrected_poses) {
            timestamps.push_back(pose_pair.first);
        }
        for (const auto& pose_pair : corrected_poses) {
            timestamps.push_back(pose_pair.first);
        }
    }
    
    // 按时间戳排序
    std::sort(timestamps.begin(), timestamps.end());
    
    // 构建前后关系
    std::map<double, double> next_frame; // 当前时间戳 -> 下一帧时间戳
    std::map<double, double> prev_frame; // 当前时间戳 -> 上一帧时间戳
    
    for (size_t i = 0; i < timestamps.size() - 1; ++i) {
        next_frame[timestamps[i]] = timestamps[i + 1];
        prev_frame[timestamps[i + 1]] = timestamps[i];
    }
    
    // 4. 创建最终的位姿传播结果
    std::map<double, std::pair<Eigen::Vector3d, Eigen::Quaterniond>> propagated_poses;
    
    // 先复制所有校正过的位姿
    for (const auto& corrected_pair : corrected_poses) {
        propagated_poses[corrected_pair.first] = corrected_pair.second;
    }
    
    // 定义前段保留的关键帧百分比（前15%的关键帧保留原始位姿）
    const double front_preserve_percentage = 0.15;
    const int front_preserve_count = static_cast<int>(timestamps.size() * front_preserve_percentage);
    
    std::cout << "将保留前 " << front_preserve_count << " 个关键帧的原始位姿 (约" 
              << front_preserve_percentage * 100 << "%)" << std::endl;
    
    // 添加前段保留的关键帧到传播位姿中
    for (int i = 0; i < front_preserve_count; ++i) {
        if (i < timestamps.size()) {
            double ts = timestamps[i];
            // 使用原始轨迹中的位姿
            if (original_poses.find(ts) != original_poses.end()) {
                propagated_poses[ts] = original_poses[ts];
                if (i % 10 == 0 || i == front_preserve_count - 1) {
                    std::cout << "保留KF " << i << " 的原始位姿 (时间戳: " << std::fixed << ts << ")" << std::endl;
                }
            }
        }
    }
    
    // 5. 创建两个传播队列，一个从前段向后，一个从后段向前
    std::queue<double> front_queue; // 从前段保留区域边界向后传播
    std::queue<double> back_queue;  // 从后段已校正区域向前传播
    std::set<double> front_visited, back_visited;
    
    // 前段保留区域的最后一帧的下一帧作为前向传播起点
    if (front_preserve_count > 0 && front_preserve_count < timestamps.size()) {
        double front_boundary_ts = timestamps[front_preserve_count - 1];
        if (next_frame.find(front_boundary_ts) != next_frame.end()) {
            double start_ts = next_frame[front_boundary_ts];
            front_queue.push(start_ts);
            front_visited.insert(start_ts);
            std::cout << "前向传播起点: KF " << front_preserve_count 
                    << " (时间戳: " << std::fixed << start_ts << ")" << std::endl;
        }
    }
    
    // 后段已校正区域的第一帧的前一帧作为后向传播起点
    if (!corrected_poses.empty()) {
        double back_boundary_ts = corrected_poses.begin()->first;
        if (prev_frame.find(back_boundary_ts) != prev_frame.end()) {
            double start_ts = prev_frame[back_boundary_ts];
            back_queue.push(start_ts);
            back_visited.insert(start_ts);
            std::cout << "后向传播起点: 时间戳 " << std::fixed << start_ts << std::endl;
        }
    }
    
    // 存储前向和后向传播的结果
    std::map<double, std::pair<Eigen::Vector3d, Eigen::Quaterniond>> front_propagated;
    std::map<double, std::pair<Eigen::Vector3d, Eigen::Quaterniond>> back_propagated;
    
    // 存储每个关键帧距离前后边界的距离
    std::map<double, int> distance_from_front;
    std::map<double, int> distance_from_back;
    
    // 辅助函数：从源轨迹获取位姿
    auto getPoseFromSource = [&](double ts) -> std::pair<Eigen::Vector3d, Eigen::Quaterniond> {
        // 首先尝试从原始轨迹获取
        if (original_poses.find(ts) != original_poses.end()) {
            return original_poses[ts];
        }
        // 其次尝试从未校正轨迹获取
        else if (uncorrected_poses.find(ts) != uncorrected_poses.end()) {
            return uncorrected_poses[ts];
        }
        // 最后从校正轨迹获取
        else if (corrected_poses.find(ts) != corrected_poses.end()) {
            return corrected_poses[ts];
        }
        // 如果都找不到，返回单位位姿
        else {
            std::cerr << "警告：时间戳 " << std::fixed << ts << " 的位姿未找到" << std::endl;
            return std::make_pair(Eigen::Vector3d::Zero(), Eigen::Quaterniond::Identity());
        }
    };
    
    // 定义衰减系数计算函数
    auto calculateDecayFactor = [](int distance) -> double {
        if (distance <= 50) {
            return std::pow(0.97, distance);
        } else if (distance <= 100) {
            return std::pow(0.97, 50) * std::pow(0.96, distance - 50);
        } else {
            return std::pow(0.97, 50) * std::pow(0.96, 50) * std::pow(0.95, distance - 100);
        }
    };
    
    // 6. 前向传播 - 从前段保留区域向后传播
    std::cout << "开始前向传播..." << std::endl;
    
    if (!front_queue.empty()) {
        distance_from_front[front_queue.front()] = 1;
    }
    
    while (!front_queue.empty()) {
        double current_ts = front_queue.front();
        front_queue.pop();
        
        // 跳过已经处理的关键帧
        if (front_propagated.find(current_ts) != front_propagated.end()) {
            continue;
        }
        
        // 获取前一帧的时间戳
        double prev_ts;
        if (prev_frame.find(current_ts) != prev_frame.end()) {
            prev_ts = prev_frame[current_ts];
        } else {
            continue;
        }
        
        // 如果前一帧已经有传播位姿
        if (propagated_poses.find(prev_ts) != propagated_poses.end() || 
            front_propagated.find(prev_ts) != front_propagated.end()) {
            
            // 获取原始轨迹中的相对位姿
            auto orig_prev_pose = getPoseFromSource(prev_ts);
            auto orig_current_pose = getPoseFromSource(current_ts);
            
            Eigen::Vector3d orig_prev_t = orig_prev_pose.first;
            Eigen::Quaterniond orig_prev_q = orig_prev_pose.second;
            Eigen::Matrix3d orig_prev_R = orig_prev_q.toRotationMatrix();
            
            Eigen::Vector3d orig_current_t = orig_current_pose.first;
            Eigen::Quaterniond orig_current_q = orig_current_pose.second;
            Eigen::Matrix3d orig_current_R = orig_current_q.toRotationMatrix();
            
            // 计算相对变换 (从前一帧到当前帧)
            Eigen::Matrix3d R_prev_to_current = orig_prev_R.transpose() * orig_current_R;
            Eigen::Vector3d t_prev_to_current = orig_prev_R.transpose() * (orig_current_t - orig_prev_t);
            
            // 获取前一帧的传播位姿
            Eigen::Vector3d prev_prop_t;
            Eigen::Quaterniond prev_prop_q;
            
            if (front_propagated.find(prev_ts) != front_propagated.end()) {
                prev_prop_t = front_propagated[prev_ts].first;
                prev_prop_q = front_propagated[prev_ts].second;
            } else {
                prev_prop_t = propagated_poses[prev_ts].first;
                prev_prop_q = propagated_poses[prev_ts].second;
            }
            
            Eigen::Matrix3d prev_prop_R = prev_prop_q.toRotationMatrix();
            
            // 应用相对变换得到当前帧的传播位姿
            Eigen::Matrix3d current_prop_R = prev_prop_R * R_prev_to_current;
            Eigen::Vector3d current_prop_t = prev_prop_t + prev_prop_R * t_prev_to_current;
            
            Eigen::Quaterniond current_prop_q(current_prop_R);
            current_prop_q.normalize();
            
            // 存储当前帧的前向传播位姿
            front_propagated[current_ts] = std::make_pair(current_prop_t, current_prop_q);
            
            // 计算距离
            int current_distance = distance_from_front[prev_ts] + 1;
            distance_from_front[current_ts] = current_distance;
            
            // 如果有下一帧且未访问过，加入队列
            if (next_frame.find(current_ts) != next_frame.end()) {
                double next_ts = next_frame[current_ts];
                if (front_visited.find(next_ts) == front_visited.end() && 
                    corrected_poses.find(next_ts) == corrected_poses.end()) {
                    front_queue.push(next_ts);
                    front_visited.insert(next_ts);
                }
            }
            
            if (current_distance % 50 == 0) {
                std::cout << "前向传播: KF距离 " << current_distance 
                          << ", 时间戳: " << std::fixed << current_ts << std::endl;
            }
        }
    }
    
    std::cout << "完成前向传播，处理了 " << front_propagated.size() << " 个关键帧" << std::endl;
    
    // 7. 后向传播 - 从后段已校正区域向前传播
    std::cout << "开始后向传播..." << std::endl;
    
    if (!back_queue.empty()) {
        distance_from_back[back_queue.front()] = 1;
    }
    
    while (!back_queue.empty()) {
        double current_ts = back_queue.front();
        back_queue.pop();
        
        // 跳过已经处理的关键帧
        if (back_propagated.find(current_ts) != back_propagated.end() || 
            corrected_poses.find(current_ts) != corrected_poses.end()) {
            continue;
        }
        
        // 获取下一帧的时间戳
        double next_ts;
        if (next_frame.find(current_ts) != next_frame.end()) {
            next_ts = next_frame[current_ts];
        } else {
            continue;
        }
        
        // 如果下一帧已经有传播位姿或是校正位姿
        if (propagated_poses.find(next_ts) != propagated_poses.end() || 
            back_propagated.find(next_ts) != back_propagated.end()) {
            
            // 获取原始轨迹中的相对位姿
            auto orig_current_pose = getPoseFromSource(current_ts);
            auto orig_next_pose = getPoseFromSource(next_ts);
            
            Eigen::Vector3d orig_current_t = orig_current_pose.first;
            Eigen::Quaterniond orig_current_q = orig_current_pose.second;
            Eigen::Matrix3d orig_current_R = orig_current_q.toRotationMatrix();
            
            Eigen::Vector3d orig_next_t = orig_next_pose.first;
            Eigen::Quaterniond orig_next_q = orig_next_pose.second;
            Eigen::Matrix3d orig_next_R = orig_next_q.toRotationMatrix();
            
            // 计算相对变换 (从当前帧到下一帧)
            Eigen::Matrix3d R_current_to_next = orig_current_R.transpose() * orig_next_R;
            Eigen::Vector3d t_current_to_next = orig_current_R.transpose() * (orig_next_t - orig_current_t);
            
            // 获取下一帧的位姿
            Eigen::Vector3d next_prop_t;
            Eigen::Quaterniond next_prop_q;
            
            if (back_propagated.find(next_ts) != back_propagated.end()) {
                next_prop_t = back_propagated[next_ts].first;
                next_prop_q = back_propagated[next_ts].second;
            } else {
                next_prop_t = propagated_poses[next_ts].first;
                next_prop_q = propagated_poses[next_ts].second;
            }
            
            Eigen::Matrix3d next_prop_R = next_prop_q.toRotationMatrix();
            
            // 计算相对变换的逆
            Eigen::Matrix3d R_next_to_current = R_current_to_next.transpose();
            Eigen::Vector3d t_next_to_current = -R_next_to_current * t_current_to_next;
            
            // 应用相对变换的逆得到当前帧的传播位姿
            Eigen::Matrix3d current_prop_R = next_prop_R * R_next_to_current;
            Eigen::Vector3d current_prop_t = next_prop_t + next_prop_R * t_next_to_current;
            
            Eigen::Quaterniond current_prop_q(current_prop_R);
            current_prop_q.normalize();
            
            // 存储当前帧的后向传播位姿
            back_propagated[current_ts] = std::make_pair(current_prop_t, current_prop_q);
            
            // 计算距离
            int current_distance = distance_from_back[next_ts] + 1;
            distance_from_back[current_ts] = current_distance;
            
            // 如果有前一帧且未访问过，加入队列
            if (prev_frame.find(current_ts) != prev_frame.end()) {
                double prev_ts = prev_frame[current_ts];
                if (back_visited.find(prev_ts) == back_visited.end() && 
                    propagated_poses.find(prev_ts) == propagated_poses.end()) {
                    back_queue.push(prev_ts);
                    back_visited.insert(prev_ts);
                }
            }
            
            if (current_distance % 50 == 0) {
                std::cout << "后向传播: KF距离 " << current_distance 
                          << ", 时间戳: " << std::fixed << current_ts << std::endl;
            }
        }
    }
    
    std::cout << "完成后向传播，处理了 " << back_propagated.size() << " 个关键帧" << std::endl;
    
    // 8. 融合前向和后向传播结果
    std::cout << "融合前向和后向传播结果..." << std::endl;
    
    for (double ts : timestamps) {
        // 如果已经在propagated_poses中，则跳过（保留原始前段和校正后段）
        if (propagated_poses.find(ts) != propagated_poses.end()) {
            continue;
        }
        
        bool has_front = (front_propagated.find(ts) != front_propagated.end());
        bool has_back = (back_propagated.find(ts) != back_propagated.end());
        
        if (has_front && has_back) {
            // 有两个方向的传播结果，需要融合
            int front_dist = distance_from_front[ts];
            int back_dist = distance_from_back[ts];
            
            // 计算融合权重 - 距离越近权重越大
            double total_dist = front_dist + back_dist;
            double front_weight = back_dist / total_dist;  // 前向传播的权重
            
            // 获取两个方向的位姿
            Eigen::Vector3d front_t = front_propagated[ts].first;
            Eigen::Quaterniond front_q = front_propagated[ts].second;
            
            Eigen::Vector3d back_t = back_propagated[ts].first;
            Eigen::Quaterniond back_q = back_propagated[ts].second;
            
            // 融合平移
            Eigen::Vector3d blended_t = front_t * front_weight + back_t * (1.0 - front_weight);
            
            // 融合旋转（球面线性插值）
            Eigen::Quaterniond blended_q = front_q.slerp(1.0 - front_weight, back_q);
            blended_q.normalize();
            
            // 存储融合后的位姿
            propagated_poses[ts] = std::make_pair(blended_t, blended_q);
            
            if ((front_dist + back_dist) % 100 == 0) {
                std::cout << "融合: 时间戳 " << std::fixed << ts 
                          << ", 前向距离: " << front_dist 
                          << ", 后向距离: " << back_dist 
                          << ", 前向权重: " << front_weight << std::endl;
            }
        } else if (has_front) {
            // 只有前向传播结果
            propagated_poses[ts] = front_propagated[ts];
        } else if (has_back) {
            // 只有后向传播结果
            propagated_poses[ts] = back_propagated[ts];
        } else {
            // 两个方向都没有结果，使用原始位姿
            propagated_poses[ts] = getPoseFromSource(ts);
        }
    }
    
    // 9. 计算前段关键帧的回归变换 - 用于整体调整
    std::cout << "计算前段关键帧的回归变换..." << std::endl;
    
    // 选取前20个关键帧进行变换匹配
    const int front_kf_count = 20;
    std::vector<Eigen::Vector3d> orig_positions;
    std::vector<Eigen::Vector3d> propagated_positions;
    
    for (int i = 0; i < std::min(front_kf_count, (int)timestamps.size()); i++) {
        double ts = timestamps[i];
        
        if (propagated_poses.find(ts) != propagated_poses.end() && 
            original_poses.find(ts) != original_poses.end()) {
            
            orig_positions.push_back(original_poses[ts].first);
            propagated_positions.push_back(propagated_poses[ts].first);
        }
    }
    
    // 至少需要3个点才能计算一个有意义的变换
    Eigen::Matrix3d front_rotation = Eigen::Matrix3d::Identity();
    Eigen::Vector3d front_translation = Eigen::Vector3d::Zero();
    bool has_front_transform = false;
    
    if (orig_positions.size() >= 3 && propagated_positions.size() >= 3) {
        // 计算质心
        Eigen::Vector3d orig_centroid = Eigen::Vector3d::Zero();
        Eigen::Vector3d prop_centroid = Eigen::Vector3d::Zero();
        
        for (size_t i = 0; i < orig_positions.size(); i++) {
            orig_centroid += orig_positions[i];
            prop_centroid += propagated_positions[i];
        }
        
        orig_centroid /= orig_positions.size();
        prop_centroid /= propagated_positions.size();
        
        // 去质心化
        std::vector<Eigen::Vector3d> orig_centered, prop_centered;
        for (size_t i = 0; i < orig_positions.size(); i++) {
            orig_centered.push_back(orig_positions[i] - orig_centroid);
            prop_centered.push_back(propagated_positions[i] - prop_centroid);
        }
        
        // 构建协方差矩阵
        Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
        for (size_t i = 0; i < orig_centered.size(); i++) {
            H += prop_centered[i] * orig_centered[i].transpose();
        }
        
        // SVD分解
        Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix3d U = svd.matrixU();
        Eigen::Matrix3d V = svd.matrixV();
        
        // 计算旋转矩阵
        front_rotation = V * U.transpose();
        
        // 确保是正交矩阵且行列式为1（确保是旋转矩阵，而非反射矩阵）
        if (front_rotation.determinant() < 0) {
            V.col(2) = -V.col(2);
            front_rotation = V * U.transpose();
        }
        
        // 计算平移向量
        front_translation = orig_centroid - front_rotation * prop_centroid;
        
        has_front_transform = true;
        std::cout << "成功计算前段关键帧回归变换" << std::endl;
    } else {
        std::cout << "前段关键帧数量不足，无法计算有效的回归变换" << std::endl;
    }
    
    // 10. 应用前段回归变换，进行整体调整
    if (has_front_transform) {
        std::cout << "应用前段回归变换，进行整体调整..." << std::endl;
        
        std::map<double, std::pair<Eigen::Vector3d, Eigen::Quaterniond>> adjusted_poses;
        
        for (double ts : timestamps) {
            if (propagated_poses.find(ts) == propagated_poses.end()) continue;
            
            auto& prop_pose = propagated_poses[ts];
            Eigen::Vector3d prop_t = prop_pose.first;
            Eigen::Quaterniond prop_q = prop_pose.second;
            Eigen::Matrix3d prop_R = prop_q.toRotationMatrix();
            
            // 确定KF ID
            int kf_id = -1;
            if (timestamp_to_kf.find(ts) != timestamp_to_kf.end()) {
                kf_id = timestamp_to_kf[ts];
            } else {
                // 如果找不到对应的KF ID，则根据在时间序列中的位置估计
                for (int i = 0; i < timestamps.size(); i++) {
                    if (std::abs(timestamps[i] - ts) < 1e-6) {
                        kf_id = i;
                        break;
                    }
                }
            }
            
            // 根据KF ID计算权重 - 使用更平滑的权重衰减
            double weight = 0.0;
            
            if (kf_id >= 0) {
                // 调整为更平滑的衰减
                const int total_kfs = timestamps.size();
                const double decay_rate = 5.0; // 调整这个值可以改变衰减速度
                
                if (kf_id < front_preserve_count) {
                    // 前段保留区域使用较高权重
                    weight = 0.95;
                } else {
                    // 之后的KF使用平滑衰减
                    double normalized_pos = static_cast<double>(kf_id - front_preserve_count) / 
                                           (total_kfs - front_preserve_count);
                    weight = 0.95 * std::exp(-decay_rate * normalized_pos);
                }
                
                // 确保权重在有效范围内
                weight = std::max(0.0, std::min(1.0, weight));
            }
            
            // 特殊处理回环关键帧，确保满足回环约束
            if (ts == loop_timestamp) {
                weight = 0.25; // 回环关键帧需要保持一定的原始位置，但也需要一些调整
            } else if (ts >= boundary_timestamp) {
                // 对于已校正的关键帧，使用较小的调整权重
                weight = std::min(weight, 0.2);
            }
            
            // 应用旋转和平移变换
            Eigen::Vector3d new_t;
            Eigen::Matrix3d new_R;
            
            if (weight > 0) {
                // 应用权重化的回归变换
                Eigen::Matrix3d weighted_rotation = Eigen::Matrix3d::Identity() * (1.0 - weight) + 
                                                   front_rotation * weight;
                
                // 使用SVD确保旋转矩阵的正交性
                Eigen::JacobiSVD<Eigen::Matrix3d> svd(weighted_rotation, Eigen::ComputeFullU | Eigen::ComputeFullV);
                Eigen::Matrix3d orthogonal_rotation = svd.matrixU() * svd.matrixV().transpose();
                
                // 应用旋转
                new_R = orthogonal_rotation * prop_R;
                
                // 应用平移
                new_t = orthogonal_rotation * prop_t + front_translation * weight;
            } else {
                // 保持原位姿不变
                new_t = prop_t;
                new_R = prop_R;
            }
            
            // 创建最终位姿
            Eigen::Quaterniond new_q(new_R);
            new_q.normalize();
            
            // 更新位姿
            adjusted_poses[ts] = std::make_pair(new_t, new_q);
            
            if (kf_id >= 0 && (kf_id % 50 == 0 || ts == loop_timestamp || kf_id == front_preserve_count)) {
                std::cout << "KF " << kf_id << " 应用了 " << weight * 100 << "% 的回归变换" << std::endl;
            }
        }
        
        // 更新位姿
        propagated_poses = adjusted_poses;
    }
    
    // 11. 确保回环约束满足
    if (loop_timestamp > 0 && current_timestamp > 0 && 
        propagated_poses.find(loop_timestamp) != propagated_poses.end() &&
        propagated_poses.find(current_timestamp) != propagated_poses.end()) {
        
        std::cout << "检查并调整回环约束..." << std::endl;
        
        // 获取原始轨迹中的回环和当前关键帧位姿
        auto orig_loop_pose = getPoseFromSource(loop_timestamp);
        auto orig_current_pose = getPoseFromSource(current_timestamp);
        
        Eigen::Matrix3d orig_loop_R = orig_loop_pose.second.toRotationMatrix();
        Eigen::Vector3d orig_loop_t = orig_loop_pose.first;
        
        Eigen::Matrix3d orig_current_R = orig_current_pose.second.toRotationMatrix();
        Eigen::Vector3d orig_current_t = orig_current_pose.first;
        
        // 计算原始回环变换
        Eigen::Matrix3d orig_loop_to_current_R = orig_loop_R.transpose() * orig_current_R;
        Eigen::Vector3d orig_loop_to_current_t = orig_loop_R.transpose() * (orig_current_t - orig_loop_t);
        
        // 获取当前的回环和当前关键帧位姿
        auto& loop_pose = propagated_poses[loop_timestamp];
        auto& current_pose = propagated_poses[current_timestamp];
        
        Eigen::Matrix3d loop_R = loop_pose.second.toRotationMatrix();
        Eigen::Vector3d loop_t = loop_pose.first;
        
        Eigen::Matrix3d current_R = current_pose.second.toRotationMatrix();
        Eigen::Vector3d current_t = current_pose.first;
        
        // 计算当前回环变换
        Eigen::Matrix3d current_loop_to_current_R = loop_R.transpose() * current_R;
        Eigen::Vector3d current_loop_to_current_t = loop_R.transpose() * (current_t - loop_t);
        
        // 计算回环误差
        double rot_error = (current_loop_to_current_R - orig_loop_to_current_R).norm();
        double trans_error = (current_loop_to_current_t - orig_loop_to_current_t).norm();
        
        std::cout << "回环约束误差: 旋转 = " << rot_error << ", 平移 = " << trans_error << std::endl;
        
        // 如果误差较大，调整当前关键帧以满足回环约束
        if (rot_error > 0.1 || trans_error > 0.5) {
            // 使用回环关键帧和原始相对变换计算当前关键帧的正确位姿
            Eigen::Matrix3d corrected_current_R = loop_R * orig_loop_to_current_R;
            Eigen::Vector3d corrected_current_t = loop_t + loop_R * orig_loop_to_current_t;
            
            // 混合原有位姿和校正位姿
            double blend_factor = 0.7; // 70%的校正，30%的原有位姿
            
            Eigen::Vector3d blended_t = current_t * (1.0 - blend_factor) + corrected_current_t * blend_factor;
            
            // 使用球面线性插值混合旋转
            Eigen::Quaterniond current_q = current_pose.second;
            Eigen::Quaterniond corrected_q(corrected_current_R);
            Eigen::Quaterniond blended_q = current_q.slerp(blend_factor, corrected_q);
            blended_q.normalize();
            
            // 更新当前关键帧位姿
            propagated_poses[current_timestamp] = std::make_pair(blended_t, blended_q);
            
            std::cout << "调整了当前关键帧位姿以满足回环约束" << std::endl;
        }
    }
    
    // 12. 对整个轨迹应用平滑处理
    std::cout << "应用轨迹平滑处理..." << std::endl;
    
    std::map<double, std::pair<Eigen::Vector3d, Eigen::Quaterniond>> smoothed_poses = propagated_poses;
    
    // 平滑窗口大小
    const int window_size = 5;
    const int half_window = window_size / 2;
    
    // 对每个关键帧应用平滑
    for (size_t i = half_window; i < timestamps.size() - half_window; ++i) {
        double current_ts = timestamps[i];
        
        // 跳过前段保留区域和后段已校正区域
        if (i < front_preserve_count || 
            (corrected_poses.find(current_ts) != corrected_poses.end())) {
            continue;
        }
        
        // 收集窗口内的位姿
        std::vector<Eigen::Vector3d> window_translations;
        std::vector<Eigen::Quaterniond> window_rotations;
        std::vector<double> weights;
        
        for (int j = -half_window; j <= half_window; ++j) {
            size_t idx = i + j;
            if (idx >= 0 && idx < timestamps.size()) {
                double ts = timestamps[idx];
                if (propagated_poses.find(ts) != propagated_poses.end()) {
                    window_translations.push_back(propagated_poses[ts].first);
                    window_rotations.push_back(propagated_poses[ts].second);
                    
                    // 中心权重最大，向两侧递减
                    double weight = 1.0 - std::abs(j) / (half_window + 1.0);
                    weights.push_back(weight);
                }
            }
        }
        
        // 如果窗口中有足够的位姿进行平滑
        if (window_translations.size() >= 3) {
            // 计算加权平均平移
            Eigen::Vector3d avg_t = Eigen::Vector3d::Zero();
            double total_weight = 0.0;
            
            for (size_t j = 0; j < window_translations.size(); ++j) {
                avg_t += window_translations[j] * weights[j];
                total_weight += weights[j];
            }
            
            if (total_weight > 0) {
                avg_t /= total_weight;
            }
            
            // 平滑系数 - 控制平滑程度
            double smooth_factor = 0.5;
            
            // 混合原始位置和平滑位置
            Eigen::Vector3d orig_t = propagated_poses[current_ts].first;
            Eigen::Vector3d smoothed_t = orig_t * (1.0 - smooth_factor) + avg_t * smooth_factor;
            
            // 对旋转进行平滑
            // 使用加权平均的四元数
            Eigen::Quaterniond orig_q = propagated_poses[current_ts].second;
            
            // 四元数加权平均
            Eigen::Vector4d avg_q_vec = Eigen::Vector4d::Zero();
            for (size_t j = 0; j < window_rotations.size(); ++j) {
                // 确保所有四元数在同一半球
                if (orig_q.dot(window_rotations[j]) < 0) {
                    window_rotations[j].coeffs() = -window_rotations[j].coeffs();
                }
                
                avg_q_vec[0] += window_rotations[j].w() * weights[j];
                avg_q_vec[1] += window_rotations[j].x() * weights[j];
                avg_q_vec[2] += window_rotations[j].y() * weights[j];
                avg_q_vec[3] += window_rotations[j].z() * weights[j];
            }
            
            // 归一化
            double q_norm = avg_q_vec.norm();
            if (q_norm > 1e-6) {
                avg_q_vec /= q_norm;
            }
            
            Eigen::Quaterniond avg_q(avg_q_vec[0], avg_q_vec[1], avg_q_vec[2], avg_q_vec[3]);
            
            // 使用球面线性插值混合原始和平滑旋转
            Eigen::Quaterniond smoothed_q = orig_q.slerp(smooth_factor, avg_q);
            smoothed_q.normalize();
            
            // 更新平滑后的位姿
            smoothed_poses[current_ts] = std::make_pair(smoothed_t, smoothed_q);
        }
    }
    
    // 更新位姿
    propagated_poses = smoothed_poses;
    
    // 13. 将优化后的位姿写入文件
    std::ofstream out_file(output_file);
    if (!out_file.is_open()) {
        std::cerr << "错误：无法打开输出文件: " << output_file << std::endl;
        return;
    }
    
    // 按时间戳排序写入
    for (double ts : timestamps) {
        if (propagated_poses.find(ts) != propagated_poses.end()) {
            const Eigen::Vector3d& t = propagated_poses[ts].first;
            const Eigen::Quaterniond& q = propagated_poses[ts].second;
            
            out_file << std::fixed << std::setprecision(9)
                     << ts << " "
                     << t.x() << " " << t.y() << " " << t.z() << " "
                     << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
        }
    }
    
    out_file.close();
    
    std::cout << "优化后的位姿已保存到: " << output_file << std::endl;
    std::cout << "轨迹优化完成！" << std::endl;
}

int main(int argc, char** argv) {
    std::string original_file = "/Datasets/CERES_Work/input/pre/standard_trajectory_no_loop.txt";
    std::string partially_corrected_file = "/Datasets/CERES_Work/input/transformed/standard_trajectory_sim3_transformed.txt";
    std::string output_file = "/Datasets/CERES_Work/output/optimized_poses.txt";
    
    // 检查是否提供了自定义文件路径
    if (argc > 1) original_file = argv[1];
    if (argc > 2) partially_corrected_file = argv[2];
    if (argc > 3) output_file = argv[3];
    
    // 创建输出目录（如果不存在）
    std::string output_dir = output_file.substr(0, output_file.find_last_of('/'));
    system(("mkdir -p " + output_dir).c_str());
    
    // 运行位姿传播算法
    PropagateAndRotateBack(original_file, partially_corrected_file, output_file);
    
    return 0;
}
