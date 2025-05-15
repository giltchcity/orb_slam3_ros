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

// 函数为计算关键帧到Sim3矫正区域的距离
int calculateDistanceToSim3Region(double ts, double boundary_timestamp, 
                                 const std::map<double, int>& timestamp_to_kf,
                                 const std::vector<double>& timestamps) {
    // 如果关键帧已经在Sim3区域内，距离为0
    if (ts >= boundary_timestamp) return 0;
    
    // 找到当前关键帧的ID
    int current_kf_id = -1;
    if (timestamp_to_kf.find(ts) != timestamp_to_kf.end()) {
        current_kf_id = timestamp_to_kf.at(ts);
    } else {
        // 尝试通过时间戳近似找出KF ID
        for (size_t i = 0; i < timestamps.size(); i++) {
            if (std::abs(timestamps[i] - ts) < 1e-6) {
                current_kf_id = i;
                break;
            }
        }
    }
    
    if (current_kf_id < 0) return 1000; // 找不到KF ID，返回一个大值
    
    // 找到最近的Sim3矫正关键帧
    int min_distance = 1000;
    for (size_t i = 0; i < timestamps.size(); i++) {
        if (timestamps[i] >= boundary_timestamp) {
            int distance = std::abs(static_cast<int>(i) - current_kf_id);
            min_distance = std::min(min_distance, distance);
        }
    }
    
    return min_distance;
}

// 函数为解析TUM格式的一行
bool parseTUMPoseLine(const std::string& line, double& timestamp, Eigen::Vector3d& translation, Eigen::Quaterniond& rotation) {
    std::istringstream iss(line);
    double tx, ty, tz, qx, qy, qz, qw;
    
    // 跳过注释行
    if (line.empty() || line[0] == '#') return false;
    
    // 读取时间戳和姿态
    if (!(iss >> timestamp >> tx >> ty >> tz >> qx >> qy >> qz >> qw)) return false;
    
    translation = Eigen::Vector3d(tx, ty, tz);
    rotation = Eigen::Quaterniond(qw, qx, qy, qz); // 注意：Eigen四元数构造函数是(w,x,y,z)
    rotation.normalize(); // 确保归一化
    
    return true;
}

// 实现轨迹优化，保持前端和尾部对齐
void PropagateAndRotateBack(const std::string& original_file, const std::string& partially_corrected_file, const std::string& output_file) {
    std::cout << "开始轨迹优化，保持前端和尾部良好对齐..." << std::endl;
    
    // 1. 读取原始轨迹
    std::map<double, std::pair<Eigen::Vector3d, Eigen::Quaterniond>> original_poses;
    std::map<int, double> kf_to_timestamp; // KF ID到时间戳的映射
    std::map<double, int> timestamp_to_kf; // 时间戳到KF ID的映射
    
    std::ifstream original_file_stream(original_file);
    if (!original_file_stream.is_open()) {
        std::cerr << "错误：无法打开原始轨迹文件：" << original_file << std::endl;
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
    
    std::cout << "读取了 " << original_poses.size() << " 个原始关键帧姿态" << std::endl;
    
    // 2. 读取部分校正的轨迹并区分已校正和未校正的帧
    std::map<double, std::pair<Eigen::Vector3d, Eigen::Quaterniond>> corrected_poses;
    std::map<double, std::pair<Eigen::Vector3d, Eigen::Quaterniond>> uncorrected_poses;
    
    std::ifstream partially_corrected_file_stream(partially_corrected_file);
    if (!partially_corrected_file_stream.is_open()) {
        std::cerr << "错误：无法打开部分校正的轨迹文件：" << partially_corrected_file << std::endl;
        return;
    }
    
    // 边界时间戳，用于区分已校正/未校正的帧
    double boundary_timestamp = 1317384588915486208.000000000; // 对应于KF441
    
    // 闭环关键帧的ID
    const int loop_kf_id = 2;     // 闭环关键帧
    const int current_kf_id = 464; // 当前关键帧
    
    // 获取闭环关键帧和当前关键帧的时间戳
    double loop_timestamp = 0;
    double current_timestamp = 0;
    
    if (kf_to_timestamp.find(loop_kf_id) != kf_to_timestamp.end()) {
        loop_timestamp = kf_to_timestamp[loop_kf_id];
        std::cout << "闭环关键帧 " << loop_kf_id << " 对应时间戳：" << std::fixed << loop_timestamp << std::endl;
    } else {
        std::cerr << "警告：未找到闭环关键帧 " << loop_kf_id << " 的时间戳" << std::endl;
    }
    
    if (kf_to_timestamp.find(current_kf_id) != kf_to_timestamp.end()) {
        current_timestamp = kf_to_timestamp[current_kf_id];
        std::cout << "当前关键帧 " << current_kf_id << " 对应时间戳：" << std::fixed << current_timestamp << std::endl;
    } else {
        std::cerr << "警告：未找到当前关键帧 " << current_kf_id << " 的时间戳" << std::endl;
    }
    
    while (std::getline(partially_corrected_file_stream, line)) {
        if (parseTUMPoseLine(line, timestamp, translation, rotation)) {
            if (timestamp >= boundary_timestamp) { // 大于等于边界时间戳的是已校正帧
                corrected_poses[timestamp] = std::make_pair(translation, rotation);
            } else { // 小于边界时间戳的是未校正帧
                uncorrected_poses[timestamp] = std::make_pair(translation, rotation);
            }
        }
    }
    
    std::cout << "读取了 " << corrected_poses.size() << " 个已校正关键帧姿态" << std::endl;
    std::cout << "读取了 " << uncorrected_poses.size() << " 个未校正关键帧姿态" << std::endl;
    
    // 3. 构建关键帧连接关系 - 按时间戳排序
    std::vector<double> timestamps;
    for (const auto& pose_pair : original_poses) {
        timestamps.push_back(pose_pair.first);
    }
    
    // 如果原始文件为空，使用部分校正文件中的所有时间戳
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
    
    // 4. 创建最终传播结果
    std::map<double, std::pair<Eigen::Vector3d, Eigen::Quaterniond>> propagated_poses;
    
    // 首先复制所有已校正的姿态
    for (const auto& corrected_pair : corrected_poses) {
        propagated_poses[corrected_pair.first] = corrected_pair.second;
    }
    
    // 定义前端保留关键帧的百分比 (减少到10%, 原始为15%)
    const double front_preserve_percentage = 0.1; // 修改：从0.15减少到0.1
    const int front_preserve_count = static_cast<int>(timestamps.size() * front_preserve_percentage);
    
    std::cout << "将保留前 " << front_preserve_count << " 个关键帧的原始姿态 (约 " 
              << front_preserve_percentage * 100 << "%)" << std::endl;
    
    // 添加前端保留的姿态
    for (int i = 0; i < front_preserve_count; ++i) {
        if (i < timestamps.size()) {
            double ts = timestamps[i];
            // 使用原始轨迹的姿态
            if (original_poses.find(ts) != original_poses.end()) {
                propagated_poses[ts] = original_poses[ts];
                if (i % 10 == 0 || i == front_preserve_count - 1) {
                    std::cout << "保留KF " << i << " 原始姿态 (时间戳: " << std::fixed << ts << ")" << std::endl;
                }
            }
        }
    }
    
    // 5. 创建两个传播队列，一个从前端向后，一个从尾部向前
    std::queue<double> front_queue; // 从前端保留区域边界向后
    std::queue<double> back_queue;  // 从尾部已校正区域向前
    std::set<double> front_visited, back_visited;
    
    // 前端保留区最后一帧的下一帧作为前向传播的起点
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
    
    // 已校正区域第一帧的前一帧作为后向传播的起点
    if (!corrected_poses.empty()) {
        double back_boundary_ts = corrected_poses.begin()->first;
        if (prev_frame.find(back_boundary_ts) != prev_frame.end()) {
            double start_ts = prev_frame[back_boundary_ts];
            back_queue.push(start_ts);
            back_visited.insert(start_ts);
            std::cout << "后向传播起点: 时间戳 " << std::fixed << start_ts << std::endl;
        }
    }
    
    // 修改：优先处理与Sim3校正区域接近的关键帧
    // 创建优先队列根据到Sim3区域的距离
    struct KeyframeInfo {
        double timestamp;
        int distance_to_sim3;
        
        bool operator<(const KeyframeInfo& other) const {
            return distance_to_sim3 > other.distance_to_sim3; // 距离小的优先
        }
    };
    
    std::priority_queue<KeyframeInfo> front_priority_queue;
    
    // 替换普通队列为优先队列
    if (!front_queue.empty()) {
        double start_ts = front_queue.front();
        front_queue.pop();
        int dist = calculateDistanceToSim3Region(start_ts, boundary_timestamp, timestamp_to_kf, timestamps);
        front_priority_queue.push({start_ts, dist});
    }
    
    // 存储前向和后向传播结果
    std::map<double, std::pair<Eigen::Vector3d, Eigen::Quaterniond>> front_propagated;
    std::map<double, std::pair<Eigen::Vector3d, Eigen::Quaterniond>> back_propagated;
    
    // 存储每个关键帧到前端和尾部边界的距离
    std::map<double, int> distance_from_front;
    std::map<double, int> distance_from_back;
    
    // 辅助函数：从源获取姿态
    auto getPoseFromSource = [&](double ts) -> std::pair<Eigen::Vector3d, Eigen::Quaterniond> {
        // 首先尝试从原始轨迹获取
        if (original_poses.find(ts) != original_poses.end()) {
            return original_poses[ts];
        }
        // 然后尝试从未校正轨迹获取
        else if (uncorrected_poses.find(ts) != uncorrected_poses.end()) {
            return uncorrected_poses[ts];
        }
        // 最后从已校正轨迹获取
        else if (corrected_poses.find(ts) != corrected_poses.end()) {
            return corrected_poses[ts];
        }
        // 如果找不到，返回单位姿态
        else {
            std::cerr << "警告：找不到时间戳 " << std::fixed << ts << " 的姿态" << std::endl;
            return std::make_pair(Eigen::Vector3d::Zero(), Eigen::Quaterniond::Identity());
        }
    };
    
    // 修改：调整衰减因子计算函数
    auto calculateDecayFactor = [](int distance, bool is_tail_section, bool is_forward = true) -> double {
        // 为前向传播使用更慢的衰减
        if (is_forward) {
            if (distance <= 50) {
                return std::pow(0.99, distance); // 从0.97改为0.99，衰减更慢
            } else if (distance <= 100) {
                return std::pow(0.99, 50) * std::pow(0.98, distance - 50); 
            } else {
                return std::pow(0.99, 50) * std::pow(0.98, 50) * std::pow(0.97, distance - 100);
            }
        } else {
            // 保持尾部区域的衰减因子
            if (is_tail_section) {
                if (distance <= 50) {
                    return std::pow(0.98, distance);
                } else if (distance <= 100) {
                    return std::pow(0.98, 50) * std::pow(0.97, distance - 50);
                } else {
                    return std::pow(0.98, 50) * std::pow(0.97, 50) * std::pow(0.96, distance - 100);
                }
            } else {
                // 原始对其他区域的因子
                if (distance <= 50) {
                    return std::pow(0.97, distance);
                } else if (distance <= 100) {
                    return std::pow(0.97, 50) * std::pow(0.96, distance - 50);
                } else {
                    return std::pow(0.97, 50) * std::pow(0.96, 50) * std::pow(0.95, distance - 100);
                }
            }
        }
    };
    
    // 6. 前向传播 - 从前端保留区域向后
    std::cout << "开始前向传播..." << std::endl;
    
    if (!front_priority_queue.empty()) {
        distance_from_front[front_priority_queue.top().timestamp] = 1;
    }
    
    while (!front_priority_queue.empty()) {
        double current_ts = front_priority_queue.top().timestamp;
        front_priority_queue.pop();
        
        // 跳过已处理的关键帧
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
        
        // 如果前一帧已有传播姿态
        if (propagated_poses.find(prev_ts) != propagated_poses.end() || 
            front_propagated.find(prev_ts) != front_propagated.end()) {
            
            // 获取原始轨迹中的相对姿态
            auto orig_prev_pose = getPoseFromSource(prev_ts);
            auto orig_current_pose = getPoseFromSource(current_ts);
            
            Eigen::Vector3d orig_prev_t = orig_prev_pose.first;
            Eigen::Quaterniond orig_prev_q = orig_prev_pose.second;
            Eigen::Matrix3d orig_prev_R = orig_prev_q.toRotationMatrix();
            
            Eigen::Vector3d orig_current_t = orig_current_pose.first;
            Eigen::Quaterniond orig_current_q = orig_current_pose.second;
            Eigen::Matrix3d orig_current_R = orig_current_q.toRotationMatrix();
            
            // 计算相对变换(从前一帧到当前帧)
            Eigen::Matrix3d R_prev_to_current = orig_prev_R.transpose() * orig_current_R;
            Eigen::Vector3d t_prev_to_current = orig_prev_R.transpose() * (orig_current_t - orig_prev_t);
            
            // 获取前一帧的传播姿态
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
            
            // 应用相对变换得到当前帧的传播姿态
            Eigen::Matrix3d current_prop_R = prev_prop_R * R_prev_to_current;
            Eigen::Vector3d current_prop_t = prev_prop_t + prev_prop_R * t_prev_to_current;
            
            Eigen::Quaterniond current_prop_q(current_prop_R);
            current_prop_q.normalize();
            
            // 存储当前帧的前向传播姿态
            front_propagated[current_ts] = std::make_pair(current_prop_t, current_prop_q);
            
            // 计算距离
            int current_distance = distance_from_front[prev_ts] + 1;
            distance_from_front[current_ts] = current_distance;
            
            // 如果有下一帧且未被访问，将其添加到队列
            if (next_frame.find(current_ts) != next_frame.end()) {
                double next_ts = next_frame[current_ts];
                if (front_visited.find(next_ts) == front_visited.end() && 
                    corrected_poses.find(next_ts) == corrected_poses.end()) {
                    
                    int dist_to_sim3 = calculateDistanceToSim3Region(next_ts, boundary_timestamp, timestamp_to_kf, timestamps);
                    front_priority_queue.push({next_ts, dist_to_sim3});
                    front_visited.insert(next_ts);
                }
            }
            
            if (current_distance % 50 == 0) {
                std::cout << "前向传播: 距离KF " << current_distance 
                          << ", 时间戳: " << std::fixed << current_ts << std::endl;
            }
        }
    }
    
    std::cout << "前向传播完成，处理了 " << front_propagated.size() << " 个关键帧" << std::endl;
    
    // 7. 后向传播 - 从尾部已校正区域向前
    std::cout << "开始后向传播..." << std::endl;
    
    if (!back_queue.empty()) {
        distance_from_back[back_queue.front()] = 1;
    }
    
    while (!back_queue.empty()) {
        double current_ts = back_queue.front();
        back_queue.pop();
        
        // 跳过已处理的关键帧
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
        
        // 如果下一帧已有传播姿态或是已校正姿态
        if (propagated_poses.find(next_ts) != propagated_poses.end() || 
            back_propagated.find(next_ts) != back_propagated.end()) {
            
            // 获取原始轨迹中的相对姿态
            auto orig_current_pose = getPoseFromSource(current_ts);
            auto orig_next_pose = getPoseFromSource(next_ts);
            
            Eigen::Vector3d orig_current_t = orig_current_pose.first;
            Eigen::Quaterniond orig_current_q = orig_current_pose.second;
            Eigen::Matrix3d orig_current_R = orig_current_q.toRotationMatrix();
            
            Eigen::Vector3d orig_next_t = orig_next_pose.first;
            Eigen::Quaterniond orig_next_q = orig_next_pose.second;
            Eigen::Matrix3d orig_next_R = orig_next_q.toRotationMatrix();
            
            // 计算相对变换(从当前帧到下一帧)
            Eigen::Matrix3d R_current_to_next = orig_current_R.transpose() * orig_next_R;
            Eigen::Vector3d t_current_to_next = orig_current_R.transpose() * (orig_next_t - orig_current_t);
            
            // 获取下一帧的姿态
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
            
            // 应用逆变换得到当前帧的传播姿态
            Eigen::Matrix3d current_prop_R = next_prop_R * R_next_to_current;
            Eigen::Vector3d current_prop_t = next_prop_t + next_prop_R * t_next_to_current;
            
            Eigen::Quaterniond current_prop_q(current_prop_R);
            current_prop_q.normalize();
            
            // 存储当前帧的后向传播姿态
            back_propagated[current_ts] = std::make_pair(current_prop_t, current_prop_q);
            
            // 计算距离
            int current_distance = distance_from_back[next_ts] + 1;
            distance_from_back[current_ts] = current_distance;
            
            // 如果有前一帧且未被访问，将其添加到队列
            if (prev_frame.find(current_ts) != prev_frame.end()) {
                double prev_ts = prev_frame[current_ts];
                if (back_visited.find(prev_ts) == back_visited.end() && 
                    propagated_poses.find(prev_ts) == propagated_poses.end()) {
                    back_queue.push(prev_ts);
                    back_visited.insert(prev_ts);
                }
            }
            
            if (current_distance % 50 == 0) {
                std::cout << "后向传播: 距离KF " << current_distance 
                          << ", 时间戳: " << std::fixed << current_ts << std::endl;
            }
        }
    }
    
    std::cout << "后向传播完成，处理了 " << back_propagated.size() << " 个关键帧" << std::endl;
    
    // 8. 融合前向和后向传播结果
    std::cout << "融合前向和后向传播结果..." << std::endl;
    
    // 确定尾部区域的起始点
    int tail_start_idx = static_cast<int>(timestamps.size() * 0.75);
    
    for (double ts : timestamps) {
        // 如果已在propagated_poses中，跳过 (保留原始前端和已校正尾部)
        if (propagated_poses.find(ts) != propagated_poses.end()) {
            continue;
        }
        
        bool has_front = (front_propagated.find(ts) != front_propagated.end());
        bool has_back = (back_propagated.find(ts) != back_propagated.end());
        
        // 确定是否在尾部区域
        int kf_id = -1;
        if (timestamp_to_kf.find(ts) != timestamp_to_kf.end()) {
            kf_id = timestamp_to_kf[ts];
        } else {
            // 如果找不到KF ID，通过位置估计
            for (size_t i = 0; i < timestamps.size(); i++) {
                if (std::abs(timestamps[i] - ts) < 1e-6) {
                    kf_id = i;
                    break;
                }
            }
        }
        
        bool is_tail_section = (kf_id >= tail_start_idx);
        
        // 计算到Sim3区域的距离，用于影响力权重
        int distance_to_sim3 = calculateDistanceToSim3Region(ts, boundary_timestamp, timestamp_to_kf, timestamps);
        // 计算Sim3影响力权重 - 指数衰减
        double sim3_influence = 1.0 + 2.0 * exp(-distance_to_sim3 / 50.0);
        
        if (has_front && has_back) {
            // 有两个方向的传播结果，需要融合
            int front_dist = distance_from_front[ts];
            int back_dist = distance_from_back[ts];
            
            // 修改：增加前向传播的偏置因子
            double front_bias = 1.5; // 新增前向偏置，增加前向影响
            double back_bias = 1.0;
            
            if (is_tail_section) {
                // 在尾部区域，逐渐增加后向偏置
                double progress = (kf_id - tail_start_idx) / static_cast<double>(timestamps.size() - tail_start_idx);
                back_bias = 1.0 + progress * 1.5;
            }
            
            // 计算融合权重 - 距离越小越有影响
            double total_dist = front_dist * front_bias + back_dist * back_bias;
            double front_weight = (back_dist * back_bias) / total_dist;  // 前向传播的权重
            
            // 根据到Sim3区域的距离增加前向权重
            front_weight = front_weight * sim3_influence;
            // 确保权重在有效范围
            front_weight = std::min(0.95, std::max(0.05, front_weight));
            
            // 获取两个方向的姿态
            Eigen::Vector3d front_t = front_propagated[ts].first;
            Eigen::Quaterniond front_q = front_propagated[ts].second;
            
            Eigen::Vector3d back_t = back_propagated[ts].first;
            Eigen::Quaterniond back_q = back_propagated[ts].second;
            
            // 融合平移
            Eigen::Vector3d blended_t = front_t * front_weight + back_t * (1.0 - front_weight);
            
            // 融合旋转 (球面线性插值)
            Eigen::Quaterniond blended_q = front_q.slerp(1.0 - front_weight, back_q);
            blended_q.normalize();
            
            // 存储融合后的姿态
            propagated_poses[ts] = std::make_pair(blended_t, blended_q);
            
            if ((front_dist + back_dist) % 100 == 0 || is_tail_section) {
                std::cout << "融合: 时间戳 " << std::fixed << ts 
                          << ", 前向距离: " << front_dist 
                          << ", 后向距离: " << back_dist 
                          << ", 前向权重: " << front_weight
                          << ", Sim3影响: " << sim3_influence
                          << (is_tail_section ? " (尾部区域)" : "") << std::endl;
            }
        } else if (has_front) {
            // 只有前向传播结果
            propagated_poses[ts] = front_propagated[ts];
        } else if (has_back) {
            // 只有后向传播结果
            propagated_poses[ts] = back_propagated[ts];
        } else {
            // 两个方向都没有结果，使用原始姿态
            propagated_poses[ts] = getPoseFromSource(ts);
        }
    }
    
    // 9. 计算前端关键帧的回归变换
    std::cout << "计算前端关键帧的回归变换..." << std::endl;
    
    // 选择前20个关键帧用于变换拟合
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
    
    // 至少需要3个点才能计算有意义的变换
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
        
        // 中心化点
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
        
        // 确保是行列式为1的正交矩阵 (旋转矩阵，不是反射矩阵)
        if (front_rotation.determinant() < 0) {
            V.col(2) = -V.col(2);
            front_rotation = V * U.transpose();
        }
        
        // 计算平移向量
        front_translation = orig_centroid - front_rotation * prop_centroid;
        
        has_front_transform = true;
        std::cout << "成功计算前端回归变换" << std::endl;
    } else {
        std::cout << "前端关键帧不足，无法计算有效的回归变换" << std::endl;
    }
    
    // 10. 应用前端回归变换进行全局调整
    if (has_front_transform) {
        std::cout << "应用前端回归变换进行全局调整..." << std::endl;
        
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
                // 如果找不到KF ID，通过位置估计
                for (int i = 0; i < timestamps.size(); i++) {
                    if (std::abs(timestamps[i] - ts) < 1e-6) {
                        kf_id = i;
                        break;
                    }
                }
            }
            
            // 修改：根据KF ID计算权重，使用更慢的衰减
            double weight = 0.0;
            
            if (kf_id >= 0) {
                // 调整为更慢的衰减
                const int total_kfs = timestamps.size();
                const double decay_rate = 3.0; // 从4.0减少到3.0，衰减更慢
                
                if (kf_id < front_preserve_count) {
                    // 前端保留区域使用高权重
                    weight = 0.95;
                } else {
                    // 后续KF使用平缓衰减
                    double normalized_pos = static_cast<double>(kf_id - front_preserve_count) / 
                                           (total_kfs - front_preserve_count);
                    weight = 0.95 * std::exp(-decay_rate * normalized_pos);
                    
                    // 对于尾部区域，进一步减少衰减
                    bool is_in_tail = (kf_id >= tail_start_idx);
                    if (is_in_tail) {
                        // 增加尾部权重
                        weight *= 0.5; // 从0.7减少到0.5，增加矫正影响
                    }
                }
                
                // 确保权重在有效范围
                weight = std::max(0.0, std::min(1.0, weight));
            }
            
            // 修改：特殊处理闭环关键帧，确保闭环约束
            if (ts == loop_timestamp) {
                weight = 0.2; // 从0.25减少到0.2，增加矫正影响
            } else if (ts >= boundary_timestamp) {
                // 修改：对已校正关键帧，使用更小的调整权重
                weight = std::min(weight, 0.15); // 从0.2减少到0.15，增加矫正影响
            }
            
            // 计算到Sim3区域的距离，用于增强影响
            int distance_to_sim3 = calculateDistanceToSim3Region(ts, boundary_timestamp, timestamp_to_kf, timestamps);
            // Sim3影响力权重 - 更强的影响
            double sim3_influence = 1.0;
            
            // 根据到Sim3区域的距离调整权重
            if (distance_to_sim3 < 100) {
                sim3_influence = 1.0 + 3.0 * exp(-distance_to_sim3 / 30.0);
                // 减小权重，使Sim3校正有更大影响
                weight = weight / sim3_influence;
            }
            
            // 应用旋转和平移变换
            Eigen::Vector3d new_t;
            Eigen::Matrix3d new_R;
            
            if (weight > 0) {
                // 应用加权回归变换
                Eigen::Matrix3d weighted_rotation = Eigen::Matrix3d::Identity() * (1.0 - weight) + 
                                                   front_rotation * weight;
                
                // 使用SVD确保旋转矩阵正交
                Eigen::JacobiSVD<Eigen::Matrix3d> svd(weighted_rotation, Eigen::ComputeFullU | Eigen::ComputeFullV);
                Eigen::Matrix3d orthogonal_rotation = svd.matrixU() * svd.matrixV().transpose();
                
                // 应用旋转
                new_R = orthogonal_rotation * prop_R;
                
                // 应用平移
                new_t = orthogonal_rotation * prop_t + front_translation * weight;
            } else {
                // 保持原始姿态不变
                new_t = prop_t;
                new_R = prop_R;
            }
            
            // 创建最终姿态
            Eigen::Quaterniond new_q(new_R);
            new_q.normalize();
            
            // 更新姿态
            adjusted_poses[ts] = std::make_pair(new_t, new_q);
            
            // 确定是否在尾部区域
            bool is_in_tail_section = (kf_id >= tail_start_idx);
            
            if (kf_id >= 0 && (kf_id % 50 == 0 || ts == loop_timestamp || kf_id == front_preserve_count || 
                               (is_in_tail_section && kf_id % 25 == 0))) {
                std::cout << "KF " << kf_id << " 应用了 " << weight * 100 << "% 的回归变换" 
                         << (is_in_tail_section ? " (尾部区域)" : "")
                         << ", Sim3影响: " << sim3_influence << std::endl;
            }
        }
        
        // 更新姿态
        propagated_poses = adjusted_poses;
    }
    
    // 11. 确保满足闭环约束
    if (loop_timestamp > 0 && current_timestamp > 0 && 
        propagated_poses.find(loop_timestamp) != propagated_poses.end() &&
        propagated_poses.find(current_timestamp) != propagated_poses.end()) {
        
        std::cout << "验证并调整闭环约束..." << std::endl;
        
        // 获取原始轨迹中闭环和当前关键帧的姿态
        auto orig_loop_pose = getPoseFromSource(loop_timestamp);
        auto orig_current_pose = getPoseFromSource(current_timestamp);
        
        Eigen::Matrix3d orig_loop_R = orig_loop_pose.second.toRotationMatrix();
        Eigen::Vector3d orig_loop_t = orig_loop_pose.first;
        
        Eigen::Matrix3d orig_current_R = orig_current_pose.second.toRotationMatrix();
        Eigen::Vector3d orig_current_t = orig_current_pose.first;
        
        // 计算原始闭环变换
        Eigen::Matrix3d orig_loop_to_current_R = orig_loop_R.transpose() * orig_current_R;
        Eigen::Vector3d orig_loop_to_current_t = orig_loop_R.transpose() * (orig_current_t - orig_loop_t);
        
        // 获取当前闭环和当前关键帧的姿态
        auto& loop_pose = propagated_poses[loop_timestamp];
        auto& current_pose = propagated_poses[current_timestamp];
        
        Eigen::Matrix3d loop_R = loop_pose.second.toRotationMatrix();
        Eigen::Vector3d loop_t = loop_pose.first;
        
        Eigen::Matrix3d current_R = current_pose.second.toRotationMatrix();
        Eigen::Vector3d current_t = current_pose.first;
        
        // 计算当前闭环变换
        Eigen::Matrix3d current_loop_to_current_R = loop_R.transpose() * current_R;
        Eigen::Vector3d current_loop_to_current_t = loop_R.transpose() * (current_t - loop_t);
        
        // 计算闭环误差
        double rot_error = (current_loop_to_current_R - orig_loop_to_current_R).norm();
        double trans_error = (current_loop_to_current_t - orig_loop_to_current_t).norm();
        
        std::cout << "闭环约束误差: 旋转 = " << rot_error << ", 平移 = " << trans_error << std::endl;
        
        // 如果误差较大，调整当前关键帧以满足闭环约束
        if (rot_error > 0.05 || trans_error > 0.3) { // 降低误差阈值，更容易触发调整
            // 使用闭环关键帧和原始相对变换计算当前关键帧的正确姿态
            Eigen::Matrix3d corrected_current_R = loop_R * orig_loop_to_current_R;
            Eigen::Vector3d corrected_current_t = loop_t + loop_R * orig_loop_to_current_t;
            
            // 融合原始姿态和校正姿态
            double blend_factor = 0.9; // 从0.8增加到0.9，更倾向于校正 (90%校正, 10%原始)
            
            Eigen::Vector3d blended_t = current_t * (1.0 - blend_factor) + corrected_current_t * blend_factor;
            
            // 使用球面线性插值融合旋转
            Eigen::Quaterniond current_q = current_pose.second;
            Eigen::Quaterniond corrected_q(corrected_current_R);
            Eigen::Quaterniond blended_q = current_q.slerp(blend_factor, corrected_q);
            blended_q.normalize();
            
            // 更新当前关键帧姿态
            propagated_poses[current_timestamp] = std::make_pair(blended_t, blended_q);
            
            std::cout << "调整当前关键帧姿态以满足闭环约束" << std::endl;
        }
    }
    
    // 11.5 尾部区域特定精化
    std::cout << "对轨迹尾部应用额外精化..." << std::endl;
    
    // 获取最后正确校正的姿态作为参考
    std::vector<double> corrected_tail_ts;
    for (const auto& pair : corrected_poses) {
        corrected_tail_ts.push_back(pair.first);
    }
    
    // 按时间排序
    std::sort(corrected_tail_ts.begin(), corrected_tail_ts.end());
    
    // 取前5个正确校正的姿态作为参考
    const int ref_pose_count = std::min(5, static_cast<int>(corrected_tail_ts.size()));
    
    if (ref_pose_count >= 3) {
        std::vector<Eigen::Vector3d> tail_ref_positions;
        std::vector<Eigen::Vector3d> tail_current_positions;
        
        // 收集参考位置和当前位置
        for (int i = 0; i < ref_pose_count; i++) {
            double ts = corrected_tail_ts[i];
            tail_ref_positions.push_back(corrected_poses[ts].first);
            tail_current_positions.push_back(propagated_poses[ts].first);
        }
        
        // 计算质心
        Eigen::Vector3d tail_ref_centroid = Eigen::Vector3d::Zero();
        Eigen::Vector3d tail_current_centroid = Eigen::Vector3d::Zero();
        
        for (int i = 0; i < ref_pose_count; i++) {
            tail_ref_centroid += tail_ref_positions[i];
            tail_current_centroid += tail_current_positions[i];
        }
        
        tail_ref_centroid /= ref_pose_count;
        tail_current_centroid /= ref_pose_count;
        
        // 中心化点
        std::vector<Eigen::Vector3d> tail_ref_centered, tail_current_centered;
        for (int i = 0; i < ref_pose_count; i++) {
            tail_ref_centered.push_back(tail_ref_positions[i] - tail_ref_centroid);
            tail_current_centered.push_back(tail_current_positions[i] - tail_current_centroid);
        }
        
        // 构建协方差矩阵
        Eigen::Matrix3d H_tail = Eigen::Matrix3d::Zero();
        for (int i = 0; i < ref_pose_count; i++) {
            H_tail += tail_current_centered[i] * tail_ref_centered[i].transpose();
        }
        
        // SVD分解
        Eigen::JacobiSVD<Eigen::Matrix3d> svd_tail(H_tail, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix3d U_tail = svd_tail.matrixU();
        Eigen::Matrix3d V_tail = svd_tail.matrixV();
        
        // 计算旋转矩阵
        Eigen::Matrix3d tail_rotation = V_tail * U_tail.transpose();
        
        // 确保是行列式为1的正交矩阵
        if (tail_rotation.determinant() < 0) {
            V_tail.col(2) = -V_tail.col(2);
            tail_rotation = V_tail * U_tail.transpose();
        }
        
        // 计算平移向量
        Eigen::Vector3d tail_translation = tail_ref_centroid - tail_rotation * tail_current_centroid;
        
        // 修改：调整尾部区域以获得更好的对齐
        std::cout << "调整尾部姿态以获得更好的对齐..." << std::endl;
        
        for (int i = tail_start_idx; i < timestamps.size(); i++) {
            double ts = timestamps[i];
            
            if (propagated_poses.find(ts) != propagated_poses.end() && 
                corrected_poses.find(ts) == corrected_poses.end()) { // 不调整已校正姿态
                
                // 根据位置计算权重因子
                double progress = static_cast<double>(i - tail_start_idx) / 
                                 (timestamps.size() - tail_start_idx);
                double weight = 0.8 * std::exp(-1.2 * progress); // 增加权重，减缓衰减
                
                // 获取当前姿态
                auto& current_pose = propagated_poses[ts];
                Eigen::Vector3d current_t = current_pose.first;
                Eigen::Quaterniond current_q = current_pose.second;
                Eigen::Matrix3d current_R = current_q.toRotationMatrix();
                
                // 应用加权变换
                Eigen::Matrix3d weighted_rotation = Eigen::Matrix3d::Identity() * (1.0 - weight) + 
                                                   tail_rotation * weight;
                
                // 确保正交性
                Eigen::JacobiSVD<Eigen::Matrix3d> svd_weighted(weighted_rotation, Eigen::ComputeFullU | Eigen::ComputeFullV);
                Eigen::Matrix3d orthogonal_rotation = svd_weighted.matrixU() * svd_weighted.matrixV().transpose();
                
                // 应用调整的旋转和平移
                Eigen::Matrix3d adjusted_R = orthogonal_rotation * current_R;
                Eigen::Vector3d adjusted_t = orthogonal_rotation * current_t + tail_translation * weight;
                
                // 创建最终姿态
                Eigen::Quaterniond adjusted_q(adjusted_R);
                adjusted_q.normalize();
                
                // 更新姿态
                propagated_poses[ts] = std::make_pair(adjusted_t, adjusted_q);
                
                if (i % 25 == 0 || i == timestamps.size() - 1) {
                    std::cout << "尾部KF " << i << " 应用了 " << weight * 100 << "% 的额外调整" << std::endl;
                }
            }
        }
    } else {
        std::cout << "尾部参考姿态不足，无法进行额外调整" << std::endl;
    }
    
    // 12. 对整个轨迹应用平滑处理
    std::cout << "对轨迹应用平滑处理..." << std::endl;
    
    std::map<double, std::pair<Eigen::Vector3d, Eigen::Quaterniond>> smoothed_poses = propagated_poses;
    
    // 平滑窗口大小
    const int window_size = 5;
    const int half_window = window_size / 2;
    
    // 对每个关键帧应用平滑
    for (size_t i = half_window; i < timestamps.size() - half_window; ++i) {
        double current_ts = timestamps[i];
        
        // 跳过前端保留区域和尾部已校正区域
        if (i < front_preserve_count || 
            (corrected_poses.find(current_ts) != corrected_poses.end())) {
            continue;
        }
        
        // 确定是否在尾部区域
        bool is_tail_section = (i >= tail_start_idx);
        
        // 收集窗口内的姿态
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
                    
                    // 中心点权重最大，向两边递减
                    double weight = 1.0 - std::abs(j) / (half_window + 1.0);
                    weights.push_back(weight);
                }
            }
        }
        
        // 如果窗口内有足够的姿态可以平滑
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
            
            // 平滑因子 - 控制平滑程度
            double smooth_factor = 0.5;
            
            // 修改：增加尾部区域的平滑因子
            if (is_tail_section) {
                // 从0.5+0.2*progress增加到0.6+0.2*progress
                double progress = (i - tail_start_idx) / static_cast<double>(timestamps.size() - tail_start_idx);
                smooth_factor = 0.6 + progress * 0.2;
            }
            
            // 计算距离Sim3区域
            int distance_to_sim3 = calculateDistanceToSim3Region(current_ts, boundary_timestamp, timestamp_to_kf, timestamps);
            
            // 如果接近Sim3区域，减少平滑以保留更多信息
            if (distance_to_sim3 < 50) {
                smooth_factor = smooth_factor * (0.5 + 0.5 * distance_to_sim3 / 50.0);
            }
            
            // 融合原始位置和平滑位置
            Eigen::Vector3d orig_t = propagated_poses[current_ts].first;
            Eigen::Vector3d smoothed_t = orig_t * (1.0 - smooth_factor) + avg_t * smooth_factor;
            
            // 平滑旋转
            // 使用四元数加权平均
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
            
            // 使用球面线性插值融合原始旋转和平滑旋转
            Eigen::Quaterniond smoothed_q = orig_q.slerp(smooth_factor, avg_q);
            smoothed_q.normalize();
            
            // 更新平滑后的姿态
            smoothed_poses[current_ts] = std::make_pair(smoothed_t, smoothed_q);
        }
    }
    
    // 更新姿态
    propagated_poses = smoothed_poses;
    
    // 13. 将优化后的姿态写入文件
    std::ofstream out_file(output_file);
    if (!out_file.is_open()) {
        std::cerr << "错误：无法打开输出文件：" << output_file << std::endl;
        return;
    }
    
    // 按时间戳顺序写入
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
    
    std::cout << "优化后的姿态已保存至：" << output_file << std::endl;
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
    
    // 创建输出目录 (如果不存在)
    std::string output_dir = output_file.substr(0, output_file.find_last_of('/'));
    system(("mkdir -p " + output_dir).c_str());
    
    // 执行姿态传播算法
    PropagateAndRotateBack(original_file, partially_corrected_file, output_file);
    
    return 0;
}
