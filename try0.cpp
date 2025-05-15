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
#include <Eigen/Core>
#include <Eigen/Geometry>

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

// 实现Twc格式的位姿传播
void PropagateCorrectionsInTwc(const std::string& original_file, const std::string& partially_corrected_file, const std::string& output_file) {
    std::cout << "开始实现Twc格式的位姿传播校正..." << std::endl;
    
    // 1. 读取未校正轨迹
    std::map<double, std::pair<Eigen::Vector3d, Eigen::Quaterniond>> original_poses;
    std::ifstream original_file_stream(original_file);
    if (!original_file_stream.is_open()) {
        std::cerr << "错误：无法打开原始轨迹文件: " << original_file << std::endl;
        return;
    }
    
    std::string line;
    double timestamp;
    Eigen::Vector3d translation;
    Eigen::Quaterniond rotation;
    
    while (std::getline(original_file_stream, line)) {
        if (parseTUMPoseLine(line, timestamp, translation, rotation)) {
            original_poses[timestamp] = std::make_pair(translation, rotation);
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
    double boundary_timestamp = 1317384588915486208.000000000; // 假设这是边界时间戳，对应KF441
    
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
    
    // 边界检查
    if (corrected_poses.empty()) {
        std::cerr << "错误：未找到任何校正后的关键帧位姿，请检查边界时间戳设置" << std::endl;
        return;
    }
    
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
    
    // 5. 创建一个从边界向外传播的队列
    std::queue<double> bfs_queue;
    std::set<double> visited;
    
    // 找到边界时间戳（已校正的最早时间戳）
    double boundary_ts = std::numeric_limits<double>::max();
    for (const auto& ts_pose : corrected_poses) {
        if (ts_pose.first < boundary_ts) {
            boundary_ts = ts_pose.first;
        }
    }
    
    std::cout << "边界时间戳: " << std::fixed << boundary_ts << std::endl;
    
    // 获取边界帧的前一帧作为传播起点
    if (prev_frame.find(boundary_ts) != prev_frame.end()) {
        double start_ts = prev_frame[boundary_ts];
        bfs_queue.push(start_ts);
        visited.insert(start_ts);
        std::cout << "开始传播的时间戳: " << std::fixed << start_ts << std::endl;
    } else {
        std::cerr << "警告：边界关键帧没有前序帧，无法向前传播" << std::endl;
    }
    
    // 定义衰减系数 - 距离边界越远，校正效果越小
    auto calculateDecayFactor = [](int distance_from_boundary) -> double {
        double base_decay = 0.95; // 衰减基数 - 可以调整，0.95表示每一步传播保留95%的校正效果
        return std::pow(base_decay, distance_from_boundary);
    };
    
    // 存储每个关键帧距离边界的距离
    std::map<double, int> distance_from_boundary;
    
    if (!bfs_queue.empty()) {
        distance_from_boundary[bfs_queue.front()] = 1;
    }
    
    // 6. BFS向前传播校正（从边界向前）
    std::cout << "开始向前传播位姿校正..." << std::endl;
    
    // 从源轨迹获取位姿的辅助函数
    auto getPoseFromSource = [&](double ts) -> std::pair<Eigen::Vector3d, Eigen::Quaterniond> {
        // 首先尝试从原始轨迹获取
        if (original_poses.find(ts) != original_poses.end()) {
            return original_poses[ts];
        }
        // 其次尝试从未校正轨迹获取
        else if (uncorrected_poses.find(ts) != uncorrected_poses.end()) {
            return uncorrected_poses[ts];
        }
        // 最后从校正轨迹获取（但这应该不会发生，因为我们跳过了校正帧）
        else if (corrected_poses.find(ts) != corrected_poses.end()) {
            return corrected_poses[ts];
        }
        // 如果都找不到，返回单位位姿
        else {
            std::cerr << "警告：时间戳 " << std::fixed << ts << " 的位姿未找到" << std::endl;
            return std::make_pair(Eigen::Vector3d::Zero(), Eigen::Quaterniond::Identity());
        }
    };
    
    while (!bfs_queue.empty()) {
        double current_ts = bfs_queue.front();
        bfs_queue.pop();
        
        // 跳过已校正的关键帧
        if (corrected_poses.find(current_ts) != corrected_poses.end()) {
            continue;
        }
        
        // 获取下一帧的时间戳
        double next_ts;
        if (next_frame.find(current_ts) != next_frame.end()) {
            next_ts = next_frame[current_ts];
        } else {
            // 如果没有下一帧，则停止
            continue;
        }
        
        // 如果下一帧已经有传播或校正位姿，则计算校正
        if (propagated_poses.find(next_ts) != propagated_poses.end()) {
            // 获取原始轨迹中的位姿
            auto orig_current_pose = getPoseFromSource(current_ts);
            auto orig_next_pose = getPoseFromSource(next_ts);
            
            Eigen::Vector3d orig_current_t = orig_current_pose.first;
            Eigen::Quaterniond orig_current_q = orig_current_pose.second;
            
            Eigen::Vector3d orig_next_t = orig_next_pose.first;
            Eigen::Quaterniond orig_next_q = orig_next_pose.second;
            
            // 计算相对位姿 (从当前帧到下一帧)
            Eigen::Matrix3d orig_current_R = orig_current_q.toRotationMatrix();
            Eigen::Matrix3d orig_next_R = orig_next_q.toRotationMatrix();
            
            // 原始轨迹中，下一帧相对于当前帧的相对位姿
            Eigen::Matrix3d R_current_to_next = orig_current_R.transpose() * orig_next_R;
            Eigen::Quaterniond q_current_to_next(R_current_to_next);
            
            // 相对平移: current^-1 * next
            Eigen::Vector3d t_current_to_next = orig_current_R.transpose() * (orig_next_t - orig_current_t);
            
            // 计算衰减因子
            int current_distance = distance_from_boundary[current_ts];
            double decay_factor = calculateDecayFactor(current_distance);
            
            // 获取下一帧的校正或传播位姿
            Eigen::Vector3d next_corrected_t = propagated_poses[next_ts].first;
            Eigen::Quaterniond next_corrected_q = propagated_poses[next_ts].second;
            Eigen::Matrix3d next_corrected_R = next_corrected_q.toRotationMatrix();
            
            // 应用相对变换的逆向得到当前帧的校正位姿
            // 这里的关键是R_next_to_current = R_current_to_next^T
            // 和t_next_to_current = -R_current_to_next^T * t_current_to_next
            Eigen::Matrix3d R_next_to_current = R_current_to_next.transpose();
            Eigen::Vector3d t_next_to_current = -R_next_to_current * t_current_to_next;
            
            // 组合下一帧的校正位姿和相对变换的逆
            Eigen::Matrix3d new_current_R = next_corrected_R * R_next_to_current;
            Eigen::Vector3d new_current_t = next_corrected_t + next_corrected_R * t_next_to_current;
            
            // 创建校正后的四元数
            Eigen::Quaterniond new_current_q(new_current_R);
            new_current_q.normalize();
            
            // 存储当前帧的校正位姿
            propagated_poses[current_ts] = std::make_pair(new_current_t, new_current_q);
            
            // 如果有前一帧且未访问过，加入队列
            if (prev_frame.find(current_ts) != prev_frame.end()) {
                double prev_ts = prev_frame[current_ts];
                if (visited.find(prev_ts) == visited.end()) {
                    bfs_queue.push(prev_ts);
                    visited.insert(prev_ts);
                    distance_from_boundary[prev_ts] = current_distance + 1;
                }
            }
            
            std::cout << "时间戳 " << std::fixed << current_ts << " 的位姿已从时间戳 " << next_ts 
                      << " 反向传播校正，衰减因子: " << decay_factor << std::endl;
        }
    }
    
    // 7. 处理没有被BFS访问到的关键帧
    for (const auto& ts_pair : timestamps) {
        double ts = ts_pair;
        if (propagated_poses.find(ts) == propagated_poses.end()) {
            // 如果是已校正的关键帧，使用校正位姿
            if (corrected_poses.find(ts) != corrected_poses.end()) {
                propagated_poses[ts] = corrected_poses[ts];
            } 
            // 如果是未校正的，则使用原始位姿
            else {
                auto source_pose = getPoseFromSource(ts);
                propagated_poses[ts] = source_pose;
            }
            std::cout << "时间戳 " << std::fixed << ts << " 未被位姿传播算法访问，使用"
                      << (corrected_poses.find(ts) != corrected_poses.end() ? "校正" : "原始")
                      << "位姿" << std::endl;
        }
    }
    
    // 8. 将优化后的位姿写入文件
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
    
    std::cout << "传播校正后的位姿已保存到: " << output_file << std::endl;
    std::cout << "位姿传播校正完成！" << std::endl;
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
    PropagateCorrectionsInTwc(original_file, partially_corrected_file, output_file);
    
    return 0;
}
