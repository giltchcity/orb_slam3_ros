#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <set>
#include <string>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <ceres/ceres.h>
#include <ceres/autodiff_manifold.h>

// 表示Sim3变换的类，类似于g2o::Sim3
class Sim3 {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    Sim3() : r_(1.0, 0.0, 0.0, 0.0), t_(0.0, 0.0, 0.0), s_(1.0) {}
    
    Sim3(const Eigen::Quaterniond& r, const Eigen::Vector3d& t, double s) : r_(r), t_(t), s_(s) {
        r_.normalize();
    }
    
    // Sim3变换的逆
    Sim3 inverse() const {
        Eigen::Quaterniond r_inv = r_.conjugate();
        Eigen::Vector3d t_inv = -(r_inv * t_) / s_;
        return Sim3(r_inv, t_inv, 1.0/s_);
    }
    
    // 组合两个Sim3变换
    Sim3 operator*(const Sim3& other) const {
        Eigen::Quaterniond r_result = r_ * other.r_;
        Eigen::Vector3d t_result = s_ * (r_ * other.t_) + t_;
        double s_result = s_ * other.s_;
        return Sim3(r_result, t_result, s_result);
    }
    
    // 变换一个3D点
    Eigen::Vector3d map(const Eigen::Vector3d& p) const {
        return s_ * (r_ * p) + t_;
    }
    
    // 访问器
    const Eigen::Quaterniond& rotation() const { return r_; }
    const Eigen::Vector3d& translation() const { return t_; }
    double scale() const { return s_; }
    
private:
    Eigen::Quaterniond r_;
    Eigen::Vector3d t_;
    double s_;
};

// 定义关键帧结构
struct KeyFrame {
    int id;
    double timestamp;
    Eigen::Quaterniond rotation;
    Eigen::Vector3d position;
    int parent_id;
    bool is_bad;
    std::set<int> loop_edges;
    std::map<int, int> covisible_keyframes; // KF_ID -> weight
};

// 地图点结构
struct MapPoint {
    int id;
    Eigen::Vector3d position;
    bool is_bad;
    int reference_kf_id;
    int corrected_by_kf;
    int corrected_reference;
};

// 边结构用于优化
struct Edge {
    enum Type {
        SPANNING_TREE,
        LOOP,
        COVISIBILITY,
        NEW_CONNECTION
    };
    
    Type type;
    int source_id;
    int target_id;
    double weight;
    
    // 相对位姿约束
    Eigen::Quaterniond rel_rotation;
    Eigen::Vector3d rel_translation;
    double rel_scale;
};

// Sim3流形函数用于AutoDiffManifold
struct Sim3Manifold {
    Sim3Manifold(bool fix_scale) : fix_scale_(fix_scale) {}
    
    template <typename T>
    bool Plus(const T* x, const T* delta, T* x_plus_delta) const {
        // 每个参数块组织为: [s, qw, qx, qy, qz, tx, ty, tz]
        
        // 提取当前参数
        const T scale = x[0];
        const Eigen::Quaternion<T> rotation(x[1], x[2], x[3], x[4]);  // w, x, y, z
        const Eigen::Matrix<T, 3, 1> translation(x[5], x[6], x[7]);

        // 初始化delta索引
        int delta_idx = 0;
        
        // 更新尺度(如果不固定)
        T new_scale = scale;
        if (!fix_scale_) {
            // 应用乘法尺度更新: s_new = s * exp(delta[0])
            new_scale = scale * exp(delta[delta_idx++]);
        }

        // 使用指数映射更新旋转
        Eigen::Matrix<T, 3, 1> omega(delta[delta_idx], delta[delta_idx+1], delta[delta_idx+2]);
        delta_idx += 3;
        
        Eigen::Quaternion<T> dq;
        const T omega_norm = omega.norm();
        
        if (omega_norm < T(1e-10)) {
            dq = Eigen::Quaternion<T>(T(1.0), T(0.0), T(0.0), T(0.0));
        } else {
            const T theta = omega_norm;
            omega = omega / omega_norm; // 归一化
            dq = Eigen::Quaternion<T>(cos(theta/T(2.0)), 
                                     sin(theta/T(2.0))*omega.x(),
                                     sin(theta/T(2.0))*omega.y(), 
                                     sin(theta/T(2.0))*omega.z());
        }
        
        // 应用旋转更新
        Eigen::Quaternion<T> new_rotation = dq * rotation;
        new_rotation.normalize();

        // 更新平移
        Eigen::Matrix<T, 3, 1> new_translation = translation + 
            Eigen::Matrix<T, 3, 1>(delta[delta_idx], delta[delta_idx+1], delta[delta_idx+2]);

        // 打包更新后的参数
        x_plus_delta[0] = new_scale;
        x_plus_delta[1] = new_rotation.w();
        x_plus_delta[2] = new_rotation.x();
        x_plus_delta[3] = new_rotation.y();
        x_plus_delta[4] = new_rotation.z();
        x_plus_delta[5] = new_translation.x();
        x_plus_delta[6] = new_translation.y();
        x_plus_delta[7] = new_translation.z();

        return true;
    }
    
    template <typename T>
    bool Minus(const T* y, const T* x, T* delta) const {
        // 提取参数
        const T scale_x = x[0];
        const Eigen::Quaternion<T> q_x(x[1], x[2], x[3], x[4]);  // w, x, y, z
        const Eigen::Matrix<T, 3, 1> t_x(x[5], x[6], x[7]);
        
        const T scale_y = y[0];
        const Eigen::Quaternion<T> q_y(y[1], y[2], y[3], y[4]);  // w, x, y, z
        const Eigen::Matrix<T, 3, 1> t_y(y[5], y[6], y[7]);
        
        // 初始化delta索引
        int delta_idx = 0;
        
        // 计算尺度delta(如果不固定)
        if (!fix_scale_) {
            delta[delta_idx++] = log(scale_y / scale_x);
        }
        
        // 计算旋转delta(四元数差)
        const Eigen::Quaternion<T> q_delta = q_x.inverse() * q_y;
        
        // 转换为轴角表示
        T angle = T(2.0) * atan2(q_delta.vec().norm(), q_delta.w());
        Eigen::Matrix<T, 3, 1> axis;
        
        if (q_delta.vec().norm() < T(1e-10)) {
            axis = Eigen::Matrix<T, 3, 1>(T(0.0), T(0.0), T(0.0));
            angle = T(0.0);
        } else {
            axis = q_delta.vec() / q_delta.vec().norm();
        }
        
        // 将角度包裹到[-pi, pi]
        if (angle > T(M_PI)) {
            angle = angle - T(2.0 * M_PI);
        } else if (angle < -T(M_PI)) {
            angle = angle + T(2.0 * M_PI);
        }
        
        // 存储旋转delta为轴角
        delta[delta_idx++] = angle * axis.x();
        delta[delta_idx++] = angle * axis.y();
        delta[delta_idx++] = angle * axis.z();
        
        // 计算平移delta
        const Eigen::Matrix<T, 3, 1> t_delta = t_y - t_x;
        delta[delta_idx++] = t_delta.x();
        delta[delta_idx++] = t_delta.y();
        delta[delta_idx++] = t_delta.z();
        
        return true;
    }
    
    bool fix_scale_;
};

// 针对Ceres的Sim3误差类，匹配g2o::EdgeSim3方法
struct Sim3Error {
    Sim3Error(const Sim3& measurement, double information_factor = 1.0)
        : m_measurement(measurement) {
        // 创建类似于ORB-SLAM3的matLambda的信息矩阵
        m_information = Eigen::Matrix<double,7,7>::Identity() * information_factor;
        
        // 可以根据需要不同地加权各组件
        // 例如，增加旋转权重
        m_information.block<3,3>(0,0) *= 10.0;  // 旋转部分
        // m_information.block<3,3>(3,3) *= 1.0;  // 平移部分
        // m_information(6,6) *= 5.0;  // 尺度部分
    }

    template <typename T>
    bool operator()(const T* const sim3_i_data, const T* const sim3_j_data, T* residuals) const {
        // 每个sim3参数块组织为:
        // [scale, rotation(quaternion), translation]
        // [s, qw, qx, qy, qz, tx, ty, tz]

        // 提取顶点i的参数
        const T scale_i = sim3_i_data[0];
        Eigen::Quaternion<T> rotation_i(sim3_i_data[1], sim3_i_data[2], sim3_i_data[3], sim3_i_data[4]);
        Eigen::Matrix<T, 3, 1> translation_i(sim3_i_data[5], sim3_i_data[6], sim3_i_data[7]);
        
        // 提取顶点j的参数
        const T scale_j = sim3_j_data[0];
        Eigen::Quaternion<T> rotation_j(sim3_j_data[1], sim3_j_data[2], sim3_j_data[3], sim3_j_data[4]);
        Eigen::Matrix<T, 3, 1> translation_j(sim3_j_data[5], sim3_j_data[6], sim3_j_data[7]);

        // 归一化四元数
        rotation_i.normalize();
        rotation_j.normalize();

        // 转换测量值为模板类型
        Eigen::Quaternion<T> meas_rotation(
            T(m_measurement.rotation().w()),
            T(m_measurement.rotation().x()),
            T(m_measurement.rotation().y()),
            T(m_measurement.rotation().z()));
        meas_rotation.normalize();
        
        Eigen::Matrix<T, 3, 1> meas_translation(
            T(m_measurement.translation().x()),
            T(m_measurement.translation().y()),
            T(m_measurement.translation().z()));
        
        T meas_scale = T(m_measurement.scale());

        // 计算相对Sim3变换: Sji_est = Sjw * Swi
        // 这是从i到j的估计变换
        const T s_ji_est = scale_j / scale_i;
        const Eigen::Quaternion<T> q_ji_est = rotation_j * rotation_i.inverse();
        
        // 平移计算: t_ji = (sj/si) * (Rj * Ri^-1 * (-ti)) + tj
        const Eigen::Matrix<T, 3, 1> negated_ti = -translation_i;
        const Eigen::Matrix<T, 3, 1> rotated_ti = rotation_j * (rotation_i.inverse() * negated_ti);
        const Eigen::Matrix<T, 3, 1> t_ji_est = (scale_j / scale_i) * rotated_ti + translation_j;

        // 计算估计与测量变换之间的误差
        // 旋转：error = log(meas_rotation^-1 * q_ji_est)
        const Eigen::Quaternion<T> q_error = meas_rotation.inverse() * q_ji_est;
        
        // 将四元数误差转换为对数（轴角表示）
        Eigen::Matrix<T, 3, 1> omega;
        T angle = T(2.0) * atan2(q_error.vec().norm(), q_error.w());
        if (angle > T(M_PI)) {
            angle -= T(2.0 * M_PI);
        } else if (angle < -T(M_PI)) {
            angle += T(2.0 * M_PI);
        }
        
        if (q_error.vec().norm() > T(1e-10)) {
            omega = (angle / q_error.vec().norm()) * q_error.vec();
        } else {
            omega.setZero();
        }

        // 平移误差: error = t_ji_est - meas_translation
        const Eigen::Matrix<T, 3, 1> t_error = t_ji_est - meas_translation;
        
        // 尺度误差: error = log(s_ji_est / meas_scale)
        const T s_error = log(s_ji_est / meas_scale);

        // 创建完整误差向量（关键部分）
        Eigen::Matrix<T, 7, 1> error;
        error.template head<3>() = omega;  // 旋转误差
        error.template segment<3>(3) = t_error;  // 平移误差
        error(6) = s_error;  // 尺度误差
        
        // 应用信息矩阵，如ORB-SLAM3
        Eigen::Matrix<T, 7, 7> information_t = m_information.template cast<T>();
        Eigen::Matrix<T, 7, 1> weighted_error = information_t * error;
        
        // 将加权误差打包到残差向量中
        for (int i = 0; i < 7; ++i) {
            residuals[i] = weighted_error(i);
        }

        return true;
    }

private:
    const Sim3 m_measurement;
    Eigen::Matrix<double,7,7> m_information;
};

class EssentialGraphOptimizer {
public:
    // 默认构造函数
    EssentialGraphOptimizer() : current_kf_id(0), loop_kf_id(1) {}
    
    // 加载关键帧轨迹
    bool LoadTrajectory(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open trajectory file: " << filename << std::endl;
            return false;
        }
        
        keyframes_.clear();
        
        std::string line;
        int id = 0;
        
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;
            
            std::istringstream iss(line);
            double timestamp_ns, tx, ty, tz, qx, qy, qz, qw;
            
            if (iss >> timestamp_ns >> tx >> ty >> tz >> qx >> qy >> qz >> qw) {
                KeyFrame kf;
                kf.id = id++;
                kf.timestamp = timestamp_ns / 1e9;
                kf.position = Eigen::Vector3d(tx, ty, tz);
                kf.rotation = Eigen::Quaterniond(qw, qx, qy, qz).normalized();
                kf.parent_id = -1;  // 稍后设置
                kf.is_bad = false;
                
                keyframes_[kf.id] = kf;
                timestamp_to_id_[kf.timestamp] = kf.id;
                
                if (kf.id > current_kf_id) {
                    current_kf_id = kf.id;
                }
            }
        }
        
        file.close();
        std::cout << "Loaded " << keyframes_.size() << " keyframes" << std::endl;
        return !keyframes_.empty();
    }
    
    // 加载回环约束
    bool LoadLoopConstraints(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open loop constraints file: " << filename << std::endl;
            return false;
        }
        
        loop_edges_.clear();
        
        std::string line;
        std::getline(file, line); // Skip header
        
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;
            
            std::istringstream iss(line);
            int source_id, target_id;
            std::string constraint_type;
            double r00, r01, r02, r10, r11, r12, r20, r21, r22, t0, t1, t2, scale;
            
            if (iss >> source_id >> target_id >> constraint_type 
                   >> r00 >> r01 >> r02 >> r10 >> r11 >> r12 >> r20 >> r21 >> r22 
                   >> t0 >> t1 >> t2 >> scale) {
                
                Edge edge;
                edge.type = Edge::LOOP;
                edge.source_id = source_id;
                edge.target_id = target_id;
                edge.weight = 5.0; // 回环边更高权重
                
                Eigen::Matrix3d R;
                R << r00, r01, r02,
                     r10, r11, r12,
                     r20, r21, r22;
                
                edge.rel_rotation = Eigen::Quaterniond(R).normalized();
                edge.rel_translation = Eigen::Vector3d(t0, t1, t2);
                edge.rel_scale = scale;
                
                loop_edges_.push_back(edge);
                
                // 添加到关键帧的回环边
                if (keyframes_.find(source_id) != keyframes_.end()) {
                    keyframes_[source_id].loop_edges.insert(target_id);
                }
                if (keyframes_.find(target_id) != keyframes_.end()) {
                    keyframes_[target_id].loop_edges.insert(source_id);
                }
                
                // 添加到回环连接（LoopConnections）
                loop_connections_[source_id].insert(target_id);
                loop_connections_[target_id].insert(source_id);
            }
        }
        
        file.close();
        std::cout << "Loaded " << loop_edges_.size() << " loop constraints" << std::endl;
        return !loop_edges_.empty();
    }
    
    // 加载Essential图
    bool LoadEssentialGraph(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open essential graph file: " << filename << std::endl;
            return false;
        }
        
        spanning_tree_edges_.clear();
        covisibility_edges_.clear();
        
        std::string line;
        std::getline(file, line); // Skip header
        
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;
            
            std::istringstream iss(line);
            std::string edge_type;
            int source_id, target_id;
            double weight, info_scale;
            
            if (iss >> edge_type >> source_id >> target_id >> weight >> info_scale) {
                Edge edge;
                edge.source_id = source_id;
                edge.target_id = target_id;
                edge.weight = weight;
                
                if (edge_type == "SPANNING_TREE") {
                    edge.type = Edge::SPANNING_TREE;
                    // 设置父子关系
                    if (keyframes_.find(source_id) != keyframes_.end() && 
                        keyframes_.find(target_id) != keyframes_.end()) {
                        
                        keyframes_[target_id].parent_id = source_id;
                        
                        const KeyFrame& kf_source = keyframes_[source_id];
                        const KeyFrame& kf_target = keyframes_[target_id];
                        
                        // 从源到目标的相对位姿
                        edge.rel_rotation = kf_source.rotation.conjugate() * kf_target.rotation;
                        edge.rel_translation = kf_source.rotation.conjugate() * 
                                            (kf_target.position - kf_source.position);
                        edge.rel_scale = 1.0;
                        
                        spanning_tree_edges_.push_back(edge);
                    }
                }
                else if (edge_type == "COVISIBILITY") {
                    edge.type = Edge::COVISIBILITY;
                    // 添加共视关系
                    if (keyframes_.find(source_id) != keyframes_.end() && 
                        keyframes_.find(target_id) != keyframes_.end()) {
                        
                        keyframes_[source_id].covisible_keyframes[target_id] = weight;
                        keyframes_[target_id].covisible_keyframes[source_id] = weight;
                        
                        const KeyFrame& kf_source = keyframes_[source_id];
                        const KeyFrame& kf_target = keyframes_[target_id];
                        
                        // 从源到目标的相对位姿
                        edge.rel_rotation = kf_source.rotation.conjugate() * kf_target.rotation;
                        edge.rel_translation = kf_source.rotation.conjugate() * 
                                            (kf_target.position - kf_source.position);
                        edge.rel_scale = 1.0;
                        
                        covisibility_edges_.push_back(edge);
                    }
                }
            }
        }
        
        file.close();
        std::cout << "Loaded " << spanning_tree_edges_.size() << " spanning tree edges and " 
                  << covisibility_edges_.size() << " covisibility edges" << std::endl;
        return true;
    }
    
    // 加载连接变化 - 回环闭合后的新连接
    bool LoadConnectionChanges(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open connection changes file: " << filename << std::endl;
            return false;
        }
        
        connection_changes_.clear();
        
        std::string line;
        std::getline(file, line); // Skip header
        
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;
            
            std::istringstream iss(line);
            int source_id, target_id, is_new;
            
            if (iss >> source_id >> target_id >> is_new) {
                if (is_new == 1) {
                    connection_changes_.push_back(std::make_pair(source_id, target_id));
                    // 添加到回环连接
                    loop_connections_[source_id].insert(target_id);
                    loop_connections_[target_id].insert(source_id);
                }
            }
        }
        
        file.close();
        std::cout << "Loaded " << connection_changes_.size() << " new connections" << std::endl;
        return true;
    }
    
    // 加载Sim3变换
    bool LoadSim3Transformations(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open Sim3 transforms file: " << filename << std::endl;
            return false;
        }
        
        sim3_non_corrected_map_.clear();
        sim3_corrected_map_.clear();
        
        std::string line;
        std::getline(file, line); // Skip header
        
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;
            
            std::istringstream iss(line);
            int kf_id;
            double s_orig, tx_orig, ty_orig, tz_orig, qx_orig, qy_orig, qz_orig, qw_orig;
            double s_corr, tx_corr, ty_corr, tz_corr, qx_corr, qy_corr, qz_corr, qw_corr;
            int propagation_source;
            
            if (iss >> kf_id
                   >> s_orig >> tx_orig >> ty_orig >> tz_orig >> qx_orig >> qy_orig >> qz_orig >> qw_orig
                   >> s_corr >> tx_corr >> ty_corr >> tz_corr >> qx_corr >> qy_corr >> qz_corr >> qw_corr
                   >> propagation_source) {
                
                // 创建未校正的Sim3
                Eigen::Quaterniond q_orig(qw_orig, qx_orig, qy_orig, qz_orig);
                Eigen::Vector3d t_orig(tx_orig, ty_orig, tz_orig);
                Sim3 orig_sim3(q_orig, t_orig, s_orig);
                sim3_non_corrected_map_[kf_id] = orig_sim3;
                
                // 创建校正后的Sim3
                Eigen::Quaterniond q_corr(qw_corr, qx_corr, qy_corr, qz_corr);
                Eigen::Vector3d t_corr(tx_corr, ty_corr, tz_corr);
                Sim3 corr_sim3(q_corr, t_corr, s_corr);
                sim3_corrected_map_[kf_id] = corr_sim3;
            }
        }
        
        file.close();
        std::cout << "Loaded " << sim3_non_corrected_map_.size() << " Sim3 transformations" << std::endl;
        return !sim3_non_corrected_map_.empty();
    }
    
    // 加载地图点
    bool LoadMapPoints(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open map points file: " << filename << std::endl;
            return false;
        }
        
        map_points_.clear();
        
        std::string line;
        std::getline(file, line); // Skip header
        
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;
            
            std::istringstream iss(line);
            int id, ref_kf_id;
            double x, y, z;
            int is_bad;
            
            if (iss >> id >> x >> y >> z >> ref_kf_id >> is_bad) {
                MapPoint mp;
                mp.id = id;
                mp.position = Eigen::Vector3d(x, y, z);
                mp.reference_kf_id = ref_kf_id;
                mp.is_bad = (is_bad == 1);
                mp.corrected_by_kf = -1;
                mp.corrected_reference = -1;
                
                map_points_.push_back(mp);
            }
        }
        
        file.close();
        std::cout << "Loaded " << map_points_.size() << " map points" << std::endl;
        return !map_points_.empty();
    }
    
    // 加载优化参数
    bool LoadOptimizationParams(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open optimization params file: " << filename << std::endl;
            return false;
        }
        
        std::string line;
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;
            
            std::istringstream iss(line);
            std::string key, value;
            
            if (std::getline(iss, key, ':') && std::getline(iss, value)) {
                // 去除前后空格
                key.erase(0, key.find_first_not_of(" \t"));
                key.erase(key.find_last_not_of(" \t") + 1);
                value.erase(0, value.find_first_not_of(" \t"));
                value.erase(value.find_last_not_of(" \t") + 1);
                
                optimization_params_[key] = value;
            }
        }
        
        file.close();
        std::cout << "Loaded optimization parameters" << std::endl;
        return !optimization_params_.empty();
    }
    
    // 获取最大关键帧ID
    int GetMaxKFId() const {
        int max_id = 0;
        for (const auto& kf_pair : keyframes_) {
            if (kf_pair.first > max_id) {
                max_id = kf_pair.first;
            }
        }
        return max_id;
    }
    
    // 优化Essential图 - 核心函数
    bool OptimizeEssentialGraph(const std::string& output_filename, int loop_kf_id = 1, bool fix_scale = true) {
        // 设置Ceres问题
        ceres::Problem problem;
        
        // 设置损失函数 - 使用Huber损失
        double huber_delta = 1.345; // 从optimization_params中获取
        if (optimization_params_.find("HUBER_DELTA") != optimization_params_.end()) {
            huber_delta = std::stod(optimization_params_["HUBER_DELTA"]);
        }
        ceres::LossFunction* loss_function = new ceres::HuberLoss(huber_delta);
        
        // 设置优化参数 - 匹配ORB-SLAM3
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        options.minimizer_progress_to_stdout = true;
        
        // 从optimization_params获取最大迭代次数
        options.max_num_iterations = 20;
        if (optimization_params_.find("NUM_ITERATIONS") != optimization_params_.end()) {
            options.max_num_iterations = std::stoi(optimization_params_["NUM_ITERATIONS"]);
        }
        
        // 设置适当的信任区域半径和收敛标准
        options.initial_trust_region_radius = 1e-4;
        options.max_trust_region_radius = 1e4;
        options.function_tolerance = 1e-4;
        options.gradient_tolerance = 1e-8;
        options.parameter_tolerance = 1e-8;
        
        std::cout << "Setting up optimization problem..." << std::endl;
        
        // 为每个关键帧创建参数块 - 使用Sim3 [scale, rotation, translation]
        std::map<int, double*> sim3_blocks;
        std::map<int, Sim3> vScw;  // 存储原始Sim3 (与ORB-SLAM3类似)
        std::map<int, Sim3> vCorrectedSwc;  // 用于存储校正后的逆Sim3
        
        const int minFeat = 100;  // 共视的最小特征，与ORB-SLAM3相同
        
        // 定义Sim3的环境大小和切空间大小
        const int ambient_size = 8;  // [s, qw, qx, qy, qz, tx, ty, tz]
        const int tangent_size = fix_scale ? 6 : 7;  // 6或7自由度

        // 添加关键帧顶点 - 遵循ORB-SLAM3的方法
        for(auto& kf_pair : keyframes_) {
            int id = kf_pair.first;
            const KeyFrame kf = kf_pair.second; 
            
            if (kf.is_bad) continue;
            
            // 创建Sim3参数块 [s, qw, qx, qy, qz, tx, ty, tz]
            double* sim3_block = new double[8];
            
            // 检查是否是校正过的关键帧
            auto it = sim3_corrected_map_.find(id);
            if(it != sim3_corrected_map_.end()) {
                // 这个关键帧有校正后的Sim3
                const Sim3& corrected_sim3 = it->second;
                
                sim3_block[0] = corrected_sim3.scale();
                sim3_block[1] = corrected_sim3.rotation().w();
                sim3_block[2] = corrected_sim3.rotation().x();
                sim3_block[3] = corrected_sim3.rotation().y();
                sim3_block[4] = corrected_sim3.rotation().z();
                sim3_block[5] = corrected_sim3.translation().x();
                sim3_block[6] = corrected_sim3.translation().y();
                sim3_block[7] = corrected_sim3.translation().z();
                
                vScw[id] = corrected_sim3;
            } else {
                // 使用当前位姿和scale=1.0
                sim3_block[0] = 1.0;  // Scale
                sim3_block[1] = kf.rotation.w();
                sim3_block[2] = kf.rotation.x();
                sim3_block[3] = kf.rotation.y();
                sim3_block[4] = kf.rotation.z();
                sim3_block[5] = kf.position.x();
                sim3_block[6] = kf.position.y();
                sim3_block[7] = kf.position.z();
                
                Sim3 Siw(kf.rotation, kf.position, 1.0);
                vScw[id] = Siw;
            }
            
            sim3_blocks[id] = sim3_block;
            
            // 为这个参数块创建流形
            ceres::Manifold* manifold = NULL;
            if (fix_scale) {
                // 使用6自由度的自动微分流形(固定尺度)
                manifold = new ceres::AutoDiffManifold<Sim3Manifold, 8, 6>(
                    new Sim3Manifold(fix_scale));
            } else {
                // 使用7自由度的自动微分流形(可变尺度)
                manifold = new ceres::AutoDiffManifold<Sim3Manifold, 8, 7>(
                    new Sim3Manifold(fix_scale));
            }
            
            // 将参数块添加到问题中
            problem.AddParameterBlock(sim3_block, 8, manifold);
            
            // 固定初始KF (类似于ORB-SLAM3)
            if(id == loop_kf_id) {
                problem.SetParameterBlockConstant(sim3_block);
            }
        }
        
        // 跟踪插入的边以避免重复 (类似于ORB-SLAM3)
        std::set<std::pair<long unsigned int, long unsigned int>> sInsertedEdges;
        
        // 添加回环边 - 与ORB-SLAM3类似
        std::cout << "Adding loop edges..." << std::endl;
        for(const auto& entry : loop_connections_) {
            int source_id = entry.first;
            const std::set<int>& connected_kfs = entry.second;
            
            if(sim3_blocks.find(source_id) == sim3_blocks.end())
                continue;
                
            const Sim3& Siw = vScw[source_id];
            const Sim3 Swi = Siw.inverse();
            
            for(int target_id : connected_kfs) {
                if(sim3_blocks.find(target_id) == sim3_blocks.end())
                    continue;
                    
                // 跳过已处理的连接
                if(sInsertedEdges.count(std::make_pair(std::min(source_id, target_id), 
                                                    std::max(source_id, target_id))))
                    continue;
                
                const Sim3& Sjw = vScw[target_id];
                const Sim3 Sji = Sjw * Swi;
                
                // 为回环连接添加Sim3边，权重更高
                // 创建信息矩阵 - 类似于ORB-SLAM3的matLambda
                Eigen::Matrix<double,7,7> information = Eigen::Matrix<double,7,7>::Identity() * 5.0;
                
                // 添加带有信息矩阵的Sim3边
                ceres::CostFunction* cost_function = 
                    new ceres::AutoDiffCostFunction<Sim3Error, 7, 8, 8>(
                        new Sim3Error(Sji, 5.0));
                
                problem.AddResidualBlock(cost_function,
                                         loss_function,
                                         sim3_blocks[source_id],
                                         sim3_blocks[target_id]);
                                         
                sInsertedEdges.insert(std::make_pair(std::min(source_id, target_id), 
                                                    std::max(source_id, target_id)));
                
                // 调试输出
                std::cout << "Added loop edge: " << source_id << " -> " << target_id << std::endl;
            }
        }
        
        // 添加生成树边 - 与ORB-SLAM3类似
        std::cout << "Adding spanning tree edges..." << std::endl;
        for(const auto& kf_pair : keyframes_) {
            int target_id = kf_pair.first;
            KeyFrame kf = kf_pair.second; 
            
            if(kf.is_bad || kf.parent_id < 0)
                continue;
                
            int source_id = kf.parent_id;
            
            if(sim3_blocks.find(source_id) == sim3_blocks.end() || 
               sim3_blocks.find(target_id) == sim3_blocks.end())
                continue;
                
            // 跳过已处理的连接
            if(sInsertedEdges.count(std::make_pair(std::min(source_id, target_id), 
                                                std::max(source_id, target_id))))
                continue;
            
            // 获取相对Sim3变换
            const Sim3& Siw = vScw[source_id];
            const Sim3& Sjw = vScw[target_id];
            const Sim3 Sji = Sjw * Siw.inverse();
            
            // 为生成树添加Sim3边，权重较高
            ceres::CostFunction* cost_function = 
                new ceres::AutoDiffCostFunction<Sim3Error, 7, 8, 8>(
                    new Sim3Error(Sji, 3.0));  // 对生成树使用更高权重(3.0)
            
            problem.AddResidualBlock(cost_function,
                                     loss_function,
                                     sim3_blocks[source_id],
                                     sim3_blocks[target_id]);
                                     
            sInsertedEdges.insert(std::make_pair(std::min(source_id, target_id), 
                                                std::max(source_id, target_id)));
            
            // 调试输出(每50条边)
            if (sInsertedEdges.size() % 50 == 0) {
                std::cout << "Added " << sInsertedEdges.size() << " edges so far..." << std::endl;
            }
        }
        
        // 添加回环边(来自之前的回环检测)
        std::cout << "Adding loop constraint edges..." << std::endl;
        for (const auto& edge : loop_edges_) {
            int source_id = edge.source_id;
            int target_id = edge.target_id;
            
            if(sim3_blocks.find(source_id) == sim3_blocks.end() || 
               sim3_blocks.find(target_id) == sim3_blocks.end() ||
               keyframes_[source_id].is_bad || keyframes_[target_id].is_bad)
                continue;
                
            // 跳过已处理的连接
            if(sInsertedEdges.count(std::make_pair(std::min(source_id, target_id), 
                                                  std::max(source_id, target_id))))
                continue;
            
            // 从边数据创建Sim3
            Sim3 Sji(edge.rel_rotation, edge.rel_translation, edge.rel_scale);
            
            // 添加Sim3边，使用回环边权重
            ceres::CostFunction* cost_function = 
                new ceres::AutoDiffCostFunction<Sim3Error, 7, 8, 8>(
                    new Sim3Error(Sji, 5.0));  // 对回环边使用更高权重(5.0)
            
            problem.AddResidualBlock(cost_function,
                                     loss_function,
                                     sim3_blocks[source_id],
                                     sim3_blocks[target_id]);
                                     
            sInsertedEdges.insert(std::make_pair(std::min(source_id, target_id), 
                                                std::max(source_id, target_id)));
            
            // 调试输出
            std::cout << "Added loop constraint edge: " << source_id << " -> " << target_id << std::endl;
        }
        
        // 添加共视边
        std::cout << "Adding covisibility edges..." << std::endl;
        for(const auto& kf_pair : keyframes_) {
            int source_id = kf_pair.first;
            KeyFrame kf = kf_pair.second; 
            
            if(kf.is_bad) continue;
            
            for(const auto& cov_pair : kf.covisible_keyframes) {
                int target_id = cov_pair.first;
                int weight = cov_pair.second;
                
                // 每条边只处理一次
                if(target_id < source_id) continue;
                
                // 跳过坏的目标KF
                if(keyframes_.find(target_id) == keyframes_.end() || 
                   keyframes_[target_id].is_bad)
                    continue;
                
                // 跳过已有边的KF
                if(sInsertedEdges.count(std::make_pair(source_id, target_id)))
                    continue;
                    
                // 跳过权重过低的
                if(weight < minFeat)
                    continue;
                    
                if(sim3_blocks.find(source_id) == sim3_blocks.end() || 
                   sim3_blocks.find(target_id) == sim3_blocks.end())
                    continue;
                    
                // 获取相对Sim3变换
                const Sim3& Siw = vScw[source_id];
                const Sim3& Sjw = vScw[target_id];
                const Sim3 Sji = Sjw * Siw.inverse();
                
                // 添加适当权重的Sim3边用于共视
                double information_factor = 1.0;  // 标准共视权重
                
                ceres::CostFunction* cost_function = 
                    new ceres::AutoDiffCostFunction<Sim3Error, 7, 8, 8>(
                        new Sim3Error(Sji, information_factor));
                
                problem.AddResidualBlock(cost_function,
                                         loss_function,
                                         sim3_blocks[source_id],
                                         sim3_blocks[target_id]);
                                         
                sInsertedEdges.insert(std::make_pair(source_id, target_id));
                
                // 调试输出(每200条边)
                if (sInsertedEdges.size() % 200 == 0) {
                    std::cout << "Added " << sInsertedEdges.size() << " edges so far..." << std::endl;
                }
            }
        }
        
        // 添加连接变化边
        std::cout << "Adding connection changes..." << std::endl;
        for(const auto& conn : connection_changes_) {
            int source_id = conn.first;
            int target_id = conn.second;
            
            // 跳过已有边的KF
            if(sInsertedEdges.count(std::make_pair(std::min(source_id, target_id), 
                                                  std::max(source_id, target_id))))
                continue;
                
            if(sim3_blocks.find(source_id) == sim3_blocks.end() || 
               sim3_blocks.find(target_id) == sim3_blocks.end())
                continue;
                
            // 计算相对Sim3变换
            const Sim3& Siw = vScw[source_id];
            const Sim3& Sjw = vScw[target_id];
            const Sim3 Sji = Sjw * Siw.inverse();
            
            // 添加新连接的Sim3边
            ceres::CostFunction* cost_function = 
                new ceres::AutoDiffCostFunction<Sim3Error, 7, 8, 8>(
                    new Sim3Error(Sji, 2.0));  // 新连接的权重=2.0
            
            problem.AddResidualBlock(cost_function,
                                     loss_function,
                                     sim3_blocks[source_id],
                                     sim3_blocks[target_id]);
                                     
            sInsertedEdges.insert(std::make_pair(std::min(source_id, target_id), 
                                                std::max(source_id, target_id)));
            
            // 调试输出
            std::cout << "Added connection change edge: " << source_id << " -> " << target_id << std::endl;
        }
        
        std::cout << "Total edges added: " << sInsertedEdges.size() << std::endl;
        
        // 检查初始误差
        std::vector<double> residuals;
        double initial_cost = 0.0;
        problem.Evaluate(ceres::Problem::EvaluateOptions(), &initial_cost, &residuals, nullptr, nullptr);
        std::cout << "Initial cost: " << initial_cost << " with " << residuals.size() << " residuals" << std::endl;
        
        // 求解问题
        std::cout << "Solving optimization problem..." << std::endl;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        
        std::cout << summary.FullReport() << std::endl;
        
        if (summary.termination_type == ceres::FAILURE) {
            std::cerr << "Optimization failed!" << std::endl;
            // 无论如何都清理
            for(auto& block_pair : sim3_blocks) {
                delete[] block_pair.second;
            }
            return false;
        }
        
        // SE3位姿恢复 (与ORB-SLAM3类似)
        // Sim3:[s,R,t] -> SE3:[R,t/s]
        for(auto& kf_pair : keyframes_) {
            int id = kf_pair.first;
            KeyFrame& kf = kf_pair.second;
            
            if(sim3_blocks.find(id) == sim3_blocks.end())
                continue;
                
            const double* sim3_block = sim3_blocks[id];
            double s = sim3_block[0];
            
            // 存储校正后的Sim3用于地图点校正
            Eigen::Quaterniond corrected_rot(sim3_block[1], sim3_block[2], sim3_block[3], sim3_block[4]);
            Eigen::Vector3d corrected_trans(sim3_block[5], sim3_block[6], sim3_block[7]);
            Sim3 CorrectedSiw(corrected_rot, corrected_trans, s);
            vCorrectedSwc[id] = CorrectedSiw.inverse();
            
            // 转换为SE3，除以尺度
            kf.rotation = Eigen::Quaterniond(sim3_block[1], sim3_block[2], sim3_block[3], sim3_block[4]).normalized();
            kf.position = Eigen::Vector3d(sim3_block[5]/s, sim3_block[6]/s, sim3_block[7]/s);
        }
        
        // 校正地图点 (与ORB-SLAM3类似)
        if (!map_points_.empty()) {
            std::cout << "Correcting map points..." << std::endl;
            
            for(MapPoint& mp : map_points_) {
                if(mp.is_bad)
                    continue;
                
                // 获取参考关键帧
                int nIDr;
                if(mp.corrected_by_kf == current_kf_id) {
                    nIDr = mp.corrected_reference;
                } else {
                    nIDr = mp.reference_kf_id;
                }
                
                if(vScw.find(nIDr) == vScw.end() || vCorrectedSwc.find(nIDr) == vCorrectedSwc.end())
                    continue;
                    
                // 获取原始和校正后的Sim3变换
                Sim3 Srw = vScw[nIDr];
                Sim3 correctedSwr = vCorrectedSwc[nIDr];
                
                // 将点从世界到参考KF变换，然后用校正后的位姿变换回来
                Eigen::Vector3d eigP3Dw = mp.position;
                Eigen::Vector3d eigCorrectedP3Dw = correctedSwr.map(Srw.map(eigP3Dw));
                
                // 更新地图点位置
                mp.position = eigCorrectedP3Dw;
            }
        }
        
        // 清理
        for(auto& block_pair : sim3_blocks) {
            delete[] block_pair.second;
        }
        
        // 保存优化后的轨迹
        SaveTrajectory(output_filename);
        
        return true;
    }
    
    // 保存轨迹
    void SaveTrajectory(const std::string& filename) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open output file: " << filename << std::endl;
            return;
        }
        
        // 按时间戳创建有序关键帧
        std::vector<const KeyFrame*> ordered_kfs;
        for (const auto& kf_pair : keyframes_) {
            if (!kf_pair.second.is_bad) {
                ordered_kfs.push_back(&kf_pair.second);
            }
        }
        
        std::sort(ordered_kfs.begin(), ordered_kfs.end(),
                 [](const KeyFrame* a, const KeyFrame* b) {
                     return a->timestamp < b->timestamp;
                 });
        
        for (const KeyFrame* kf : ordered_kfs) {
            file << std::fixed << std::setprecision(9)
                 << kf->timestamp * 1e9 << " "
                 << kf->position.x() << " "
                 << kf->position.y() << " "
                 << kf->position.z() << " "
                 << kf->rotation.x() << " "
                 << kf->rotation.y() << " "
                 << kf->rotation.z() << " "
                 << kf->rotation.w() << std::endl;
        }
        
        file.close();
        std::cout << "Saved optimized trajectory to: " << filename << std::endl;
    }
    
    // 保存优化后的地图点
    void SaveMapPoints(const std::string& filename) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open output file: " << filename << std::endl;
            return;
        }
        
        file << "# ID X Y Z RefKF IsBad CorrectedByKF CorrectedRef" << std::endl;
        
        for (const MapPoint& mp : map_points_) {
            if (mp.is_bad) continue;
            
            file << mp.id << " "
                 << mp.position.x() << " "
                 << mp.position.y() << " "
                 << mp.position.z() << " "
                 << mp.reference_kf_id << " "
                 << (mp.is_bad ? 1 : 0) << " "
                 << mp.corrected_by_kf << " "
                 << mp.corrected_reference << std::endl;
        }
        
        file.close();
        std::cout << "Saved optimized map points to: " << filename << std::endl;
    }
    
    // 成员变量
    std::map<int, KeyFrame> keyframes_;
    std::map<double, int> timestamp_to_id_;
    std::vector<Edge> loop_edges_;
    std::vector<Edge> spanning_tree_edges_;
    std::vector<Edge> covisibility_edges_;
    std::vector<std::pair<int, int>> connection_changes_;
    std::vector<MapPoint> map_points_;
    std::map<std::string, std::string> optimization_params_;
    
    // Sim3变换的映射 (类似于ORB-SLAM3的NonCorrectedSim3和CorrectedSim3)
    std::map<int, Sim3> sim3_non_corrected_map_;
    std::map<int, Sim3> sim3_corrected_map_;
    
    // 回环连接 (类似于ORB-SLAM3的LoopConnections)
    std::map<int, std::set<int>> loop_connections_;
    
    // 当前KF和回环KF ID
    int current_kf_id;
    int loop_kf_id;
};

int main(int argc, char** argv) {
    std::string input_dir = "/Datasets/CERES_Work/input";
    std::string output_file = "/Datasets/CERES_Work/build/ceres_optimized_trajectory.txt";
    
    if (argc > 1) input_dir = argv[1];
    if (argc > 2) output_file = argv[2];
    
    EssentialGraphOptimizer optimizer;
    
    // 加载Sim3变换后的轨迹(这是Sim3变换后但优化前的输入数据)
    std::cout << "Loading Sim3 transformed trajectory..." << std::endl;
    if (!optimizer.LoadTrajectory(input_dir + "/sim3_transformed_trajectory.txt")) {
        std::cerr << "Failed to load trajectory!" << std::endl;
        return 1;
    }
    
    // 加载约束和图结构(Essential图)
    std::cout << "Loading constraints..." << std::endl;
    optimizer.LoadLoopConstraints(input_dir + "/metadata/loop_constraints.txt");
    optimizer.LoadEssentialGraph(input_dir + "/pre/essential_graph.txt");
    optimizer.LoadConnectionChanges(input_dir + "/metadata/connection_changes.txt");
    
    // 尝试加载Sim3变换，但如果不可用则继续
    std::cout << "Loading Sim3 transformations..." << std::endl;
    optimizer.LoadSim3Transformations(input_dir + "/metadata/sim3_transforms.txt");
    
    // 加载优化参数
    std::cout << "Loading optimization parameters..." << std::endl;
    optimizer.LoadOptimizationParams(input_dir + "/metadata/optimization_params.txt");
    
    // 尝试加载地图点，但如果不可用则继续
    std::cout << "Loading map points..." << std::endl;
    try {
        optimizer.LoadMapPoints(input_dir + "/transformed/mappoints_sim3_transformed.txt");
    } catch (const std::exception& e) {
        std::cerr << "Warning: Failed to load map points: " << e.what() << std::endl;
        std::cerr << "Continuing without map points..." << std::endl;
    }
    
    // 设置回环关键帧ID(从loop_info.txt元数据读取)
    optimizer.loop_kf_id = 1; // 根据您提供的loop_info.txt的默认值
    optimizer.current_kf_id = optimizer.GetMaxKFId();
    
    std::cout << "Key information before optimization:" << std::endl;
    std::cout << "Current KF ID: " << optimizer.current_kf_id << std::endl;
    std::cout << "Loop KF ID: " << optimizer.loop_kf_id << std::endl;
    std::cout << "Total keyframes: " << optimizer.keyframes_.size() << std::endl;
    std::cout << "Total loop connections: " << optimizer.loop_connections_.size() << std::endl;
    
    // 优化Essential图
    std::cout << "Optimizing essential graph..." << std::endl;
    bool fix_scale = true; // 从optimization_params获取
    if (optimizer.optimization_params_.find("FIXED_SCALE") != optimizer.optimization_params_.end()) {
        fix_scale = (optimizer.optimization_params_["FIXED_SCALE"] == "true");
    }
    optimizer.OptimizeEssentialGraph(output_file, optimizer.loop_kf_id, fix_scale);
    
    // 如果加载了地图点，则保存更新后的地图点
    if (!optimizer.map_points_.empty()) {
        std::cout << "Saving optimized map points..." << std::endl;
        optimizer.SaveMapPoints(output_file.substr(0, output_file.find_last_of('.')) + "_mappoints.txt");
    }
    
    std::cout << "Essential Graph optimization completed!" << std::endl;
    return 0;
}
