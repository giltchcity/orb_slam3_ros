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
#include <cstring>
#include <sys/stat.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <ceres/ceres.h>

// Sim3 class that replicates g2o's behavior exactly
class Sim3 {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // Default constructor - identity transform
    Sim3() : 
        q_(Eigen::Quaterniond::Identity()),
        t_(Eigen::Vector3d::Zero()),
        s_(1.0) {}
    
    // Constructor from rotation, translation, and scale
    Sim3(const Eigen::Quaterniond& q, const Eigen::Vector3d& t, double s) : 
        q_(q), t_(t), s_(s) {
        q_.normalize();
    }
    
    // Get rotation as quaternion
    inline const Eigen::Quaterniond& quaternion() const { return q_; }
    
    // Get rotation as matrix
    inline Eigen::Matrix3d rotation() const { return q_.toRotationMatrix(); }
    
    // Get translation vector
    inline const Eigen::Vector3d& translation() const { return t_; }
    
    // Get scale factor
    inline double scale() const { return s_; }
    
    // Apply transformation to a 3D point (s * R * p + t)
    Eigen::Vector3d map(const Eigen::Vector3d& p) const {
        return s_ * (q_ * p) + t_;
    }
    
    // Inverse transformation
    Sim3 inverse() const {
        Eigen::Quaterniond q_inv = q_.conjugate();
        Eigen::Vector3d t_inv = -(q_inv * t_) / s_;
        return Sim3(q_inv, t_inv, 1.0/s_);
    }
    
    // Right multiplication operator - CRITICAL for matching g2o's behavior
    Sim3 operator*(const Sim3& other) const {
        Eigen::Quaterniond q_res = q_ * other.quaternion();
        Eigen::Vector3d t_res = s_ * (q_ * other.translation()) + t_;
        double s_res = s_ * other.scale();
        return Sim3(q_res, t_res, s_res);
    }

private:
    Eigen::Quaterniond q_;  // Rotation as quaternion
    Eigen::Vector3d t_;     // Translation vector
    double s_;              // Scale factor
};

// KeyFrame structure - Simplified for trajectory-only optimization
struct KeyFrame {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    int id;                        // KeyFrame ID
    double timestamp;              // Timestamp
    Eigen::Quaterniond rotation;   // Rotation as quaternion
    Eigen::Vector3d position;      // Position
    
    // For the optimization
    bool is_bad = false;           // Is this a bad KF
    int parent_id = -1;            // Parent ID
    std::set<int> loop_edges;      // Loop edge connections
    std::map<int, int> covisible_keyframes; // KF_ID -> weight
};

class Sim3Manifold : public ceres::Manifold {
public:
    Sim3Manifold(bool fix_scale = true) : fix_scale_(fix_scale) {}
    
    virtual ~Sim3Manifold() {}
    
    // 参数块格式: [tx, ty, tz, qx, qy, qz, qw, s]
    virtual int AmbientSize() const { return 8; }
    
    // 局部切空间维度
    virtual int TangentSize() const { return fix_scale_ ? 6 : 7; }
    
    // 这是确保与g2o行为一致的关键函数
    virtual bool Plus(const double* x, const double* delta, double* x_plus_delta) const {
        // 提取当前参数
        Eigen::Vector3d t(x[0], x[1], x[2]);
        Eigen::Quaterniond q(x[6], x[3], x[4], x[5]); // w, x, y, z顺序
        q.normalize();
        double s = x[7];
        
        // 提取增量参数
        Eigen::Vector3d omega(delta[0], delta[1], delta[2]); // 旋转增量（轴角表示）
        Eigen::Vector3d dt(delta[3], delta[4], delta[5]);    // 平移增量
        double ds = fix_scale_ ? 0.0 : (TangentSize() > 6 ? delta[6] : 0.0); // 尺度增量
        
        // 从轴角计算四元数增量
        double theta = omega.norm();
        Eigen::Quaterniond dq;
        
        if (theta > 1e-10) {
            omega = omega / theta; // 归一化
            dq = Eigen::Quaterniond(cos(theta/2.0), 
                                   sin(theta/2.0) * omega.x(),
                                   sin(theta/2.0) * omega.y(), 
                                   sin(theta/2.0) * omega.z());
        } else {
            dq = Eigen::Quaterniond(1.0, 0.0, 0.0, 0.0);
        }
        
        // 右乘更新 - 这与ORB-SLAM3的g2o实现匹配
        Eigen::Quaterniond q_plus = q * dq;
        q_plus.normalize();
        
        // 尺度更新是乘法（不是加法）
        double s_plus = s * exp(ds);
        
        // 稳定性检查
        if (!std::isfinite(s_plus)) s_plus = s;
        
        // 平移更新是加法
        Eigen::Vector3d t_plus = t + dt;
        
        // 稳定性检查
        for (int i = 0; i < 3; i++) {
            if (!std::isfinite(t_plus[i])) t_plus[i] = t[i];
            if (std::abs(t_plus[i]) > 1e10) t_plus[i] = t[i]; // 避免极端值
        }
        
        // 存储更新后的值
        x_plus_delta[0] = t_plus.x();
        x_plus_delta[1] = t_plus.y();
        x_plus_delta[2] = t_plus.z();
        x_plus_delta[3] = q_plus.x();
        x_plus_delta[4] = q_plus.y();
        x_plus_delta[5] = q_plus.z();
        x_plus_delta[6] = q_plus.w();
        x_plus_delta[7] = s_plus;
        
        return true;
    }
    
    // PlusJacobian必须与Plus一致
    virtual bool PlusJacobian(const double* x, double* jacobian) const {
        // 初始化为零
        std::memset(jacobian, 0, sizeof(double) * 8 * TangentSize());
        
        Eigen::Map<Eigen::Matrix<double, 8, Eigen::Dynamic, Eigen::RowMajor>> J(
            jacobian, 8, TangentSize());
        
        // 旋转的Jacobian
        J.block<3, 3>(3, 0) = Eigen::Matrix3d::Identity(); // dq/domega
        
        // 平移的Jacobian
        J.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity(); // dt/ddt
        
        // 尺度的Jacobian
        if (!fix_scale_ && TangentSize() > 6) {
            J(7, 6) = x[7]; // ds/dds = s (因为s_new = s * exp(ds))
        }
        
        return true;
    }
    
    // Ceres 2.2需要的辅助函数
    virtual bool Minus(const double* y, const double* x, double* delta) const {
        // 提取参数
        Eigen::Vector3d t_x(x[0], x[1], x[2]);
        Eigen::Quaterniond q_x(x[6], x[3], x[4], x[5]); // w, x, y, z
        q_x.normalize();
        double s_x = x[7];
        
        Eigen::Vector3d t_y(y[0], y[1], y[2]);
        Eigen::Quaterniond q_y(y[6], y[3], y[4], y[5]); // w, x, y, z
        q_y.normalize();
        double s_y = y[7];
        
        // 计算旋转差异
        Eigen::Quaterniond q_delta = q_x.conjugate() * q_y;
        q_delta.normalize();
        
        // 转换为轴角
        Eigen::Vector3d omega;
        double angle = 2.0 * atan2(q_delta.vec().norm(), q_delta.w());
        
        if (q_delta.vec().norm() > 1e-10) {
            omega = angle * q_delta.vec() / q_delta.vec().norm();
        } else {
            omega.setZero();
        }
        
        // 计算平移差异
        Eigen::Vector3d dt = t_y - t_x;
        
        // 计算尺度差异
        double ds = log(s_y / s_x);
        
        // 填充增量向量
        delta[0] = omega.x();
        delta[1] = omega.y();
        delta[2] = omega.z();
        delta[3] = dt.x();
        delta[4] = dt.y();
        delta[5] = dt.z();
        
        if (!fix_scale_ && TangentSize() > 6) {
            delta[6] = ds;
        }
        
        return true;
    }
    
    virtual bool MinusJacobian(const double* x, double* jacobian) const {
        std::memset(jacobian, 0, sizeof(double) * TangentSize() * 8);
        
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 8, Eigen::RowMajor>> J(
            jacobian, TangentSize(), 8);
        
        // 旋转
        J.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity();
        
        // 平移
        J.block<3, 3>(3, 0) = Eigen::Matrix3d::Identity();
        
        // 尺度
        if (!fix_scale_ && TangentSize() > 6) {
            J(6, 7) = 1.0 / x[7]; // dds/ds = 1/s
        }
        
        return true;
    }

private:
    bool fix_scale_;  // 是否固定尺度
};

struct Sim3Error {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Sim3Error(const Sim3& measurement) : measurement_(measurement) {}

    template <typename T>
    bool operator()(const T* const param_i, const T* const param_j, T* residuals) const {
        // 提取参数
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> t_i(param_i);
        Eigen::Quaternion<T> q_i(param_i[6], param_i[3], param_i[4], param_i[5]); // w, x, y, z
        q_i.normalize();
        T s_i = param_i[7];
        
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> t_j(param_j);
        Eigen::Quaternion<T> q_j(param_j[6], param_j[3], param_j[4], param_j[5]); // w, x, y, z
        q_j.normalize();
        T s_j = param_j[7];
        
        // 转换测量值到模板类型T
        Eigen::Quaternion<T> q_meas(
            T(measurement_.quaternion().w()),
            T(measurement_.quaternion().x()),
            T(measurement_.quaternion().y()),
            T(measurement_.quaternion().z()));
        q_meas.normalize();
        
        Eigen::Matrix<T, 3, 1> t_meas(
            T(measurement_.translation().x()),
            T(measurement_.translation().y()),
            T(measurement_.translation().z()));
        
        T s_meas = T(measurement_.scale());
        
        // 计算相对Sim3变换: Sji = Sj * Si^-1
        // 首先计算Si^-1
        Eigen::Quaternion<T> q_i_inv = q_i.conjugate();
        Eigen::Matrix<T, 3, 1> t_i_inv = -(q_i_inv * t_i) / s_i;
        T s_i_inv = T(1.0) / s_i;
        
        // 然后计算Sji = Sj * Si^-1
        Eigen::Quaternion<T> q_ji = q_j * q_i_inv;
        q_ji.normalize();
        
        Eigen::Matrix<T, 3, 1> t_ji = s_j * (q_j * t_i_inv) + t_j;
        T s_ji = s_j * s_i_inv;
        
        // 计算测量值与估计值之间的误差
        // 旋转误差: log(q_meas^-1 * q_ji)
        Eigen::Quaternion<T> q_error = q_meas.conjugate() * q_ji;
        q_error.normalize();
        
        // 转换四元数误差到轴角表示
        Eigen::Matrix<T, 3, 1> omega;
        T angle = T(2.0) * atan2(q_error.vec().norm(), q_error.w());
        
        if (q_error.vec().norm() > T(1e-10)) {
            omega = angle * q_error.vec() / q_error.vec().norm();
        } else {
            omega.setZero();
        }
        
        // 平移误差: t_ji - t_meas
        Eigen::Matrix<T, 3, 1> t_error = t_ji - t_meas;
        
        // 尺度误差: log(s_ji / s_meas)
        T s_error = log(s_ji / s_meas);
        
        // 构造7维误差向量 [omega, t_error, s_error]
        residuals[0] = omega.x();
        residuals[1] = omega.y();
        residuals[2] = omega.z();
        residuals[3] = t_error.x();
        residuals[4] = t_error.y();
        residuals[5] = t_error.z();
        residuals[6] = s_error;
        
        return true;
    }

    static ceres::CostFunction* Create(const Sim3& measurement) {
        return new ceres::AutoDiffCostFunction<Sim3Error, 7, 8, 8>(
            new Sim3Error(measurement));
    }

private:
    Sim3 measurement_;
};


class TrajectoryOptimizer {
public:
    // Constructor
    TrajectoryOptimizer() 
        : current_kf_id(0), loop_kf_id(1), init_kf_id(0) {}
    
    // Load keyframe trajectory
    bool LoadTrajectory(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open trajectory file: " << filename << std::endl;
            return false;
        }
        
        keyframes_.clear();
        
        std::string line;
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;
            
            std::istringstream iss(line);
            double timestamp, tx, ty, tz, qx, qy, qz, qw;
            int id = -1;
            
            if (iss >> id >> timestamp >> tx >> ty >> tz >> qx >> qy >> qz >> qw) {
                // This format has ID as the first column
                KeyFrame kf;
                kf.id = id;
                kf.timestamp = timestamp;
                kf.position = Eigen::Vector3d(tx, ty, tz);
                kf.rotation = Eigen::Quaterniond(qw, qx, qy, qz).normalized();
                kf.is_bad = false;
                
                keyframes_[kf.id] = kf;
                if (kf.id > current_kf_id) {
                    current_kf_id = kf.id;
                }
            } else {
                // Try the other format with just timestamp
                iss.clear();
                iss.str(line);
                
                if (iss >> timestamp >> tx >> ty >> tz >> qx >> qy >> qz >> qw) {
                    KeyFrame kf;
                    kf.id = keyframes_.size();
                    kf.timestamp = timestamp;
                    kf.position = Eigen::Vector3d(tx, ty, tz);
                    kf.rotation = Eigen::Quaterniond(qw, qx, qy, qz).normalized();
                    kf.is_bad = false;
                    
                    keyframes_[kf.id] = kf;
                    if (kf.id > current_kf_id) {
                        current_kf_id = kf.id;
                    }
                }
            }
        }
        
        file.close();
        std::cout << "Loaded " << keyframes_.size() << " keyframes from " << filename << std::endl;
        return !keyframes_.empty();
    }
    
    // Load loop constraints
    bool LoadLoopConstraints(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open loop constraints file: " << filename << std::endl;
            return false;
        }
        
        std::string line;
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;
            
            std::istringstream iss(line);
            int source_id, target_id;
            std::string constraint_type;
            double r00, r01, r02, r10, r11, r12, r20, r21, r22, t0, t1, t2, scale;
            
            if (iss >> source_id >> target_id >> constraint_type 
                   >> r00 >> r01 >> r02 >> r10 >> r11 >> r12 >> r20 >> r21 >> r22 
                   >> t0 >> t1 >> t2 >> scale) {
                
                Eigen::Matrix3d rot;
                rot << r00, r01, r02,
                       r10, r11, r12,
                       r20, r21, r22;
                       
                Eigen::Quaterniond q(rot);
                Eigen::Vector3d t(t0, t1, t2);
                
                Sim3 constraint(q, t, scale);
                
                // Store the constraint
                loop_constraints_.push_back(std::make_tuple(source_id, target_id, constraint));
                
                // Add to KeyFrame loop edges
                if (keyframes_.find(source_id) != keyframes_.end()) {
                    keyframes_[source_id].loop_edges.insert(target_id);
                }
                if (keyframes_.find(target_id) != keyframes_.end()) {
                    keyframes_[target_id].loop_edges.insert(source_id);
                }
                
                // Add to loop connections
                loop_connections_[source_id].insert(target_id);
            }
        }
        
        file.close();
        std::cout << "Loaded " << loop_constraints_.size() << " loop constraints" << std::endl;
        return !loop_constraints_.empty();
    }
    
    // Load essential graph
    bool LoadEssentialGraph(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open essential graph file: " << filename << std::endl;
            return false;
        }
        
        std::string line;
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;
            
            std::istringstream iss(line);
            std::string edge_type;
            int source_id, target_id;
            double weight, info_scale;
            
            if (iss >> edge_type >> source_id >> target_id >> weight >> info_scale) {
                if (edge_type == "SPANNING_TREE") {
                    // Set parent relationship
                    if (keyframes_.find(source_id) != keyframes_.end() && 
                        keyframes_.find(target_id) != keyframes_.end()) {
                        keyframes_[source_id].parent_id = target_id;
                        spanning_tree_edges_.push_back(std::make_pair(source_id, target_id));
                    }
                } else if (edge_type == "COVISIBILITY") {
                    // Add covisibility relationship
                    if (keyframes_.find(source_id) != keyframes_.end() && 
                        keyframes_.find(target_id) != keyframes_.end()) {
                        keyframes_[source_id].covisible_keyframes[target_id] = weight;
                        covisibility_edges_.push_back(std::make_tuple(source_id, target_id, weight));
                    }
                } else if (edge_type == "LOOP") {
                    // Add loop relationship
                    if (keyframes_.find(source_id) != keyframes_.end() && 
                        keyframes_.find(target_id) != keyframes_.end()) {
                        keyframes_[source_id].loop_edges.insert(target_id);
                        keyframes_[target_id].loop_edges.insert(source_id);
                    }
                }
            }
        }
        
        file.close();
        std::cout << "Loaded essential graph with " << spanning_tree_edges_.size() 
                  << " spanning tree edges and " << covisibility_edges_.size() 
                  << " covisibility edges" << std::endl;
        return true;
    }
    
    // Load connection changes
    bool LoadConnectionChanges(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open connection changes file: " << filename << std::endl;
            return false;
        }
        
        std::string line;
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;
            
            std::istringstream iss(line);
            int source_id, target_id, is_new;
            
            if (iss >> source_id >> target_id >> is_new) {
                if (is_new == 1) {
                    connection_changes_.push_back(std::make_pair(source_id, target_id));
                    
                    // Add to loop connections
                    loop_connections_[source_id].insert(target_id);
                    loop_connections_[target_id].insert(source_id);
                }
            }
        }
        
        file.close();
        std::cout << "Loaded " << connection_changes_.size() << " connection changes" << std::endl;
        return true;
    }
    
    // Load Sim3 transformations
    bool LoadSim3Transformations(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open Sim3 transforms file: " << filename << std::endl;
            return false;
        }
        
        std::string line;
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;
            
            std::istringstream iss(line);
            int kf_id;
            double orig_scale, orig_tx, orig_ty, orig_tz, orig_qx, orig_qy, orig_qz, orig_qw;
            double corr_scale, corr_tx, corr_ty, corr_tz, corr_qx, corr_qy, corr_qz, corr_qw;
            int propagation_source;
            
            if (iss >> kf_id 
                   >> orig_scale >> orig_tx >> orig_ty >> orig_tz >> orig_qx >> orig_qy >> orig_qz >> orig_qw 
                   >> corr_scale >> corr_tx >> corr_ty >> corr_tz >> corr_qx >> corr_qy >> corr_qz >> corr_qw 
                   >> propagation_source) {
                
                // Original Sim3
                Eigen::Quaterniond orig_q(orig_qw, orig_qx, orig_qy, orig_qz);
                Eigen::Vector3d orig_t(orig_tx, orig_ty, orig_tz);
                Sim3 orig_sim3(orig_q, orig_t, orig_scale);
                
                // Corrected Sim3
                Eigen::Quaterniond corr_q(corr_qw, corr_qx, corr_qy, corr_qz);
                Eigen::Vector3d corr_t(corr_tx, corr_ty, corr_tz);
                Sim3 corr_sim3(corr_q, corr_t, corr_scale);
                
                // Store
                non_corrected_sim3_[kf_id] = orig_sim3;
                corrected_sim3_[kf_id] = corr_sim3;
            }
        }
        
        file.close();
        std::cout << "Loaded " << non_corrected_sim3_.size() << " Sim3 transformations" << std::endl;
        return !non_corrected_sim3_.empty();
    }
    
    // Load loop info
    bool LoadLoopInfo(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open loop info file: " << filename << std::endl;
            return false;
        }
        
        std::string line;
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;
            
            std::istringstream iss(line);
            std::string key;
            std::string value;
            
            if (std::getline(iss, key, ':') && std::getline(iss, value)) {
                // Trim whitespace
                key.erase(0, key.find_first_not_of(" \t"));
                key.erase(key.find_last_not_of(" \t") + 1);
                value.erase(0, value.find_first_not_of(" \t"));
                value.erase(value.find_last_not_of(" \t") + 1);
                
                if (key == "CURRENT_KF_ID") {
                    current_kf_id = std::stoi(value);
                } else if (key == "MATCHED_KF_ID") {
                    loop_kf_id = std::stoi(value);
                } else if (key == "FIXED_SCALE") {
                    // Store in optimization parameters
                    optimization_params["FIXED_SCALE"] = value;
                }
            }
        }
        
        file.close();
        std::cout << "Loaded loop info: current_kf=" << current_kf_id << ", loop_kf=" << loop_kf_id << std::endl;
        return true;
    }
    
    // Load optimization parameters
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
                // Trim whitespace
                key.erase(0, key.find_first_not_of(" \t"));
                key.erase(key.find_last_not_of(" \t") + 1);
                value.erase(0, value.find_first_not_of(" \t"));
                value.erase(value.find_last_not_of(" \t") + 1);
                
                optimization_params[key] = value;
            }
        }
        
        file.close();
        std::cout << "Loaded optimization parameters" << std::endl;
        return !optimization_params.empty();
    }
    
    bool OptimizeEssentialGraph(const std::string& output_file) {
        // 确定是否应该固定尺度
        bool fix_scale = true;  // 默认固定尺度
        if (optimization_params.find("FIXED_SCALE") != optimization_params.end()) {
            fix_scale = (optimization_params["FIXED_SCALE"] == "true");
        }
        
        // 设置Ceres优化问题
        ceres::Problem problem;
        
        // 为所有关键帧创建参数块
        std::map<int, double*> sim3_blocks;
        std::map<int, Sim3> vScw;  // 原始Sim3
        
        // 共视图中最小共视特征点数阈值
        const int minFeat = 100; 
        
        // 添加关键帧顶点
        std::cout << "Setting up keyframe vertices..." << std::endl;
        for (const auto& kf_pair : keyframes_) {
            int id = kf_pair.first;
            const KeyFrame& kf = kf_pair.second;
            
            if (kf.is_bad) continue;
            
            // 参数块格式: [tx, ty, tz, qx, qy, qz, qw, s]
            double* sim3_block = new double[8];
            
            // 检查该关键帧是否有校正后的Sim3
            auto it_corrected = corrected_sim3_.find(id);
            if (it_corrected != corrected_sim3_.end()) {
                // 使用校正后的Sim3
                const Sim3& corrected_sim3 = it_corrected->second;
                
                sim3_block[0] = corrected_sim3.translation().x();
                sim3_block[1] = corrected_sim3.translation().y();
                sim3_block[2] = corrected_sim3.translation().z();
                sim3_block[3] = corrected_sim3.quaternion().x();
                sim3_block[4] = corrected_sim3.quaternion().y();
                sim3_block[5] = corrected_sim3.quaternion().z();
                sim3_block[6] = corrected_sim3.quaternion().w();
                sim3_block[7] = corrected_sim3.scale();
                
                vScw[id] = corrected_sim3;
            } else {
                // 使用当前位姿，尺度=1.0
                sim3_block[0] = kf.position.x();
                sim3_block[1] = kf.position.y();
                sim3_block[2] = kf.position.z();
                sim3_block[3] = kf.rotation.x();
                sim3_block[4] = kf.rotation.y();
                sim3_block[5] = kf.rotation.z();
                sim3_block[6] = kf.rotation.w();
                sim3_block[7] = 1.0;  // 尺度=1.0
                
                Sim3 Siw(kf.rotation, kf.position, 1.0);
                vScw[id] = Siw;
            }
            
            // 存储参数块
            sim3_blocks[id] = sim3_block;
            
            // 添加参数块到优化问题，并设置正确的流形
            ceres::Manifold* manifold = new Sim3Manifold(fix_scale);
            problem.AddParameterBlock(sim3_block, 8, manifold);
            
            // 固定初始关键帧和回环关键帧
            if (id == init_kf_id || id == loop_kf_id) {
                problem.SetParameterBlockConstant(sim3_block);
            }
        }
        
        // 用于跟踪已添加的边，避免重复
        std::set<std::pair<int, int>> inserted_edges;
        
        // 添加回环闭合边 (从LoopConnections)
        std::cout << "Adding loop closure edges..." << std::endl;
        int count_loop = 0;
        for (const auto& entry : loop_connections_) {
            int source_id = entry.first;
            const std::set<int>& connected_kfs = entry.second;
            
            if (sim3_blocks.find(source_id) == sim3_blocks.end()) continue;
            
            const Sim3& Siw = vScw[source_id];
            const Sim3 Swi = Siw.inverse();
            
            for (int target_id : connected_kfs) {
                if (sim3_blocks.find(target_id) == sim3_blocks.end()) continue;
                
                // 如果已处理过，则跳过（考虑两个方向）
                std::pair<int, int> edge_pair(std::min(source_id, target_id), std::max(source_id, target_id));
                if (inserted_edges.count(edge_pair)) continue;
                
                // 从共视图获取权重（如果可用）
                int weight = 0;
                if (keyframes_.find(source_id) != keyframes_.end() &&
                    keyframes_[source_id].covisible_keyframes.find(target_id) != 
                    keyframes_[source_id].covisible_keyframes.end()) {
                    weight = keyframes_[source_id].covisible_keyframes[target_id];
                }
                
                // 跳过权重低的边，除非是当前回环
                if ((source_id != current_kf_id || target_id != loop_kf_id) && weight < minFeat) continue;
                
                const Sim3& Sjw = vScw[target_id];
                const Sim3 Sji = Sjw * Swi;
                
                // 创建误差函数
                ceres::CostFunction* cost_function = Sim3Error::Create(Sji);
                
                // 添加鲁棒核
                ceres::LossFunction* loss_function = new ceres::HuberLoss(1.345);
                
                // 添加到优化问题
                problem.AddResidualBlock(
                    cost_function,
                    loss_function,
                    sim3_blocks[source_id],
                    sim3_blocks[target_id]);
                
                // 标记为已处理
                inserted_edges.insert(edge_pair);
                count_loop++;
            }
        }
        std::cout << "Added " << count_loop << " loop edges" << std::endl;
        
        // 添加生成树边
        std::cout << "Adding spanning tree edges..." << std::endl;
        int count_spanning = 0;
        for (const auto& kf_pair : keyframes_) {
            int source_id = kf_pair.first;
            const KeyFrame& kf = kf_pair.second;
            
            if (kf.is_bad || kf.parent_id < 0) continue;
            
            int target_id = kf.parent_id;
            
            if (sim3_blocks.find(source_id) == sim3_blocks.end() || 
                sim3_blocks.find(target_id) == sim3_blocks.end()) continue;
            
            // 跳过已处理的边
            std::pair<int, int> edge_pair(std::min(source_id, target_id), std::max(source_id, target_id));
            if (inserted_edges.count(edge_pair)) continue;
            
            // 计算相对Sim3
            const Sim3& Siw = vScw[source_id];
            const Sim3& Sjw = vScw[target_id];
            const Sim3 Sji = Sjw * Siw.inverse();
            
            // 创建误差函数
            ceres::CostFunction* cost_function = Sim3Error::Create(Sji);
            
            // 添加鲁棒核
            ceres::LossFunction* loss_function = new ceres::HuberLoss(1.345);
            
            // 添加到优化问题
            problem.AddResidualBlock(
                cost_function,
                loss_function,
                sim3_blocks[source_id],
                sim3_blocks[target_id]);
            
            // 标记为已处理
            inserted_edges.insert(edge_pair);
            count_spanning++;
        }
        std::cout << "Added " << count_spanning << " spanning tree edges" << std::endl;
        
        // 添加已存在的回环边
        std::cout << "Adding existing loop edges..." << std::endl;
        int count_existing_loop = 0;
        for (const auto& kf_pair : keyframes_) {
            int source_id = kf_pair.first;
            const KeyFrame& kf = kf_pair.second;
            
            if (kf.is_bad) continue;
            
            for (int target_id : kf.loop_edges) {
                // 只处理一次
                if (target_id >= source_id) continue;
                
                if (sim3_blocks.find(source_id) == sim3_blocks.end() || 
                    sim3_blocks.find(target_id) == sim3_blocks.end()) continue;
                
                // 跳过已处理的边
                std::pair<int, int> edge_pair(std::min(source_id, target_id), std::max(source_id, target_id));
                if (inserted_edges.count(edge_pair)) continue;
                
                // 计算相对Sim3
                const Sim3& Siw = vScw[source_id];
                const Sim3& Sjw = vScw[target_id];
                const Sim3 Sji = Sjw * Siw.inverse();
                
                // 创建误差函数
                ceres::CostFunction* cost_function = Sim3Error::Create(Sji);
                
                // 添加鲁棒核
                ceres::LossFunction* loss_function = new ceres::HuberLoss(1.345);
                
                // 添加到优化问题
                problem.AddResidualBlock(
                    cost_function,
                    loss_function,
                    sim3_blocks[source_id],
                    sim3_blocks[target_id]);
                
                // 标记为已处理
                inserted_edges.insert(edge_pair);
                count_existing_loop++;
            }
        }
        std::cout << "Added " << count_existing_loop << " existing loop edges" << std::endl;
        
        // 添加共视图边
        std::cout << "Adding covisibility edges..." << std::endl;
        int count_covisibility = 0;
        for (const auto& kf_pair : keyframes_) {
            int source_id = kf_pair.first;
            const KeyFrame& kf = kf_pair.second;
            
            if (kf.is_bad) continue;
            
            // 处理共视关键帧
            for (const auto& covis_pair : kf.covisible_keyframes) {
                int target_id = covis_pair.first;
                int weight = covis_pair.second;
                
                // 只处理一个方向，并跳过父关键帧
                if (target_id <= source_id || target_id == kf.parent_id) continue;
                
                // 跳过已在回环边中的
                if (kf.loop_edges.count(target_id)) continue;
                
                if (sim3_blocks.find(source_id) == sim3_blocks.end() || 
                    sim3_blocks.find(target_id) == sim3_blocks.end()) continue;
                
                // 跳过已处理的边
                std::pair<int, int> edge_pair(source_id, target_id);
                if (inserted_edges.count(edge_pair)) continue;
                
                // 跳过权重过低的边
                if (weight < minFeat) continue;
                
                // 计算相对Sim3
                const Sim3& Siw = vScw[source_id];
                const Sim3& Sjw = vScw[target_id];
                const Sim3 Sji = Sjw * Siw.inverse();
                
                // 创建误差函数
                ceres::CostFunction* cost_function = Sim3Error::Create(Sji);
                
                // 添加鲁棒核
                ceres::LossFunction* loss_function = new ceres::HuberLoss(1.345);
                
                // 添加到优化问题
                problem.AddResidualBlock(
                    cost_function,
                    loss_function,
                    sim3_blocks[source_id],
                    sim3_blocks[target_id]);
                
                // 标记为已处理
                inserted_edges.insert(edge_pair);
                count_covisibility++;
            }
        }
        std::cout << "Added " << count_covisibility << " covisibility edges" << std::endl;
        
        // 设置求解器选项
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        options.minimizer_progress_to_stdout = true;
        options.num_threads = 8;
        
        // 从优化参数获取最大迭代次数，默认为20
        options.max_num_iterations = 20;
        if (optimization_params.find("NUM_ITERATIONS") != optimization_params.end()) {
            options.max_num_iterations = std::stoi(optimization_params["NUM_ITERATIONS"]);
        }
        
        // 关键：设置初始信任区域半径以匹配g2o的lambda
        options.initial_trust_region_radius = 1e-16;
        
        // 添加稳定性设置
        options.function_tolerance = 1e-6;
        options.parameter_tolerance = 1e-8;
        options.gradient_tolerance = 1e-10;
        options.max_num_consecutive_invalid_steps = 5;
        
        // 解决优化问题
        std::cout << "Optimizing..." << std::endl;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        
        std::cout << summary.BriefReport() << std::endl;
        
        // 更新关键帧位姿
        std::cout << "Updating keyframes..." << std::endl;
        for (auto& kf_pair : keyframes_) {
            int id = kf_pair.first;
            KeyFrame& kf = kf_pair.second;
            
            if (sim3_blocks.find(id) == sim3_blocks.end()) continue;
            
            // 获取优化后的Sim3
            const double* sim3_block = sim3_blocks[id];
            
            // 提取组件
            Eigen::Vector3d t(sim3_block[0], sim3_block[1], sim3_block[2]);
            Eigen::Quaterniond q(sim3_block[6], sim3_block[3], sim3_block[4], sim3_block[5]); // w, x, y, z
            q.normalize();
            double s = sim3_block[7];
            
            // 检查极端值或NaN
            bool has_nan = false;
            bool has_extreme = false;
            
            for (int i = 0; i < 3; i++) {
                if (!std::isfinite(t[i])) {
                    has_nan = true;
                    break;
                }
                if (std::abs(t[i]) > 1e10) {
                    has_extreme = true;
                    break;
                }
            }
            
            // 如果值极端，则跳过此关键帧
            if (has_nan || has_extreme || !std::isfinite(s) || s <= 0) {
                std::cerr << "Warning: Extreme values detected for KF " << id << ", skipping update" << std::endl;
                continue;
            }
            
            // 更新关键帧位姿 - 将Sim3转换为SE3，通过除以尺度
            kf.rotation = q;
            kf.position = t / s;
        }
        
        // 保存优化后的轨迹
        SaveOptimizedTrajectory(output_file);
        
        // 清理
        for (auto& block_pair : sim3_blocks) {
            delete[] block_pair.second;
        }
        
        std::cout << "Essential Graph optimization completed successfully!" << std::endl;
        return true;
    }
    
    // Save optimized trajectory
    void SaveOptimizedTrajectory(const std::string& output_file) {
        // Save trajectory file
        std::ofstream file(output_file);
        if (!file.is_open()) {
            std::cerr << "Failed to open output file: " << output_file << std::endl;
            return;
        }
        
        file << "# id timestamp tx ty tz qx qy qz qw" << std::endl;
        
        // Sort keyframes by ID for consistency
        std::vector<int> sorted_ids;
        for (const auto& kf_pair : keyframes_) {
            if (!kf_pair.second.is_bad) {
                sorted_ids.push_back(kf_pair.first);
            }
        }
        
        std::sort(sorted_ids.begin(), sorted_ids.end());
        
        for (int id : sorted_ids) {
            const KeyFrame& kf = keyframes_[id];
            
            // Sanity check for output
            bool has_nan = false;
            for (int i = 0; i < 3; i++) {
                if (!std::isfinite(kf.position[i])) {
                    has_nan = true;
                    break;
                }
            }
            
            if (has_nan) {
                std::cerr << "Warning: KF " << id << " has NaN values, skipping in output" << std::endl;
                continue;
            }
            
            file << id << " " 
                 << std::fixed << std::setprecision(9) << kf.timestamp << " "
                 << std::fixed << std::setprecision(6)
                 << kf.position.x() << " " 
                 << kf.position.y() << " " 
                 << kf.position.z() << " "
                 << kf.rotation.x() << " " 
                 << kf.rotation.y() << " " 
                 << kf.rotation.z() << " " 
                 << kf.rotation.w() << std::endl;
        }
        
        file.close();
        std::cout << "Saved optimized trajectory to: " << output_file << std::endl;
    }
    
    // Data storage
    std::map<int, KeyFrame> keyframes_;
    std::map<int, Sim3> non_corrected_sim3_;
    std::map<int, Sim3> corrected_sim3_;
    std::map<int, std::set<int>> loop_connections_;
    std::map<std::string, std::string> optimization_params;
    
    std::vector<std::tuple<int, int, Sim3>> loop_constraints_;
    std::vector<std::pair<int, int>> spanning_tree_edges_;
    std::vector<std::tuple<int, int, double>> covisibility_edges_;
    std::vector<std::pair<int, int>> connection_changes_;
    
    // Current state
    int current_kf_id;
    int loop_kf_id;
    int init_kf_id;
};

int main(int argc, char** argv) {
    // Set default paths
    std::string input_dir = "/Datasets/CERES_Work/input";
    std::string output_file = "/Datasets/CERES_Work/output/optimized_trajectory.txt";
    
    // Override if provided
    if (argc > 1) input_dir = argv[1];
    if (argc > 2) output_file = argv[2];
    
    // Create output directory if it doesn't exist
    std::string output_dir = output_file.substr(0, output_file.find_last_of('/'));
    struct stat info;
    if (stat(output_dir.c_str(), &info) != 0) {
        // Directory doesn't exist, create it
        #ifdef _WIN32
            _mkdir(output_dir.c_str());
        #else
            mkdir(output_dir.c_str(), 0755);
        #endif
    }
    
    // Create the optimizer
    TrajectoryOptimizer optimizer;
    
    // Load input data
    std::cout << "Loading input data..." << std::endl;
    
    // Load Sim3 transformed trajectory
    if (!optimizer.LoadTrajectory(input_dir + "/sim3_transformed_trajectory.txt")) {
        std::cerr << "Failed to load trajectory!" << std::endl;
        return 1;
    }
    
    // Load Essential Graph
    optimizer.LoadEssentialGraph(input_dir + "/pre/essential_graph.txt");
    
    // Load Loop Constraints
    optimizer.LoadLoopConstraints(input_dir + "/metadata/loop_constraints.txt");
    
    // Load Connection Changes
    optimizer.LoadConnectionChanges(input_dir + "/metadata/connection_changes.txt");
    
    // Load Sim3 Transformations
    optimizer.LoadSim3Transformations(input_dir + "/metadata/sim3_transforms.txt");
    
    // Load Loop Info
    optimizer.LoadLoopInfo(input_dir + "/metadata/loop_info.txt");
    
    // Load Optimization Parameters
    optimizer.LoadOptimizationParams(input_dir + "/metadata/optimization_params.txt");
    
    // Load Connections (if available)
    try {
        std::ifstream test_file(input_dir + "/connections.txt");
        if (test_file.is_open()) {
            test_file.close();
            optimizer.LoadConnectionChanges(input_dir + "/connections.txt");
        }
    } catch (const std::exception& e) {
        std::cout << "No connections.txt file available (this is optional)" << std::endl;
    }
    
    // Run the optimization
    std::cout << "Starting Essential Graph optimization..." << std::endl;
    if (!optimizer.OptimizeEssentialGraph(output_file)) {
        std::cerr << "Optimization failed!" << std::endl;
        return 1;
    }
    
    std::cout << "Essential Graph optimization completed successfully!" << std::endl;
    std::cout << "Optimized trajectory saved to: " << output_file << std::endl;
    
    return 0;
}
