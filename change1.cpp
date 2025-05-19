#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <unordered_map>
#include <memory>
#include <algorithm>
#include <iomanip>

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>

// SE3 李代数参数化 - 模仿 ORB-SLAM3 的 VertexSim3Expmap 
// 但固定尺度为1，使用SE3李代数 [rho(3), phi(3)]
class SE3Parameterization : public ceres::Manifold {
public:
    ~SE3Parameterization() {}
    
    // 6维切空间：[rho(3), phi(3)] - SE3的李代数
    // 7维环境空间：[tx, ty, tz, qx, qy, qz, qw] - SE3的四元数表示
    int AmbientSize() const override { return 7; }
    int TangentSize() const override { return 6; }
    
    // SE3的指数映射 - 类似 ORB-SLAM3 的 oplusImpl
    bool Plus(const double* x, const double* delta, double* x_plus_delta) const override {
        // x 是当前的SE3状态: [tx, ty, tz, qx, qy, qz, qw]
        // delta 是SE3李代数增量: [rho(3), phi(3)]
        
        // 提取当前状态
        Eigen::Vector3d t_current(x[0], x[1], x[2]);
        Eigen::Quaterniond q_current(x[6], x[3], x[4], x[5]);
        q_current.normalize();
        
        // SE3李代数增量
        Eigen::Vector3d rho(delta[0], delta[1], delta[2]);    // 平移部分
        Eigen::Vector3d phi(delta[3], delta[4], delta[5]);    // 旋转部分
        
        // 旋转增量的指数映射 (Rodrigues公式)
        double angle = phi.norm();
        Eigen::Matrix3d R_delta = Eigen::Matrix3d::Identity();
        Eigen::Vector3d t_delta = rho;
        
        if (angle > 1e-8) {
            Eigen::Vector3d axis = phi / angle;
            
            // Rodrigues公式计算旋转矩阵
            Eigen::Matrix3d K = Eigen::Matrix3d::Zero();
            K << 0, -axis(2), axis(1),
                 axis(2), 0, -axis(0),
                 -axis(1), axis(0), 0;
            
            R_delta = Eigen::Matrix3d::Identity() + 
                      sin(angle) * K + 
                      (1 - cos(angle)) * K * K;
            
            // SE3的雅可比矩阵用于平移部分
            Eigen::Matrix3d V = Eigen::Matrix3d::Identity();
            if (angle > 1e-8) {
                V = (sin(angle) / angle) * Eigen::Matrix3d::Identity() + 
                    (1 - cos(angle)) / angle * K + 
                    (angle - sin(angle)) / angle * K * K;
            }
            t_delta = V * rho;
        }
        
        // 应用SE3增量: T_new = exp([rho, phi]) * T_current
        Eigen::Matrix3d R_current = q_current.toRotationMatrix();
        Eigen::Matrix3d R_new = R_delta * R_current;
        Eigen::Vector3d t_new = R_delta * t_current + t_delta;
        
        // 转换回四元数
        Eigen::Quaterniond q_new(R_new);
        q_new.normalize();
        
        // 输出新状态
        x_plus_delta[0] = t_new(0);
        x_plus_delta[1] = t_new(1);
        x_plus_delta[2] = t_new(2);
        x_plus_delta[3] = q_new.x();
        x_plus_delta[4] = q_new.y();
        x_plus_delta[5] = q_new.z();
        x_plus_delta[6] = q_new.w();
        
        return true;
    }
    
    bool PlusJacobian(const double* x, double* jacobian) const override {
        // 计算Plus操作的雅可比矩阵
        Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> J(jacobian);
        J.setZero();
        
        // 数值微分计算雅可比(简化实现)
        const double eps = 1e-8;
        double x_plus_eps[7], x_minus_eps[7];
        double delta_plus[6], delta_minus[6];
        
        for (int i = 0; i < 6; ++i) {
            // 正向扰动
            for (int j = 0; j < 6; ++j) {
                delta_plus[j] = (i == j) ? eps : 0.0;
                delta_minus[j] = (i == j) ? -eps : 0.0;
            }
            
            Plus(x, delta_plus, x_plus_eps);
            Plus(x, delta_minus, x_minus_eps);
            
            // 数值微分
            for (int k = 0; k < 7; ++k) {
                J(k, i) = (x_plus_eps[k] - x_minus_eps[k]) / (2.0 * eps);
            }
        }
        
        return true;
    }
    
    bool Minus(const double* y, const double* x, double* y_minus_x) const override {
        // SE3的对数映射 - 计算从x到y的SE3李代数
        Eigen::Vector3d t_x(x[0], x[1], x[2]);
        Eigen::Vector3d t_y(y[0], y[1], y[2]);
        Eigen::Quaterniond q_x(x[6], x[3], x[4], x[5]);
        Eigen::Quaterniond q_y(y[6], y[3], y[4], y[5]);
        
        q_x.normalize();
        q_y.normalize();
        
        // 计算相对变换
        Eigen::Matrix3d R_x = q_x.toRotationMatrix();
        Eigen::Matrix3d R_y = q_y.toRotationMatrix();
        Eigen::Matrix3d R_rel = R_x.transpose() * R_y;
        Eigen::Vector3d t_rel = R_x.transpose() * (t_y - t_x);
        
        // 旋转的对数映射
        Eigen::Vector3d phi;
        double trace = R_rel.trace();
        if (trace > 3.0 - 1e-6) {
            // 接近单位矩阵的情况
            phi = 0.5 * Eigen::Vector3d(R_rel(2,1) - R_rel(1,2),
                                       R_rel(0,2) - R_rel(2,0),
                                       R_rel(1,0) - R_rel(0,1));
        } else {
            double angle = acos(0.5 * (trace - 1.0));
            Eigen::Vector3d axis;
            axis(0) = R_rel(2,1) - R_rel(1,2);
            axis(1) = R_rel(0,2) - R_rel(2,0);
            axis(2) = R_rel(1,0) - R_rel(0,1);
            axis.normalize();
            phi = angle * axis;
        }
        
        // 平移的对数映射 (需要考虑SE3的雅可比)
        double angle = phi.norm();
        Eigen::Vector3d rho = t_rel;
        if (angle > 1e-8) {
            Eigen::Matrix3d V_inv = Eigen::Matrix3d::Identity();
            Eigen::Matrix3d K = Eigen::Matrix3d::Zero();
            K << 0, -phi(2), phi(1),
                 phi(2), 0, -phi(0),
                 -phi(1), phi(0), 0;
            K /= angle;
            
            V_inv = Eigen::Matrix3d::Identity() - 0.5 * K + 
                    (2*sin(angle) - angle*(1 + cos(angle))) / 
                    (2*angle*angle*sin(angle)) * K * K;
            rho = V_inv * t_rel;
        }
        
        y_minus_x[0] = rho(0);
        y_minus_x[1] = rho(1);
        y_minus_x[2] = rho(2);
        y_minus_x[3] = phi(0);
        y_minus_x[4] = phi(1);
        y_minus_x[5] = phi(2);
        
        return true;
    }
    
    bool MinusJacobian(const double* x, double* jacobian) const override {
        // 数值微分计算MinusJacobian
        Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> J(jacobian);
        J.setZero();
        
        const double eps = 1e-8;
        double y_plus[7], y_minus[7];
        double diff_plus[6], diff_minus[6];
        
        for (int i = 0; i < 7; ++i) {
            // 对y的第i个分量进行扰动
            for (int j = 0; j < 7; ++j) {
                y_plus[j] = x[j] + ((i == j) ? eps : 0.0);
                y_minus[j] = x[j] - ((i == j) ? eps : 0.0);
            }
            
            Minus(y_plus, x, diff_plus);
            Minus(y_minus, x, diff_minus);
            
            for (int k = 0; k < 6; ++k) {
                J(k, i) = (diff_plus[k] - diff_minus[k]) / (2.0 * eps);
            }
        }
        
        return true;
    }
};


// SO(3)的对数映射：将旋转矩阵转换为轴角向量
template<typename T>
Eigen::Matrix<T, 3, 1> LogSO3(const Eigen::Matrix<T, 3, 3>& R) {
    // 计算旋转角度
    T trace = R.trace();
    T cos_angle = (trace - T(1.0)) * T(0.5);
    
    // 限制cos_angle在[-1, 1]范围内
    if (cos_angle > T(1.0)) cos_angle = T(1.0);
    if (cos_angle < T(-1.0)) cos_angle = T(-1.0);
    
    T angle = acos(cos_angle);
    
    Eigen::Matrix<T, 3, 1> omega;
    
    // 处理小角度情况
    if (angle < T(1e-6)) {
        // 对于小角度，使用一阶近似
        T factor = T(0.5) * (T(1.0) + trace * trace / T(12.0));
        omega << factor * (R(2, 1) - R(1, 2)),
                 factor * (R(0, 2) - R(2, 0)),
                 factor * (R(1, 0) - R(0, 1));
    } else if (angle > T(M_PI - 1e-6)) {
        // 处理接近180度的情况
        Eigen::Matrix<T, 3, 3> A = (R + R.transpose()) * T(0.5);
        A.diagonal().array() -= T(1.0);
        
        // 找到最大的对角元素
        int max_idx = 0;
        T max_val = ceres::abs(A(0, 0));
        for (int i = 1; i < 3; ++i) {
            if (ceres::abs(A(i, i)) > max_val) {
                max_val = ceres::abs(A(i, i));
                max_idx = i;
            }
        }
        
        // 计算轴向量
        Eigen::Matrix<T, 3, 1> axis;
        axis[max_idx] = sqrt(A(max_idx, max_idx));
        for (int i = 0; i < 3; ++i) {
            if (i != max_idx) {
                axis[i] = A(max_idx, i) / axis[max_idx];
            }
        }
        axis.normalize();
        
        // 确定正确的符号
        if ((R(2, 1) - R(1, 2)) * axis[0] + 
            (R(0, 2) - R(2, 0)) * axis[1] + 
            (R(1, 0) - R(0, 1)) * axis[2] < T(0.0)) {
            axis = -axis;
        }
        
        omega = angle * axis;
    } else {
        // 一般情况
        T sin_angle = sin(angle);
        T factor = angle / (T(2.0) * sin_angle);
        omega << factor * (R(2, 1) - R(1, 2)),
                 factor * (R(0, 2) - R(2, 0)),
                 factor * (R(1, 0) - R(0, 1));
    }
    
    return omega;
}



// 专门用于回环约束的类 - 对应g2o的EdgeSim3
class SE3LoopConstraintCost {
public:
    SE3LoopConstraintCost(const Eigen::Matrix4d& relative_transform, const Eigen::Matrix<double, 6, 6>& information)
        : relative_rotation_(relative_transform.block<3, 3>(0, 0)),
          relative_translation_(relative_transform.block<3, 1>(0, 3)),
          sqrt_information_(information.llt().matrixL()) {
    }
    
    template <typename T>
    bool operator()(const T* const pose_i, const T* const pose_j, T* residuals) const {
        // pose_i: 第一个关键帧, pose_j: 第二个关键帧
        // 格式: [tx, ty, tz, qx, qy, qz, qw]
        
        // 提取姿态
        Eigen::Matrix<T, 3, 1> t_i(pose_i[0], pose_i[1], pose_i[2]);
        Eigen::Matrix<T, 3, 1> t_j(pose_j[0], pose_j[1], pose_j[2]);
        Eigen::Quaternion<T> q_i(pose_i[6], pose_i[3], pose_i[4], pose_i[5]);
        Eigen::Quaternion<T> q_j(pose_j[6], pose_j[3], pose_j[4], pose_j[5]);
        
        // 转换为旋转矩阵
        Eigen::Matrix<T, 3, 3> R_i = q_i.toRotationMatrix();
        Eigen::Matrix<T, 3, 3> R_j = q_j.toRotationMatrix();
        
        // 计算相对变换 T_ji = T_j * T_i^{-1} (对应ORB-SLAM3的 Sji = Sjw * Swi)
        Eigen::Matrix<T, 3, 3> R_ji = R_j * R_i.transpose();
        Eigen::Matrix<T, 3, 1> t_ji = R_j * (R_i.transpose() * (-t_i)) + t_j;
        
        // 预期的相对变换
        Eigen::Matrix<T, 3, 3> R_expected = relative_rotation_.cast<T>();
        Eigen::Matrix<T, 3, 1> t_expected = relative_translation_.cast<T>();
        
        // 计算旋转误差
        Eigen::Matrix<T, 3, 3> R_error_mat = R_expected.transpose() * R_ji;
        Eigen::Matrix<T, 3, 1> rotation_error = LogSO3(R_error_mat);
        
        // 计算平移误差
        Eigen::Matrix<T, 3, 1> translation_error = t_ji - t_expected;
        
        // 组合残差
        residuals[0] = rotation_error[0];
        residuals[1] = rotation_error[1];
        residuals[2] = rotation_error[2];
        residuals[3] = translation_error[0];
        residuals[4] = translation_error[1];
        residuals[5] = translation_error[2];
        
        // 应用信息矩阵
        Eigen::Map<Eigen::Matrix<T, 6, 1>> residuals_map(residuals);
        residuals_map = sqrt_information_.cast<T>() * residuals_map;
        
        return true;
    }
    
    static ceres::CostFunction* Create(const Eigen::Matrix4d& relative_transform,
                                       const Eigen::Matrix<double, 6, 6>& information) {
        return new ceres::AutoDiffCostFunction<SE3LoopConstraintCost, 6, 7, 7>(
            new SE3LoopConstraintCost(relative_transform, information));
    }
    
private:
    const Eigen::Matrix3d relative_rotation_;
    const Eigen::Vector3d relative_translation_;
    const Eigen::Matrix<double, 6, 6> sqrt_information_;
};






// 关键帧结构
struct KeyFrame {
    int id;
    double timestamp;
    Eigen::Vector3d translation;
    Eigen::Quaterniond quaternion;
    int parent_id;
    bool has_velocity;
    bool is_fixed;
    bool is_bad;
    bool is_inertial;
    bool is_virtual;
    
    // SE3 状态 [tx, ty, tz, qx, qy, qz, qw]
    std::vector<double> se3_state;
    
    KeyFrame() {
        se3_state.resize(7);
        se3_state[6] = 1.0; // qw = 1.0
        quaternion.setIdentity();
        translation.setZero();
    }
    
    void SetPose(const Eigen::Vector3d& t, const Eigen::Quaterniond& q) {
        translation = t;
        quaternion = q.normalized();
        // 更新SE3状态
        se3_state[0] = t.x();
        se3_state[1] = t.y();
        se3_state[2] = t.z();
        se3_state[3] = quaternion.x();
        se3_state[4] = quaternion.y();
        se3_state[5] = quaternion.z();
        se3_state[6] = quaternion.w();
    }
    
    Eigen::Matrix4d GetTransformMatrix() const {
        Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
        T.block<3, 3>(0, 0) = quaternion.toRotationMatrix();
        T.block<3, 1>(0, 3) = translation;
        return T;
    }
    
    // 从SE3状态更新translation和quaternion
    void UpdateFromState() {
        translation = Eigen::Vector3d(se3_state[0], se3_state[1], se3_state[2]);
        quaternion = Eigen::Quaterniond(se3_state[6], se3_state[3], se3_state[4], se3_state[5]);
        quaternion.normalize();
    }
};

// 数据结构
struct OptimizationData {
    std::map<int, std::shared_ptr<KeyFrame>> keyframes;
    std::map<int, std::vector<int>> spanning_tree;
    std::map<int, std::map<int, int>> covisibility;
    std::map<int, std::vector<int>> loop_connections;

    // 回环边信息
    std::map<int, std::set<int>> loop_edges;  // KF_ID -> {Loop_KF_IDs}
    
    int loop_kf_id;
    int current_kf_id;
    bool fixed_scale;
    int init_kf_id;
    int max_kf_id;
    
    // 回环匹配相对变换
    Eigen::Matrix4d loop_transform_matrix;
    
    // SE3 修正姿态
    std::map<int, KeyFrame> corrected_poses;
    std::map<int, KeyFrame> non_corrected_poses;
};

class ORBSlamLoopOptimizer {
private:
    OptimizationData data_;
    std::unique_ptr<ceres::Problem> problem_;
    
public:
    ORBSlamLoopOptimizer() {
        problem_ = std::make_unique<ceres::Problem>();
    }
    
    // 解析关键帧姿态文件
    bool ParseKeyFramePoses(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "无法打开文件: " << filename << std::endl;
            return false;
        }
        
        std::string line;
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;
            
            std::istringstream iss(line);
            int id;
            double timestamp, tx, ty, tz, qx, qy, qz, qw;
            
            if (iss >> id >> timestamp >> tx >> ty >> tz >> qx >> qy >> qz >> qw) {
                auto kf = std::make_shared<KeyFrame>();
                kf->id = id;
                kf->timestamp = timestamp;
                
                Eigen::Vector3d translation(tx, ty, tz);
                Eigen::Quaterniond quaternion(qw, qx, qy, qz);
                kf->SetPose(translation, quaternion);
                
                data_.keyframes[id] = kf;
            }
        }
        
        std::cout << "解析关键帧姿态: " << data_.keyframes.size() << " 个关键帧" << std::endl;
        return true;
    }
    
    // 解析关键帧信息文件
    bool ParseKeyFrameInfo(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "无法打开文件: " << filename << std::endl;
            return false;
        }
        
        std::string line;
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;
            
            std::istringstream iss(line);
            int id, parent_id, has_velocity, is_fixed, is_bad, is_inertial, is_virtual;
            
            if (iss >> id >> parent_id >> has_velocity >> is_fixed >> is_bad >> is_inertial >> is_virtual) {
                if (data_.keyframes.find(id) != data_.keyframes.end()) {
                    auto& kf = data_.keyframes[id];
                    kf->parent_id = parent_id;
                    kf->has_velocity = has_velocity;
                    kf->is_fixed = is_fixed;
                    kf->is_bad = is_bad;
                    kf->is_inertial = is_inertial;
                    kf->is_virtual = is_virtual;
                }
            }
        }
        
        std::cout << "解析关键帧信息完成" << std::endl;
        return true;
    }
    
    // 解析地图信息
    bool ParseMapInfo(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "无法打开文件: " << filename << std::endl;
            return false;
        }
        
        std::string line;
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;
            
            std::istringstream iss(line);
            std::string key;
            iss >> key;
            
            if (key == "INIT_KF_ID") {
                iss >> data_.init_kf_id;
            } else if (key == "MAX_KF_ID") {
                iss >> data_.max_kf_id;
            }
        }
        
        std::cout << "解析地图信息: INIT_KF_ID=" << data_.init_kf_id 
                  << ", MAX_KF_ID=" << data_.max_kf_id << std::endl;
        return true;
    }
    
    // 解析关键帧ID信息
    bool ParseKeyFrameIds(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "无法打开文件: " << filename << std::endl;
            return false;
        }
        
        std::string line;
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;
            
            std::istringstream iss(line);
            std::string key;
            iss >> key;
            
            if (key == "LOOP_KF_ID") {
                iss >> data_.loop_kf_id;
            } else if (key == "CURRENT_KF_ID") {
                iss >> data_.current_kf_id;
            } else if (key == "FIXED_SCALE") {
                iss >> data_.fixed_scale;
            }
        }
        
        std::cout << "解析关键帧ID: LOOP_KF_ID=" << data_.loop_kf_id 
                  << ", CURRENT_KF_ID=" << data_.current_kf_id 
                  << ", FIXED_SCALE=" << data_.fixed_scale << std::endl;
        return true;
    }
    
    // 解析回环匹配
    bool ParseLoopMatch(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "无法打开文件: " << filename << std::endl;
            return false;
        }
        
        std::string line;
        data_.loop_transform_matrix.setIdentity();
        
        int line_count = 0;
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;
            
            if (line_count == 0) {
                // 第一行是关键帧ID，已经从keyframe_ids.txt读取
                line_count++;
                continue;
            }
            
            std::istringstream iss(line);
            // 读取4x4变换矩阵
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    if (!(iss >> data_.loop_transform_matrix(i, j))) {
                        std::cerr << "错误：无法解析变换矩阵" << std::endl;
                        return false;
                    }
                }
            }
            break;
        }
        
        std::cout << "解析回环匹配变换矩阵：" << std::endl;
        std::cout << data_.loop_transform_matrix << std::endl;
        return true;
    }
    
    // 解析修正的Sim3（转为SE3，忽略尺度）
    bool ParseCorrectedSim3(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "无法打开文件: " << filename << std::endl;
            return false;
        }
        
        std::string line;
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;
            
            std::istringstream iss(line);
            int id;
            double scale, tx, ty, tz, qx, qy, qz, qw;
            
            if (iss >> id >> scale >> tx >> ty >> tz >> qx >> qy >> qz >> qw) {
                KeyFrame kf;
                kf.id = id;
                Eigen::Vector3d translation(tx, ty, tz);
                Eigen::Quaterniond quaternion(qw, qx, qy, qz);
                kf.SetPose(translation, quaternion);
                data_.corrected_poses[id] = kf;
            }
        }
        
        std::cout << "解析修正SE3: " << data_.corrected_poses.size() << " 个关键帧" << std::endl;
        return true;
    }
    
    // 解析非修正的Sim3（转为SE3，忽略尺度）
    bool ParseNonCorrectedSim3(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "无法打开文件: " << filename << std::endl;
            return false;
        }
        
        std::string line;
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;
            
            std::istringstream iss(line);
            int id;
            double scale, tx, ty, tz, qx, qy, qz, qw;
            
            if (iss >> id >> scale >> tx >> ty >> tz >> qx >> qy >> qz >> qw) {
                KeyFrame kf;
                kf.id = id;
                Eigen::Vector3d translation(tx, ty, tz);
                Eigen::Quaterniond quaternion(qw, qx, qy, qz);
                kf.SetPose(translation, quaternion);
                data_.non_corrected_poses[id] = kf;
            }
        }
        
        std::cout << "解析非修正SE3: " << data_.non_corrected_poses.size() << " 个关键帧" << std::endl;
        return true;
    }
    
    // 解析生成树
    bool ParseSpanningTree(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "无法打开文件: " << filename << std::endl;
            return false;
        }
        
        std::string line;
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;
            
            std::istringstream iss(line);
            int child_id, parent_id;
            
            if (iss >> child_id >> parent_id) {
                data_.spanning_tree[parent_id].push_back(child_id);
            }
        }
        
        std::cout << "解析生成树: " << data_.spanning_tree.size() << " 个父节点" << std::endl;
        return true;
    }

    // 解析回环边文件
    bool ParseLoopEdges(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "无法打开文件: " << filename << std::endl;
            return false;
        }
        
        std::string line;
        int loop_edge_count = 0;
        
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;
            
            std::istringstream iss(line);
            int kf_id, loop_kf_id;
            
            if (iss >> kf_id >> loop_kf_id) {
                // 双向添加回环边
                data_.loop_edges[kf_id].insert(loop_kf_id);
                data_.loop_edges[loop_kf_id].insert(kf_id);
                loop_edge_count++;
            }
        }
        
        std::cout << "解析回环边: " << loop_edge_count << " 条边，涉及 " << data_.loop_edges.size() << " 个关键帧" << std::endl;
        return true;
    }

    // 解析共视关系
    bool ParseCovisibility(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "无法打开文件: " << filename << std::endl;
            return false;
        }
        
        std::string line;
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;
            
            std::istringstream iss(line);
            int kf_id, connected_id, weight;
            
            if (iss >> kf_id >> connected_id >> weight) {
                data_.covisibility[kf_id][connected_id] = weight;
            }
        }
        
        std::cout << "解析共视关系: " << data_.covisibility.size() << " 个关键帧" << std::endl;
        return true;
    }
    
    // 解析回环连接
    bool ParseLoopConnections(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "无法打开文件: " << filename << std::endl;
            return false;
        }
        
        std::string line;
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;
            
            std::istringstream iss(line);
            int kf_id;
            iss >> kf_id;
            
            int connected_id;
            while (iss >> connected_id) {
                data_.loop_connections[kf_id].push_back(connected_id);
            }
        }
        
        std::cout << "解析回环连接: " << data_.loop_connections.size() << " 个关键帧" << std::endl;
        return true;
    }
    
    // 解析所有数据文件
    bool ParseAllData(const std::string& data_dir) {
        std::string base_path = data_dir;
        if (base_path.back() != '/') base_path += "/";
        
        // 解析各种数据文件
        if (!ParseKeyFramePoses(base_path + "keyframe_poses.txt")) return false;
        if (!ParseKeyFrameInfo(base_path + "keyframes.txt")) return false;
        if (!ParseMapInfo(base_path + "map_info.txt")) return false;
        if (!ParseKeyFrameIds(base_path + "keyframe_ids.txt")) return false;
        if (!ParseLoopMatch(base_path + "loop_match.txt")) return false;
        if (!ParseCorrectedSim3(base_path + "corrected_sim3.txt")) return false;
        if (!ParseNonCorrectedSim3(base_path + "non_corrected_sim3.txt")) return false;
        if (!ParseSpanningTree(base_path + "spanning_tree.txt")) return false;
        if (!ParseCovisibility(base_path + "covisibility.txt")) return false;
        if (!ParseLoopConnections(base_path + "loop_connections.txt")) return false;
        if (!ParseLoopEdges(base_path + "loop_edges.txt")) return false;  // 新添加
        
        std::cout << "\n所有数据文件解析完成" << std::endl;
        return true;
    }
    
    // 设置优化问题并添加关键帧顶点
    void SetupOptimizationProblem() {
        std::cout << "\n开始设置优化问题..." << std::endl;
        
        // 为每个关键帧添加顶点
        for (auto& kf_pair : data_.keyframes) {
            auto& kf = kf_pair.second;
            
            // 跳过坏的关键帧
            if (kf->is_bad) continue;
            
            // 检查是否有修正的姿态
            if (data_.corrected_poses.find(kf->id) != data_.corrected_poses.end()) {
                // 使用修正的SE3
                const auto& corrected = data_.corrected_poses[kf->id];
                for (int i = 0; i < 7; ++i) {
                    kf->se3_state[i] = corrected.se3_state[i];
                }
                kf->UpdateFromState();
            }
            
            // 添加参数块到优化问题
            problem_->AddParameterBlock(kf->se3_state.data(), 7);
            
            // 设置SE3参数化（流形）- 类似 ORB-SLAM3 的 VertexSim3Expmap
            problem_->SetManifold(kf->se3_state.data(), new SE3Parameterization());
            
            // 固定初始关键帧
            if (kf->id == data_.init_kf_id || kf->is_fixed) {
                problem_->SetParameterBlockConstant(kf->se3_state.data());
                std::cout << "固定关键帧 " << kf->id << std::endl;
            }
        }
        
        std::cout << "添加了 " << data_.keyframes.size() << " 个关键帧顶点" << std::endl;
        std::cout << "优化问题设置完成，参数块数量: " << problem_->NumParameterBlocks() << std::endl;
    }


    // 添加回环约束
    // 添加回环约束 - 精确匹配ORB-SLAM3的Set Loop edges部分
    // 修正后的回环约束添加函数 - 精确匹配ORB-SLAM3
    // 修正后的回环约束添加函数 - 解决权重查询问题
    void AddLoopConstraints() {
        std::cout << "\n=== Adding Loop Edges ===" << std::endl;
        
        const int minFeat = 100; // 最小特征点数阈值
        std::set<std::pair<long unsigned int, long unsigned int>> sInsertedEdges;
        Eigen::Matrix<double, 6, 6> matLambda = Eigen::Matrix<double, 6, 6>::Identity();
        
        int count_loop = 0;
        int attempted_loop = 0;
        
        std::cout << "Minimum features for edge connection: " << minFeat << std::endl;
        std::cout << "\n=== Information Matrix ===" << std::endl;
        std::cout << matLambda << std::endl;
        
        // 遍历所有回环连接 - 精确匹配 ORB-SLAM3 的外层循环
        for (const auto& mit : data_.loop_connections) {
            int nIDi = mit.first;  // 对应 ORB-SLAM3 的 pKF->mnId
            const auto& spConnections = mit.second;  // 对应 ORB-SLAM3 的 spConnections
            
            // 检查关键帧i是否有效
            if (data_.keyframes.find(nIDi) == data_.keyframes.end() || 
                data_.keyframes[nIDi]->is_bad) {
                continue;
            }
            
            auto pKF = data_.keyframes[nIDi];
            
            std::cout << "KF" << nIDi << " connections: " << spConnections.size() << std::endl;
            
            // ========== 获取关键帧i的变换矩阵（对应 ORB-SLAM3 的 vScw[nIDi]） ==========
            Eigen::Matrix4d Siw = Eigen::Matrix4d::Identity();
            bool useCorrectedSim3_i = false;
            
            // 关键修正：按照 ORB-SLAM3 的逻辑选择姿态
            if (data_.corrected_poses.find(nIDi) != data_.corrected_poses.end()) {
                // 使用修正姿态 - 对应 ORB-SLAM3 的 it->second
                const auto& corrected = data_.corrected_poses[nIDi];
                Siw.block<3, 3>(0, 0) = corrected.quaternion.toRotationMatrix();
                Siw.block<3, 1>(0, 3) = corrected.translation;
                useCorrectedSim3_i = true;
            } else {
                // 使用原始姿态 - 对应 ORB-SLAM3 的 pKF->GetPose()
                Siw.block<3, 3>(0, 0) = pKF->quaternion.toRotationMatrix();
                Siw.block<3, 1>(0, 3) = pKF->translation;
            }
            
            // ========== 计算逆变换（对应 ORB-SLAM3 的 Swi = Siw.inverse()） ==========
            Eigen::Matrix4d Swi = Siw.inverse();
            
            // ========== 内层循环：遍历连接的关键帧 ==========
            for (int nIDj : spConnections) {
                attempted_loop++;
                
                // 检查关键帧j是否有效
                if (data_.keyframes.find(nIDj) == data_.keyframes.end() || 
                    data_.keyframes[nIDj]->is_bad) {
                    continue;
                }
                
                auto pKFj = data_.keyframes[nIDj];
                
                // ========== 权重检查（修正版本） ==========
                bool skipEdge = false;
                int weight = 0;
                
                // 检查是否是主要回环边
                bool isMainLoopEdge = (nIDi == data_.current_kf_id && nIDj == data_.loop_kf_id);
                
                if (!isMainLoopEdge) {
                    // 不是主要回环边，需要检查共视权重
                    // 修正：双向查找权重，因为共视关系可能只存储在一个方向上
                    
                    // 首先尝试 i->j 方向
                    auto it_i = data_.covisibility.find(nIDi);
                    if (it_i != data_.covisibility.end()) {
                        auto it_j = it_i->second.find(nIDj);
                        if (it_j != it_i->second.end()) {
                            weight = it_j->second;
                        }
                    }
                    
                    // 如果没找到，尝试 j->i 方向
                    if (weight == 0) {
                        auto it_j = data_.covisibility.find(nIDj);
                        if (it_j != data_.covisibility.end()) {
                            auto it_i = it_j->second.find(nIDi);
                            if (it_i != it_j->second.end()) {
                                weight = it_i->second;
                            }
                        }
                    }
                    
                    // 应用权重阈值检查
                    if (weight < minFeat) {
                        skipEdge = true;
                        // 调试输出：显示为什么边被跳过
                        if (nIDi == 456 && (nIDj == 0 || nIDj == 1 || nIDj == 2)) {
                            std::cout << "  跳过边 KF" << nIDi << " -> KF" << nIDj 
                                      << " | Weight: " << weight << " < " << minFeat << std::endl;
                        }
                    }
                } else {
                    // 主要回环边，获取权重但不进行阈值检查
                    auto it_i = data_.covisibility.find(nIDi);
                    if (it_i != data_.covisibility.end()) {
                        auto it_j = it_i->second.find(nIDj);
                        if (it_j != it_i->second.end()) {
                            weight = it_j->second;
                        }
                    }
                    if (weight == 0) {
                        auto it_j = data_.covisibility.find(nIDj);
                        if (it_j != data_.covisibility.end()) {
                            auto it_i = it_j->second.find(nIDi);
                            if (it_i != it_j->second.end()) {
                                weight = it_i->second;
                            }
                        }
                    }
                }
                
                if (skipEdge) {
                    continue;
                }
                
                // ========== 获取关键帧j的变换矩阵（对应 ORB-SLAM3 的 vScw[nIDj]） ==========
                Eigen::Matrix4d Sjw = Eigen::Matrix4d::Identity();
                bool useCorrectedSim3_j = false;
                
                if (data_.corrected_poses.find(nIDj) != data_.corrected_poses.end()) {
                    // 使用修正姿态
                    const auto& corrected = data_.corrected_poses[nIDj];
                    Sjw.block<3, 3>(0, 0) = corrected.quaternion.toRotationMatrix();
                    Sjw.block<3, 1>(0, 3) = corrected.translation;
                    useCorrectedSim3_j = true;
                } else {
                    // 使用原始姿态
                    Sjw.block<3, 3>(0, 0) = pKFj->quaternion.toRotationMatrix();
                    Sjw.block<3, 1>(0, 3) = pKFj->translation;
                }
                
                // ========== 计算相对变换（对应 ORB-SLAM3 的 Sji = Sjw * Swi） ==========
                Eigen::Matrix4d Sji = Sjw * Swi;
                
                // ========== 创建优化边 ==========
                ceres::CostFunction* cost_function = SE3LoopConstraintCost::Create(Sji, matLambda);
                
                // ========== 添加残差块（注意顶点顺序） ==========
                problem_->AddResidualBlock(cost_function,
                                         nullptr,
                                         pKF->se3_state.data(),   // 对应vertex(0) - 关键帧i
                                         pKFj->se3_state.data()); // 对应vertex(1) - 关键帧j
                
                count_loop++;
                
                // ========== 记录边（与 ORB-SLAM3 完全一致） ==========
                sInsertedEdges.insert(std::make_pair(std::min((long unsigned int)nIDi, (long unsigned int)nIDj), 
                                                   std::max((long unsigned int)nIDi, (long unsigned int)nIDj)));
                
                // ========== 输出详细信息（匹配 ORB-SLAM3 格式） ==========
                Eigen::Vector3d translation = Sji.block<3, 1>(0, 3);
                std::cout << "  Added Loop Edge: KF" << nIDi << " -> KF" << nIDj 
                          << " | Weight: " << weight
                          << " | Translation: [" << translation.transpose() << "]" 
                          << " | Scale: " << 1.0  // SE3没有尺度，固定为1
                          << std::endl;
            }
        }
        
        std::cout << "\nSuccessful Loop Edges: " << count_loop << "/" << attempted_loop << std::endl;
        std::cout << "Unique Edge Pairs: " << sInsertedEdges.size() << std::endl;
    }
    
    // 添加一个调试函数来检查共视关系数据
    void DebugCovisibilityData() {
        std::cout << "\n=== 共视关系调试 ===" << std::endl;
        
        // 检查关键帧456的共视关系
        int kf_id = 456;
        std::cout << "检查KF456的共视关系：" << std::endl;
        
        // 检查456作为主键的情况
        if (data_.covisibility.find(kf_id) != data_.covisibility.end()) {
            const auto& connections = data_.covisibility[kf_id];
            std::cout << "  KF456 -> 其他KF: " << connections.size() << " 个连接" << std::endl;
            for (const auto& conn : connections) {
                if (conn.first <= 5) {  // 只打印前几个
                    std::cout << "    KF456 -> KF" << conn.first << ": " << conn.second << std::endl;
                }
            }
        }
        
        // 检查其他关键帧指向456的情况
        std::cout << "  其他KF -> KF456 的连接：" << std::endl;
        for (const auto& kf_conn : data_.covisibility) {
            int other_kf = kf_conn.first;
            const auto& connections = kf_conn.second;
            
            if (connections.find(kf_id) != connections.end()) {
                int weight = connections.at(kf_id);
                if (other_kf <= 5) {  // 只打印前几个
                    std::cout << "    KF" << other_kf << " -> KF456: " << weight << std::endl;
                }
            }
        }
        
        // 具体检查456->0, 456->1, 456->2的权重
        std::vector<int> target_kfs = {0, 1, 2};
        for (int target_kf : target_kfs) {
            int weight = 0;
            
            // 检查456->target方向
            if (data_.covisibility.find(456) != data_.covisibility.end()) {
                const auto& conn456 = data_.covisibility[456];
                if (conn456.find(target_kf) != conn456.end()) {
                    weight = conn456.at(target_kf);
                    std::cout << "  KF456 -> KF" << target_kf << ": " << weight << " (方向1)" << std::endl;
                }
            }
            
            // 检查target->456方向
            if (weight == 0 && data_.covisibility.find(target_kf) != data_.covisibility.end()) {
                const auto& conn_target = data_.covisibility[target_kf];
                if (conn_target.find(456) != conn_target.end()) {
                    weight = conn_target.at(456);
                    std::cout << "  KF" << target_kf << " -> KF456: " << weight << " (方向2)" << std::endl;
                }
            }
            
            if (weight == 0) {
                std::cout << "  KF456 <-> KF" << target_kf << ": 未找到共视关系！" << std::endl;
            }
        }
    }




    void AddNormalEdgeConstraints() {
        std::cout << "\n开始添加正常边约束（生成树）..." << std::endl;
        
        // 信息矩阵（与回环约束相同）
        Eigen::Matrix<double, 6, 6> information = Eigen::Matrix<double, 6, 6>::Identity();
        
        int normal_edges_added = 0;
        int skipped_edges = 0;
        
        // 遍历所有关键帧
        for (const auto& kf_pair : data_.keyframes) {
            int kf_id = kf_pair.first;
            auto& kf = kf_pair.second;
            
            // 跳过坏的关键帧
            if (kf->is_bad) continue;
            
            // 获取父关键帧ID
            int parent_id = kf->parent_id;
            
            // 如果没有父关键帧，或者父关键帧就是自己，跳过
            if (parent_id < 0 || parent_id == kf_id || 
                data_.keyframes.find(parent_id) == data_.keyframes.end() || 
                data_.keyframes[parent_id]->is_bad) {
                skipped_edges++;
                continue;
            }
            
            // 验证不是自循环（冗余检查）
            if (data_.keyframes[kf_id]->se3_state.data() == data_.keyframes[parent_id]->se3_state.data()) {
                std::cout << "  警告：检测到自循环约束 KF" << kf_id << " -> Parent" << parent_id << std::endl;
                skipped_edges++;
                continue;
            }
            
            // 获取当前关键帧的逆变换 Swi
            Eigen::Matrix4d T_iw = Eigen::Matrix4d::Identity();
            
            if (data_.non_corrected_poses.find(kf_id) != data_.non_corrected_poses.end()) {
                const auto& non_corrected_i = data_.non_corrected_poses[kf_id];
                Eigen::Matrix4d T_i = Eigen::Matrix4d::Identity();
                T_i.block<3, 3>(0, 0) = non_corrected_i.quaternion.toRotationMatrix();
                T_i.block<3, 1>(0, 3) = non_corrected_i.translation;
                T_iw = T_i.inverse();
            } else {
                Eigen::Matrix4d T_i = Eigen::Matrix4d::Identity();
                T_i.block<3, 3>(0, 0) = kf->quaternion.toRotationMatrix();
                T_i.block<3, 1>(0, 3) = kf->translation;
                T_iw = T_i.inverse();
            }
            
            // 获取父关键帧的变换 Sjw
            Eigen::Matrix4d T_j = Eigen::Matrix4d::Identity();
            
            if (data_.non_corrected_poses.find(parent_id) != data_.non_corrected_poses.end()) {
                const auto& non_corrected_parent = data_.non_corrected_poses[parent_id];
                T_j.block<3, 3>(0, 0) = non_corrected_parent.quaternion.toRotationMatrix();
                T_j.block<3, 1>(0, 3) = non_corrected_parent.translation;
            } else {
                T_j.block<3, 3>(0, 0) = data_.keyframes[parent_id]->quaternion.toRotationMatrix();
                T_j.block<3, 1>(0, 3) = data_.keyframes[parent_id]->translation;
            }
            
            // 计算相对变换 Sji = Sjw * Swi
            Eigen::Matrix4d T_ji = T_j * T_iw;
            
            // 添加约束
            ceres::CostFunction* cost_function = SE3LoopConstraintCost::Create(T_ji, information);
            problem_->AddResidualBlock(cost_function, nullptr, 
                                      data_.keyframes[kf_id]->se3_state.data(),      // 子节点
                                      data_.keyframes[parent_id]->se3_state.data()); // 父节点
            
            normal_edges_added++;
            
            if (normal_edges_added % 100 == 0) {
                std::cout << "  已添加 " << normal_edges_added << " 条生成树边约束..." << std::endl;
            }
        }
        
        std::cout << "添加了 " << normal_edges_added << " 条生成树边约束，跳过了 " << skipped_edges << " 条无效边" << std::endl;
    }

    // 获取参数块信息
    void PrintProblemInfo() {
        std::cout << "\n优化问题信息:" << std::endl;
        std::cout << "参数块数量: " << problem_->NumParameterBlocks() << std::endl;
        std::cout << "残差块数量: " << problem_->NumResidualBlocks() << std::endl;
        std::cout << "参数数量: " << problem_->NumParameters() << std::endl;
        std::cout << "残差数量: " << problem_->NumResiduals() << std::endl;
    }
    
    
    // 输出优化后的姿态
    void OutputOptimizedPoses(const std::string& output_file) {
        std::ofstream file(output_file);
        if (!file.is_open()) {
            std::cerr << "无法创建输出文件: " << output_file << std::endl;
            return;
        }
        
        file << "# KF_ID tx ty tz qx qy qz qw" << std::endl;
        
        for (const auto& kf_pair : data_.keyframes) {
            const auto& kf = kf_pair.second;
            if (kf->is_bad) continue;
            
            const auto& state = kf->se3_state;
            file << kf->id << " "
                 << state[0] << " " << state[1] << " " << state[2] << " "
                 << state[3] << " " << state[4] << " " << state[5] << " " << state[6] << std::endl;
        }
        
        std::cout << "优化后的姿态已保存到: " << output_file << std::endl;
    }

    // 输出优化后的Twc格式姿态（相机在世界坐标系中的位置）
    void OutputOptimizedPosesTwc(const std::string& output_file) {
        std::ofstream file(output_file);
        if (!file.is_open()) {
            std::cerr << "无法创建输出文件: " << output_file << std::endl;
            return;
        }
        
        file << "# KF_ID tx ty tz qx qy qz qw (Twc format - 相机在世界坐标系中的位置)" << std::endl;
        
        for (const auto& kf_pair : data_.keyframes) {
            const auto& kf = kf_pair.second;
            if (kf->is_bad) continue;
            
            // 从SE3状态获取Tcw
            Eigen::Vector3d t_cw(kf->se3_state[0], kf->se3_state[1], kf->se3_state[2]);
            Eigen::Quaterniond q_cw(kf->se3_state[6], kf->se3_state[3], kf->se3_state[4], kf->se3_state[5]);
            
            // 构建Tcw变换矩阵
            Eigen::Matrix4d T_cw = Eigen::Matrix4d::Identity();
            T_cw.block<3, 3>(0, 0) = q_cw.toRotationMatrix();
            T_cw.block<3, 1>(0, 3) = t_cw;
            
            // 计算Twc = Tcw^(-1)
            Eigen::Matrix4d T_wc = T_cw.inverse();
            
            // 提取Twc的平移和旋转
            Eigen::Vector3d t_wc = T_wc.block<3, 1>(0, 3);
            Eigen::Matrix3d R_wc = T_wc.block<3, 3>(0, 0);
            Eigen::Quaterniond q_wc(R_wc);
            
            // 输出Twc格式
            file << kf->id << " "
                 << t_wc.x() << " " << t_wc.y() << " " << t_wc.z() << " "
                 << q_wc.x() << " " << q_wc.y() << " " << q_wc.z() << " " << q_wc.w() << std::endl;
        }
        
        std::cout << "优化后的Twc姿态已保存到: " << output_file << std::endl;
    }
    
    // 同时输出Tcw和Twc格式
    void OutputBothFormats(const std::string& output_dir, const std::string& suffix = "") {
        std::string tcw_file = output_dir + "/poses_tcw" + suffix + ".txt";
        std::string twc_file = output_dir + "/poses_twc" + suffix + ".txt";
        
        // 输出Tcw格式
        OutputOptimizedPoses(tcw_file);
        
        // 输出Twc格式  
        OutputOptimizedPosesTwc(twc_file);
    }

    // 输出优化后的Twc格式姿态（TUM格式，按时间戳排序）
    void OutputOptimizedPosesTwcTUM(const std::string& output_file) {
        std::ofstream file(output_file);
        if (!file.is_open()) {
            std::cerr << "无法创建输出文件: " << output_file << std::endl;
            return;
        }
        
        // TUM格式头部
        // file << "# TUM trajectory format (Twc - camera pose in world frame)" << std::endl;
        // file << "# timestamp tx ty tz qx qy qz qw" << std::endl;
        
        // 收集所有关键帧的时间戳和姿态
        struct KeyFramePose {
            double timestamp;
            int kf_id;
            Eigen::Vector3d position_wc;
            Eigen::Quaterniond quaternion_wc;
        };
        
        std::vector<KeyFramePose> kf_poses;
        
        for (const auto& kf_pair : data_.keyframes) {
            const auto& kf = kf_pair.second;
            if (kf->is_bad) continue;
            
            // 从SE3状态获取Tcw
            Eigen::Vector3d t_cw(kf->se3_state[0], kf->se3_state[1], kf->se3_state[2]);
            Eigen::Quaterniond q_cw(kf->se3_state[6], kf->se3_state[3], kf->se3_state[4], kf->se3_state[5]);
            
            // 构建Tcw变换矩阵
            Eigen::Matrix4d T_cw = Eigen::Matrix4d::Identity();
            T_cw.block<3, 3>(0, 0) = q_cw.toRotationMatrix();
            T_cw.block<3, 1>(0, 3) = t_cw;
            
            // 计算Twc = Tcw^(-1)
            Eigen::Matrix4d T_wc = T_cw.inverse();
            
            // 提取Twc的平移和旋转
            Eigen::Vector3d t_wc = T_wc.block<3, 1>(0, 3);
            Eigen::Matrix3d R_wc = T_wc.block<3, 3>(0, 0);
            Eigen::Quaterniond q_wc(R_wc);
            
            // 创建KeyFramePose对象
            KeyFramePose kf_pose;
            kf_pose.timestamp = kf->timestamp;
            kf_pose.kf_id = kf->id;
            kf_pose.position_wc = t_wc;
            kf_pose.quaternion_wc = q_wc;
            
            kf_poses.push_back(kf_pose);
        }
        
        // 按时间戳排序
        std::sort(kf_poses.begin(), kf_poses.end(), 
                  [](const KeyFramePose& a, const KeyFramePose& b) {
                      return a.timestamp < b.timestamp;
                  });
        
        // 输出TUM格式
        for (const auto& kf_pose : kf_poses) {
            // TUM格式：timestamp tx ty tz qx qy qz qw
            file << std::fixed << std::setprecision(9) << kf_pose.timestamp << " "
                 << std::setprecision(9)
                 << kf_pose.position_wc.x() << " " 
                 << kf_pose.position_wc.y() << " " 
                 << kf_pose.position_wc.z() << " "
                 << kf_pose.quaternion_wc.x() << " " 
                 << kf_pose.quaternion_wc.y() << " " 
                 << kf_pose.quaternion_wc.z() << " " 
                 << kf_pose.quaternion_wc.w() << std::endl;
        }
        
        std::cout << "优化后的TUM格式Twc轨迹已保存到: " << output_file << std::endl;
        std::cout << "包含 " << kf_poses.size() << " 个关键帧，按时间戳排序" << std::endl;
    }
    
    // 输出优化前的Twc格式姿态（TUM格式，按时间戳排序）
    void OutputInitialPosesTwcTUM(const std::string& output_file) {
        std::ofstream file(output_file);
        if (!file.is_open()) {
            std::cerr << "无法创建输出文件: " << output_file << std::endl;
            return;
        }
        
        // TUM格式头部
        // file << "# TUM trajectory format (Twc - camera pose in world frame)" << std::endl;
        // file << "# timestamp tx ty tz qx qy qz qw" << std::endl;
        
        // 收集所有关键帧的时间戳和姿态
        struct KeyFramePose {
            double timestamp;
            int kf_id;
            Eigen::Vector3d position_wc;
            Eigen::Quaterniond quaternion_wc;
        };
        
        std::vector<KeyFramePose> kf_poses;
        
        for (const auto& kf_pair : data_.keyframes) {
            const auto& kf = kf_pair.second;
            if (kf->is_bad) continue;
            
            // 检查是否有修正姿态，如果有则使用修正姿态，否则使用当前姿态
            Eigen::Vector3d t_cw;
            Eigen::Quaterniond q_cw;
            
            if (data_.corrected_poses.find(kf->id) != data_.corrected_poses.end()) {
                const auto& corrected = data_.corrected_poses[kf->id];
                t_cw = corrected.translation;
                q_cw = corrected.quaternion;
            } else {
                // 使用当前姿态作为初始姿态
                t_cw = kf->translation;
                q_cw = kf->quaternion;
            }
            
            // 构建Tcw变换矩阵
            Eigen::Matrix4d T_cw = Eigen::Matrix4d::Identity();
            T_cw.block<3, 3>(0, 0) = q_cw.toRotationMatrix();
            T_cw.block<3, 1>(0, 3) = t_cw;
            
            // 计算Twc = Tcw^(-1)
            Eigen::Matrix4d T_wc = T_cw.inverse();
            
            // 提取Twc的平移和旋转
            Eigen::Vector3d t_wc = T_wc.block<3, 1>(0, 3);
            Eigen::Matrix3d R_wc = T_wc.block<3, 3>(0, 0);
            Eigen::Quaterniond q_wc(R_wc);
            
            // 创建KeyFramePose对象
            KeyFramePose kf_pose;
            kf_pose.timestamp = kf->timestamp;
            kf_pose.kf_id = kf->id;
            kf_pose.position_wc = t_wc;
            kf_pose.quaternion_wc = q_wc;
            
            kf_poses.push_back(kf_pose);
        }
        
        // 按时间戳排序
        std::sort(kf_poses.begin(), kf_poses.end(), 
                  [](const KeyFramePose& a, const KeyFramePose& b) {
                      return a.timestamp < b.timestamp;
                  });
        
        // 输出TUM格式
        for (const auto& kf_pose : kf_poses) {
            // TUM格式：timestamp tx ty tz qx qy qz qw
            file << std::fixed << std::setprecision(9) << kf_pose.timestamp << " "
                 << std::setprecision(9)
                 << kf_pose.position_wc.x() << " " 
                 << kf_pose.position_wc.y() << " " 
                 << kf_pose.position_wc.z() << " "
                 << kf_pose.quaternion_wc.x() << " " 
                 << kf_pose.quaternion_wc.y() << " " 
                 << kf_pose.quaternion_wc.z() << " " 
                 << kf_pose.quaternion_wc.w() << std::endl;
        }
        
        std::cout << "优化前的TUM格式Twc轨迹已保存到: " << output_file << std::endl;
        std::cout << "包含 " << kf_poses.size() << " 个关键帧，按时间戳排序" << std::endl;
    }
    
    // 输出TUM格式的轨迹文件
    void OutputTUMTrajectory(const std::string& output_dir) {
        std::string tum_before_file = output_dir + "/trajectory_before_optimization.txt";
        std::string tum_after_file = output_dir + "/trajectory_after_optimization.txt";
        
        // 输出优化前的TUM格式轨迹
        OutputInitialPosesTwcTUM(tum_before_file);
        
        // 输出优化后的TUM格式轨迹
        OutputOptimizedPosesTwcTUM(tum_after_file);
    }

    // 获取关键帧信息用于调试
    void PrintKeyFrameInfo(int id) {
        if (data_.keyframes.find(id) != data_.keyframes.end()) {
            const auto& kf = data_.keyframes[id];
            std::cout << "关键帧 " << id << ":" << std::endl;
            std::cout << "  位置: [" << kf->translation.transpose() << "]" << std::endl;
            std::cout << "  四元数: [" << kf->quaternion.x() << ", " << kf->quaternion.y() 
                      << ", " << kf->quaternion.z() << ", " << kf->quaternion.w() << "]" << std::endl;
            std::cout << "  固定: " << (kf->is_fixed ? "是" : "否") << std::endl;
            std::cout << "  坏帧: " << (kf->is_bad ? "是" : "否") << std::endl;
        }
    }


    // 打印优化结果
    void PrintOptimizationResults() {
        std::cout << "\n=== 优化结果分析 ===" << std::endl;
        
        // 打印几个关键帧的优化前后对比
        std::vector<int> key_frames = {0, data_.loop_kf_id, data_.current_kf_id};
        
        for (int kf_id : key_frames) {
            if (data_.keyframes.find(kf_id) == data_.keyframes.end()) continue;
            
            auto kf = data_.keyframes[kf_id];
            
            // 优化前的姿态（从corrected_poses或原始姿态）
            Eigen::Vector3d pos_before;
            Eigen::Quaterniond quat_before;
            
            if (data_.corrected_poses.find(kf_id) != data_.corrected_poses.end()) {
                pos_before = data_.corrected_poses[kf_id].translation;
                quat_before = data_.corrected_poses[kf_id].quaternion;
            } else {
                // 使用初始姿态
                std::ifstream file("/Datasets/CERES_Work/input/optimization_data/keyframe_poses.txt");
                std::string line;
                while (std::getline(file, line)) {
                    if (line.empty() || line[0] == '#') continue;
                    std::istringstream iss(line);
                    int id;
                    double timestamp, tx, ty, tz, qx, qy, qz, qw;
                    if (iss >> id >> timestamp >> tx >> ty >> tz >> qx >> qy >> qz >> qw && id == kf_id) {
                        pos_before = Eigen::Vector3d(tx, ty, tz);
                        quat_before = Eigen::Quaterniond(qw, qx, qy, qz);
                        break;
                    }
                }
            }
            
            // 优化后的姿态
            Eigen::Vector3d pos_after(kf->se3_state[0], kf->se3_state[1], kf->se3_state[2]);
            Eigen::Quaterniond quat_after(kf->se3_state[6], kf->se3_state[3], kf->se3_state[4], kf->se3_state[5]);
            
            // 计算位置变化
            double pos_change = (pos_after - pos_before).norm();
            
            // 计算旋转变化（角度）
            Eigen::Quaterniond quat_diff = quat_before.inverse() * quat_after;
            double angle_change = 2.0 * acos(std::abs(quat_diff.w())) * 180.0 / M_PI;
            
            std::cout << "关键帧 " << kf_id << ":" << std::endl;
            std::cout << "  位置变化: " << pos_change << " 米" << std::endl;
            std::cout << "  旋转变化: " << angle_change << " 度" << std::endl;
            std::cout << "  优化前位置: [" << pos_before.transpose() << "]" << std::endl;
            std::cout << "  优化后位置: [" << pos_after.transpose() << "]" << std::endl;
            std::cout << std::endl;
        }
        
        // 计算所有关键帧的平均位置变化
        double total_pos_change = 0.0;
        int count = 0;
        
        for (const auto& kf_pair : data_.keyframes) {
            if (kf_pair.second->is_bad) continue;
            
            // 这里简化，假设没有修正姿态的帧位置变化为0
            if (data_.corrected_poses.find(kf_pair.first) != data_.corrected_poses.end()) {
                const auto& before = data_.corrected_poses[kf_pair.first];
                Eigen::Vector3d after(kf_pair.second->se3_state[0], 
                                    kf_pair.second->se3_state[1], 
                                    kf_pair.second->se3_state[2]);
                total_pos_change += (after - before.translation).norm();
                count++;
            }
        }
        
        if (count > 0) {
            std::cout << "平均位置变化: " << total_pos_change / count << " 米" << std::endl;
        }
        
        std::cout << "参与优化的关键帧数: " << count << std::endl;
    }


    bool OptimizeEssentialGraph() {
        std::cout << "\n开始本质图优化..." << std::endl;
        
        // 配置求解器选项
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY; // 可以尝试不同的求解器
        options.minimizer_progress_to_stdout = true;
        options.max_num_iterations = 100;  // 增加迭代次数
        options.num_threads = 4;
        options.function_tolerance = 1e-6;
        options.gradient_tolerance = 1e-8;
        options.parameter_tolerance = 1e-8;
        
        // 求解
        ceres::Solver::Summary summary;
        ceres::Solve(options, problem_.get(), &summary);
        
        std::cout << "\n=== 优化详细报告 ===" << std::endl;
        std::cout << summary.FullReport() << std::endl;
        
        // 更新所有关键帧的姿态
        for (auto& kf_pair : data_.keyframes) {
            kf_pair.second->UpdateFromState();
        }
        
        return summary.IsSolutionUsable();
    }
};



int main() {
    // 数据文件路径
    std::string data_dir = "/Datasets/CERES_Work/input/optimization_data";
    std::string output_dir = "/Datasets/CERES_Work/output";
    
    // 创建输出目录
    system(("mkdir -p " + output_dir).c_str());
    
    // 创建优化器
    ORBSlamLoopOptimizer optimizer;
    
    // 解析所有数据文件
    if (!optimizer.ParseAllData(data_dir)) {
        std::cerr << "数据解析失败" << std::endl;
        return -1;
    }
    
    // 设置优化问题
    optimizer.SetupOptimizationProblem();
    
    // 调试共视关系数据
    optimizer.DebugCovisibilityData();
    // 添加回环约束
    optimizer.AddLoopConstraints();
    

    // 添加正常边约束（生成树）
    optimizer.AddNormalEdgeConstraints();
    
    // 打印问题信息
    optimizer.PrintProblemInfo();
    
    // 打印一些关键帧信息用于调试
    optimizer.PrintKeyFrameInfo(0);  // 初始关键帧
    
    // std::cout << "\n注意：当前版本只添加了关键帧顶点，约束部分将在后续版本中添加" << std::endl;
    
    // // 输出初始姿态（用于验证）
    // optimizer.OutputOptimizedPoses(output_dir + "/initial_poses.txt");
    
    // 输出优化前的姿态（两种格式）
    std::cout << "\n保存优化前姿态..." << std::endl;
    optimizer.OutputBothFormats(output_dir, "_before_optimization");
    
    // 执行优化！
    std::cout << "\n=== 开始执行回环优化 ===" << std::endl;
    
    bool success = optimizer.OptimizeEssentialGraph();
    
    if (success) {
        std::cout << "\n=== 优化成功完成 ===" << std::endl;
        
        // // 输出优化后的姿态
        // optimizer.OutputOptimizedPoses(output_dir + "/poses_after_optimization.txt");


        // 输出TUM格式的轨迹文件（按时间戳排序）
        std::cout << "\n保存TUM格式轨迹文件..." << std::endl;
        optimizer.OutputTUMTrajectory(output_dir);
        
        // 输出优化后的姿态（两种格式）
        std::cout << "\n保存优化后姿态..." << std::endl;
        optimizer.OutputBothFormats(output_dir, "_after_optimization");
        
        // 打印一些关键帧的优化前后对比
        optimizer.PrintOptimizationResults();
        
        std::cout << "\n输出文件说明:" << std::endl;
        std::cout << "- trajectory_before_optimization.txt: 优化前TUM格式轨迹（按时间戳排序）" << std::endl;
        std::cout << "- trajectory_after_optimization.txt: 优化后TUM格式轨迹（按时间戳排序）" << std::endl;
        std::cout << "- poses_tcw_*.txt: Tcw格式（世界到相机的变换）" << std::endl;
        std::cout << "- poses_twc_*.txt: Twc格式（相机在世界坐标系中的位置）" << std::endl;
        
    } else {
        std::cout << "\n=== 优化失败 ===" << std::endl;
        return -1;
    }
    
    
    return 0;
}
