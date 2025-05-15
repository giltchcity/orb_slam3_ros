#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <unordered_map>
#include <memory>

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


// SE3相对姿态约束 - 对应g2o的Edge4DoF
class SE3RelativePoseCost {
public:
    SE3RelativePoseCost(const Eigen::Matrix4d& relative_transform, const Eigen::Matrix<double, 6, 6>& information)
        : relative_rotation_(relative_transform.block<3, 3>(0, 0)),
          relative_translation_(relative_transform.block<3, 1>(0, 3)),
          sqrt_information_(information.llt().matrixL()) {
    }
    
    template <typename T>
    bool operator()(const T* const pose_i, const T* const pose_j, T* residuals) const {
        // 提取姿态
        Eigen::Matrix<T, 3, 1> t_i(pose_i[0], pose_i[1], pose_i[2]);
        Eigen::Matrix<T, 3, 1> t_j(pose_j[0], pose_j[1], pose_j[2]);
        Eigen::Quaternion<T> q_i(pose_i[6], pose_i[3], pose_i[4], pose_i[5]);
        Eigen::Quaternion<T> q_j(pose_j[6], pose_j[3], pose_j[4], pose_j[5]);
        
        // 转换为旋转矩阵
        Eigen::Matrix<T, 3, 3> R_i = q_i.toRotationMatrix();
        Eigen::Matrix<T, 3, 3> R_j = q_j.toRotationMatrix();
        
        // 预期的相对变换
        Eigen::Matrix<T, 3, 3> R_ij = relative_rotation_.cast<T>();
        Eigen::Matrix<T, 3, 1> t_ij = relative_translation_.cast<T>();
        
        // 按照ORB-SLAM3的公式计算error
        // 旋转error: LogSO3(R_i * R_j^T * R_ij^T)
        Eigen::Matrix<T, 3, 3> R_error_mat = R_i * R_j.transpose() * R_ij.transpose();
        Eigen::Matrix<T, 3, 1> rotation_error = LogSO3(R_error_mat);
        
        // 平移error: R_i * (-R_j^T * t_j) + t_i - t_ij
        Eigen::Matrix<T, 3, 1> translation_error = 
            R_i * (-R_j.transpose() * t_j) + t_i - t_ij;
        
        // 组合residuals
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
        return new ceres::AutoDiffCostFunction<SE3RelativePoseCost, 6, 7, 7>(
            new SE3RelativePoseCost(relative_transform, information));
    }
    
private:
    const Eigen::Matrix3d relative_rotation_;
    const Eigen::Vector3d relative_translation_;
    const Eigen::Matrix<double, 6, 6> sqrt_information_;
};


// 生成树约束 - 对应g2o的EdgeSim3
class SE3SpanningTreeCost {
public:
    SE3SpanningTreeCost(const Eigen::Matrix4d& relative_transform, const Eigen::Matrix<double, 6, 6>& information)
        : relative_rotation_(relative_transform.block<3, 3>(0, 0)),
          relative_translation_(relative_transform.block<3, 1>(0, 3)),
          sqrt_information_(information.llt().matrixL()) {
    }
    
    template <typename T>
    bool operator()(const T* const pose_i, const T* const pose_j, T* residuals) const {
        // pose_i: 子关键帧, pose_j: 父关键帧
        // 格式: [tx, ty, tz, qx, qy, qz, qw]
        
        // 提取姿态
        Eigen::Matrix<T, 3, 1> t_i(pose_i[0], pose_i[1], pose_i[2]);
        Eigen::Matrix<T, 3, 1> t_j(pose_j[0], pose_j[1], pose_j[2]);
        Eigen::Quaternion<T> q_i(pose_i[6], pose_i[3], pose_i[4], pose_i[5]);
        Eigen::Quaternion<T> q_j(pose_j[6], pose_j[3], pose_j[4], pose_j[5]);
        
        // 转换为旋转矩阵
        Eigen::Matrix<T, 3, 3> R_i = q_i.toRotationMatrix();
        Eigen::Matrix<T, 3, 3> R_j = q_j.toRotationMatrix();
        
        // 计算相对变换 T_ji = T_j * T_i^{-1}
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
        return new ceres::AutoDiffCostFunction<SE3SpanningTreeCost, 6, 7, 7>(
            new SE3SpanningTreeCost(relative_transform, information));
    }
    
private:
    const Eigen::Matrix3d relative_rotation_;
    const Eigen::Vector3d relative_translation_;
    const Eigen::Matrix<double, 6, 6> sqrt_information_;
};

// 共视约束 - 对应g2o的EdgeSim3（用于共视关系）
class SE3CovisibilityCost {
public:
    SE3CovisibilityCost(const Eigen::Matrix4d& relative_transform, const Eigen::Matrix<double, 6, 6>& information)
        : relative_rotation_(relative_transform.block<3, 3>(0, 0)),
          relative_translation_(relative_transform.block<3, 1>(0, 3)),
          sqrt_information_(information.llt().matrixL()) {
    }
    
    template <typename T>
    bool operator()(const T* const pose_i, const T* const pose_n, T* residuals) const {
        // pose_i: 当前关键帧, pose_n: 共视关键帧
        // 格式: [tx, ty, tz, qx, qy, qz, qw]
        
        // 提取姿态
        Eigen::Matrix<T, 3, 1> t_i(pose_i[0], pose_i[1], pose_i[2]);
        Eigen::Matrix<T, 3, 1> t_n(pose_n[0], pose_n[1], pose_n[2]);
        Eigen::Quaternion<T> q_i(pose_i[6], pose_i[3], pose_i[4], pose_i[5]);
        Eigen::Quaternion<T> q_n(pose_n[6], pose_n[3], pose_n[4], pose_n[5]);
        
        // 转换为旋转矩阵
        Eigen::Matrix<T, 3, 3> R_i = q_i.toRotationMatrix();
        Eigen::Matrix<T, 3, 3> R_n = q_n.toRotationMatrix();
        
        // 计算相对变换 T_ni = T_n * T_i^{-1}
        Eigen::Matrix<T, 3, 3> R_ni = R_n * R_i.transpose();
        Eigen::Matrix<T, 3, 1> t_ni = R_n * (R_i.transpose() * (-t_i)) + t_n;
        
        // 预期的相对变换
        Eigen::Matrix<T, 3, 3> R_expected = relative_rotation_.cast<T>();
        Eigen::Matrix<T, 3, 1> t_expected = relative_translation_.cast<T>();
        
        // 计算旋转误差
        Eigen::Matrix<T, 3, 3> R_error_mat = R_expected.transpose() * R_ni;
        Eigen::Matrix<T, 3, 1> rotation_error = LogSO3(R_error_mat);
        
        // 计算平移误差
        Eigen::Matrix<T, 3, 1> translation_error = t_ni - t_expected;
        
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
        return new ceres::AutoDiffCostFunction<SE3CovisibilityCost, 6, 7, 7>(
            new SE3CovisibilityCost(relative_transform, information));
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
    void AddLoopConstraints() {
        std::cout << "\n开始添加回环约束..." << std::endl;
        
        const int minFeat = 100; // 最小特征点数阈值
        
        // 设置信息矩阵 - 对应g2o代码中的matLambda
        Eigen::Matrix<double, 6, 6> information = Eigen::Matrix<double, 6, 6>::Identity();
        information(0, 0) = 1e3;  // x轴旋转权重
        information(1, 1) = 1e3;  // y轴旋转权重
        information(2, 2) = 1e3;  // z轴旋转权重 (原代码中这行重复了，我改为z轴)
        // 平移部分使用默认权重1.0
        
        int loop_edges_added = 0;
        
        // 遍历所有回环连接
        for (const auto& connection : data_.loop_connections) {
            int kf_i_id = connection.first;
            const auto& connected_kfs = connection.second;
            
            // 检查关键帧i是否存在且不是坏帧
            if (data_.keyframes.find(kf_i_id) == data_.keyframes.end() || 
                data_.keyframes[kf_i_id]->is_bad) {
                continue;
            }
            
            auto kf_i = data_.keyframes[kf_i_id];
            
            // 获取关键帧i的修正后姿态（如果有的话）
            Eigen::Matrix4d T_i = Eigen::Matrix4d::Identity();
            if (data_.corrected_poses.find(kf_i_id) != data_.corrected_poses.end()) {
                const auto& corrected_kf = data_.corrected_poses[kf_i_id];
                T_i.block<3, 3>(0, 0) = corrected_kf.quaternion.toRotationMatrix();
                T_i.block<3, 1>(0, 3) = corrected_kf.translation;
            } else {
                T_i.block<3, 3>(0, 0) = kf_i->quaternion.toRotationMatrix();
                T_i.block<3, 1>(0, 3) = kf_i->translation;
            }
            
            // 遍历与关键帧i连接的所有关键帧
            for (int kf_j_id : connected_kfs) {
                // 检查关键帧j是否存在且不是坏帧
                if (data_.keyframes.find(kf_j_id) == data_.keyframes.end() || 
                    data_.keyframes[kf_j_id]->is_bad) {
                    continue;
                }
                
                auto kf_j = data_.keyframes[kf_j_id];
                
                // 检查权重条件（除了当前关键帧和回环关键帧的组合）
                // if (!((kf_i_id == data_.current_kf_id && kf_j_id == data_.loop_kf_id) ||
                //       (kf_j_id == data_.current_kf_id && kf_i_id == data_.loop_kf_id))) {
                //     // 这里简化处理，假设所有回环连接都有足够的权重
                //     // 实际ORB-SLAM3会检查GetWeight()，但我们的数据中没有直接的权重信息
                // }
                
                // 检查权重条件（除了当前关键帧和回环关键帧的组合）
                if (!((kf_i_id == data_.current_kf_id && kf_j_id == data_.loop_kf_id) ||
                      (kf_j_id == data_.current_kf_id && kf_i_id == data_.loop_kf_id))) {
                    
                    // 查询共视权重
                    int weight = 0;
                    if (data_.covisibility.find(kf_i_id) != data_.covisibility.end() &&
                        data_.covisibility[kf_i_id].find(kf_j_id) != data_.covisibility[kf_i_id].end()) {
                        weight = data_.covisibility[kf_i_id][kf_j_id];
                    } else if (data_.covisibility.find(kf_j_id) != data_.covisibility.end() &&
                               data_.covisibility[kf_j_id].find(kf_i_id) != data_.covisibility[kf_j_id].end()) {
                        weight = data_.covisibility[kf_j_id][kf_i_id];
                    }
                    
                    // 应用minFeat阈值
                    if (weight < minFeat) {
                        continue;
                    }
                }
                
                // 获取关键帧j的修正后姿态
                Eigen::Matrix4d T_j = Eigen::Matrix4d::Identity();
                if (data_.corrected_poses.find(kf_j_id) != data_.corrected_poses.end()) {
                    const auto& corrected_kf = data_.corrected_poses[kf_j_id];
                    T_j.block<3, 3>(0, 0) = corrected_kf.quaternion.toRotationMatrix();
                    T_j.block<3, 1>(0, 3) = corrected_kf.translation;
                } else {
                    T_j.block<3, 3>(0, 0) = kf_j->quaternion.toRotationMatrix();
                    T_j.block<3, 1>(0, 3) = kf_j->translation;
                }
                
                // 计算相对变换 T_ij = T_i^{-1} * T_j
                Eigen::Matrix4d T_ij = T_i.inverse() * T_j;
                
                // 创建相对姿态约束
                ceres::CostFunction* cost_function = SE3RelativePoseCost::Create(T_ij, information);
                
                // 添加到优化问题
                problem_->AddResidualBlock(cost_function,
                                         nullptr,  // 不使用鲁棒核函数
                                         kf_i->se3_state.data(),
                                         kf_j->se3_state.data());
                
                loop_edges_added++;
                
                // 调试输出
                if (loop_edges_added <= 5) {
                    std::cout << "添加回环约束: " << kf_i_id << " <-> " << kf_j_id << std::endl;
                }
            }
        }
        
        std::cout << "共添加了 " << loop_edges_added << " 个回环约束" << std::endl;
    }


    // 添加正常边约束（生成树 + 共视）
    void AddNormalEdgeConstraints() {
        std::cout << "\n开始添加正常边约束（生成树 + 共视）..." << std::endl;
        
        const int minFeat = 100;
        
        // 信息矩阵
        Eigen::Matrix<double, 6, 6> information = Eigen::Matrix<double, 6, 6>::Identity();
        information(0, 0) = 1e3;
        information(1, 1) = 1e3;
        information(2, 2) = 1e3;
        
        int spanning_tree_edges = 0;
        int covisibility_edges = 0;
        std::set<std::pair<int, int>> inserted_edges;
        
        // 遍历所有关键帧 - 完全按照ORB-SLAM3的结构
        for (const auto& kf_pair : data_.keyframes) {
            auto kf_i = kf_pair.second;
            int nIDi = kf_i->id;
            
            if (kf_i->is_bad) continue;
            
            // 计算Swi（世界到当前帧的变换的逆）- 对应ORB-SLAM3的逻辑
            Eigen::Matrix4d T_wi = Eigen::Matrix4d::Identity();
            if (data_.non_corrected_poses.find(nIDi) != data_.non_corrected_poses.end()) {
                const auto& non_corrected_i = data_.non_corrected_poses[nIDi];
                T_wi.block<3, 3>(0, 0) = non_corrected_i.quaternion.toRotationMatrix();
                T_wi.block<3, 1>(0, 3) = non_corrected_i.translation;
            } else {
                T_wi.block<3, 3>(0, 0) = kf_i->quaternion.toRotationMatrix();
                T_wi.block<3, 1>(0, 3) = kf_i->translation;
            }
            Eigen::Matrix4d T_iw = T_wi.inverse(); // 这就是Swi
            
            // 1. Spanning tree edge - 复用T_iw
            if (kf_i->parent_id != -1 && kf_i->parent_id != kf_i->id) {
                if (data_.keyframes.find(kf_i->parent_id) != data_.keyframes.end() &&
                    !data_.keyframes[kf_i->parent_id]->is_bad) {
                    
                    auto kf_j = data_.keyframes[kf_i->parent_id];
                    int nIDj = kf_j->id;
                    
                    // 计算父关键帧的变换
                    Eigen::Matrix4d T_wj = Eigen::Matrix4d::Identity();
                    if (data_.non_corrected_poses.find(nIDj) != data_.non_corrected_poses.end()) {
                        const auto& non_corrected_j = data_.non_corrected_poses[nIDj];
                        T_wj.block<3, 3>(0, 0) = non_corrected_j.quaternion.toRotationMatrix();
                        T_wj.block<3, 1>(0, 3) = non_corrected_j.translation;
                    } else {
                        T_wj.block<3, 3>(0, 0) = kf_j->quaternion.toRotationMatrix();
                        T_wj.block<3, 1>(0, 3) = kf_j->translation;
                    }
                    
                    // T_ji = T_wj * T_iw (对应ORB-SLAM3的 Sji = Sjw * Swi)
                    Eigen::Matrix4d T_ji = T_wj * T_iw;
                    
                    ceres::CostFunction* cost_function = SE3SpanningTreeCost::Create(T_ji, information);
                    problem_->AddResidualBlock(cost_function, nullptr,
                                             kf_i->se3_state.data(),
                                             kf_j->se3_state.data());
                    spanning_tree_edges++;
                }
            }
            
            // 2. Covisibility graph edges - 复用T_iw
            if (data_.covisibility.find(nIDi) != data_.covisibility.end()) {
                for (const auto& covis_pair : data_.covisibility[nIDi]) {
                    int nIDn = covis_pair.first;
                    int weight = covis_pair.second;
                    
                    if (weight < minFeat) continue;
                    if (data_.keyframes.find(nIDn) == data_.keyframes.end() ||
                        data_.keyframes[nIDn]->is_bad) continue;
                    
                    auto kf_n = data_.keyframes[nIDn];
                    
                    // 跳过父关键帧和子关键帧
                    if (kf_i->parent_id == nIDn) continue;
                    
                    // 检查是否为子关键帧
                    bool is_child = false;
                    if (data_.spanning_tree.find(nIDi) != data_.spanning_tree.end()) {
                        for (int child_id : data_.spanning_tree[nIDi]) {
                            if (child_id == nIDn) {
                                is_child = true;
                                break;
                            }
                        }
                    }
                    if (is_child) continue;
                    
                    // 确保 nIDn < nIDi
                    if (nIDn >= nIDi) continue;
                    
                    // 检查是否已添加
                    std::pair<int, int> edge_pair = std::make_pair(std::min(nIDi, nIDn), std::max(nIDi, nIDn));
                    if (inserted_edges.count(edge_pair)) continue;
                    
                    // 计算共视关键帧的变换
                    Eigen::Matrix4d T_wn = Eigen::Matrix4d::Identity();
                    if (data_.non_corrected_poses.find(nIDn) != data_.non_corrected_poses.end()) {
                        const auto& non_corrected_n = data_.non_corrected_poses[nIDn];
                        T_wn.block<3, 3>(0, 0) = non_corrected_n.quaternion.toRotationMatrix();
                        T_wn.block<3, 1>(0, 3) = non_corrected_n.translation;
                    } else {
                        T_wn.block<3, 3>(0, 0) = kf_n->quaternion.toRotationMatrix();
                        T_wn.block<3, 1>(0, 3) = kf_n->translation;
                    }
                    
                    // T_ni = T_wn * T_iw (对应ORB-SLAM3的 Sni = Snw * Swi)
                    Eigen::Matrix4d T_ni = T_wn * T_iw;
                    
                    ceres::CostFunction* cost_function = SE3CovisibilityCost::Create(T_ni, information);
                    problem_->AddResidualBlock(cost_function, nullptr,
                                             kf_i->se3_state.data(),
                                             kf_n->se3_state.data());
                    
                    inserted_edges.insert(edge_pair);
                    covisibility_edges++;
                }
            }
        }
        
        std::cout << "添加生成树约束: " << spanning_tree_edges << " 个" << std::endl;
        std::cout << "添加共视约束: " << covisibility_edges << " 个" << std::endl;
    }


    // 获取参数块信息
    void PrintProblemInfo() {
        std::cout << "\n优化问题信息:" << std::endl;
        std::cout << "参数块数量: " << problem_->NumParameterBlocks() << std::endl;
        std::cout << "残差块数量: " << problem_->NumResidualBlocks() << std::endl;
        std::cout << "参数数量: " << problem_->NumParameters() << std::endl;
        std::cout << "残差数量: " << problem_->NumResiduals() << std::endl;
    }
    
    // // 主要优化函数
    // bool OptimizeEssentialGraph() {
    //     std::cout << "\n开始本质图优化..." << std::endl;
        
    //     // 配置求解器选项
    //     ceres::Solver::Options options;
    //     options.linear_solver_type = ceres::SPARSE_SCHUR;
    //     options.minimizer_progress_to_stdout = true;
    //     options.max_num_iterations = 50;
    //     options.num_threads = 4;
        
    //     // 求解
    //     ceres::Solver::Summary summary;
    //     ceres::Solve(options, problem_.get(), &summary);
        
    //     std::cout << "\n优化完成" << std::endl;
    //     std::cout << summary.BriefReport() << std::endl;
        
    //     // 更新所有关键帧的姿态
    //     for (auto& kf_pair : data_.keyframes) {
    //         kf_pair.second->UpdateFromState();
    //     }
        
    //     return summary.IsSolutionUsable();
    // }
    
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
    
    // 添加回环约束
    optimizer.AddLoopConstraints();
    

    // 添加正常边约束（生成树 + 共视）
    optimizer.AddNormalEdgeConstraints();
    
    // 打印问题信息
    optimizer.PrintProblemInfo();
    
    // 打印一些关键帧信息用于调试
    optimizer.PrintKeyFrameInfo(0);  // 初始关键帧
    
    std::cout << "\n注意：当前版本只添加了关键帧顶点，约束部分将在后续版本中添加" << std::endl;
    
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
        
        // 输出优化后的姿态（两种格式）
        std::cout << "\n保存优化后姿态..." << std::endl;
        optimizer.OutputBothFormats(output_dir, "_after_optimization");
        
        // 打印一些关键帧的优化前后对比
        optimizer.PrintOptimizationResults();
        
        std::cout << "\n输出文件说明:" << std::endl;
        std::cout << "- poses_tcw_*.txt: Tcw格式（世界到相机的变换）" << std::endl;
        std::cout << "- poses_twc_*.txt: Twc格式（相机在世界坐标系中的位置，直接用于可视化）" << std::endl;
        
    } else {
        std::cout << "\n=== 优化失败 ===" << std::endl;
        return -1;
    }
    
    
    return 0;
}
