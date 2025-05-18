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

// SE3 李代数参数化 - 严格按照g2o::Sim3实现但固定尺度为1
class SE3Parameterization : public ceres::Manifold {
public:
    SE3Parameterization() {}
    ~SE3Parameterization() {}
    
    // 6维切空间：[rho(3), phi(3)] - SE3的李代数
    // 7维环境空间：[tx, ty, tz, qx, qy, qz, qw] - SE3的四元数表示
    int AmbientSize() const override { return 7; }
    int TangentSize() const override { return 6; }
    
    // 创建反对称矩阵
    template <typename T>
    static Eigen::Matrix<T, 3, 3> skew(const Eigen::Matrix<T, 3, 1>& v) {
        Eigen::Matrix<T, 3, 3> m;
        m << T(0), -v(2), v(1),
             v(2), T(0), -v(0),
             -v(1), v(0), T(0);
        return m;
    }
    
    // SE3的指数映射 - 严格匹配g2o::Sim3构造函数
    bool Plus(const double* x, const double* delta, double* x_plus_delta) const override {
        // 提取当前状态
        Eigen::Map<const Eigen::Vector3d> t(x);
        Eigen::Map<const Eigen::Quaterniond> q(x + 3);
        
        // 提取增量
        Eigen::Map<const Eigen::Vector3d> omega(delta);    // 旋转增量
        Eigen::Map<const Eigen::Vector3d> upsilon(delta + 3);   // 平移增量
        
        // 进行指数映射，严格匹配g2o::Sim3构造函数
        double sigma = 0.0; // 固定尺度为1，对应log(1) = 0
        
        // 计算旋转部分
        double theta = omega.norm();
        Eigen::Matrix3d Omega = skew(omega);
        Eigen::Matrix3d R;
        Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
        
        // 以下变量用于计算平移部分
        double A, B, C;
        C = 1.0;  // 因为exp(sigma) = 1.0 (sigma = 0)
        
        const double eps = 1e-8;
        
        if (theta < eps) {
            // 小角度情况
            A = 0.5;
            B = 1.0/6.0;
            // 近似计算旋转矩阵，避免数值问题
            R = I + Omega + 0.5 * Omega * Omega;
        } else {
            // 一般情况
            A = (1.0 - cos(theta)) / (theta * theta);
            B = (theta - sin(theta)) / (theta * theta * theta);
            // 使用Rodrigues公式
            R = I + sin(theta)/theta * Omega + 
                (1.0 - cos(theta))/(theta * theta) * Omega * Omega;
        }
        
        // 计算平移部分的转换矩阵
        Eigen::Matrix3d W = A * Omega + B * Omega * Omega + C * I;
        Eigen::Vector3d new_t = W * upsilon;
        
        // 应用SE3增量
        Eigen::Matrix3d R_current = q.normalized().toRotationMatrix();
        Eigen::Matrix3d R_new = R * R_current;
        Eigen::Vector3d t_new = R * t + new_t;
        
        // 转换为四元数
        Eigen::Quaterniond q_new(R_new);
        q_new.normalize();
        
        // 输出新状态
        Eigen::Map<Eigen::Vector3d> t_plus_delta(x_plus_delta);
        Eigen::Map<Eigen::Quaterniond> q_plus_delta(x_plus_delta + 3);
        
        t_plus_delta = t_new;
        q_plus_delta = q_new;
        
        return true;
    }
    
    // 计算deltaR，用于对数映射
    template <typename T>
    static Eigen::Matrix<T, 3, 1> deltaR(const Eigen::Matrix<T, 3, 3>& R) {
        Eigen::Matrix<T, 3, 1> v;
        v(0) = R(2, 1) - R(1, 2);
        v(1) = R(0, 2) - R(2, 0);
        v(2) = R(1, 0) - R(0, 1);
        return v;
    }
    
    // 解析计算Plus操作的雅可比矩阵
    bool PlusJacobian(const double* x, double* jacobian) const override {
        Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> J(jacobian);
        J.setZero();
        
        // 提取当前状态
        Eigen::Map<const Eigen::Vector3d> t(x);
        Eigen::Map<const Eigen::Quaterniond> q(x + 3);
        Eigen::Matrix3d R = q.normalized().toRotationMatrix();
        
        // 设置平移部分的雅可比
        // 旋转增量对平移的影响 (前3行，前3列)
        J.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
        
        // 平移增量对平移的影响 (前3行，后3列)
        // 在SE3情况下，平移增量在当前旋转下的效果
        J.block<3, 3>(0, 3) = R;
        
        // 设置旋转部分的雅可比
        // 旋转增量对旋转的影响 (后4行，前3列)
        // 这部分复杂，涉及四元数的导数
        // 我们使用四元数与旋转向量的关系
        // 对于小的旋转向量delta，四元数更新可近似为:
        // q' = q * [1, delta/2]
        Eigen::Matrix<double, 4, 3> dq_domega;
        dq_domega.setZero();
        // d(qw)/d(omega) = -0.5 * qx, qy, qz
        dq_domega(0, 0) = -0.5 * q.x();
        dq_domega(0, 1) = -0.5 * q.y();
        dq_domega(0, 2) = -0.5 * q.z();
        // d(qx)/d(omega) = 0.5 * qw, -qz, qy
        dq_domega(1, 0) = 0.5 * q.w();
        dq_domega(1, 1) = -0.5 * q.z();
        dq_domega(1, 2) = 0.5 * q.y();
        // d(qy)/d(omega) = qz, qw, -qx
        dq_domega(2, 0) = 0.5 * q.z();
        dq_domega(2, 1) = 0.5 * q.w();
        dq_domega(2, 2) = -0.5 * q.x();
        // d(qz)/d(omega) = -qy, qx, qw
        dq_domega(3, 0) = -0.5 * q.y();
        dq_domega(3, 1) = 0.5 * q.x();
        dq_domega(3, 2) = 0.5 * q.w();
        
        J.block<4, 3>(3, 0) = dq_domega;
        
        // 平移增量对旋转没有影响 (后4行，后3列) 保持为零
        
        return true;
    }
    
    // SE3的对数映射 - 严格匹配g2o::Sim3::log()
    bool Minus(const double* y, const double* x, double* y_minus_x) const override {
        // 提取变换
        Eigen::Map<const Eigen::Vector3d> t_x(x);
        Eigen::Map<const Eigen::Quaterniond> q_x(x + 3);
        Eigen::Map<const Eigen::Vector3d> t_y(y);
        Eigen::Map<const Eigen::Quaterniond> q_y(y + 3);
        
        // 计算相对变换
        Eigen::Matrix3d R_x = q_x.normalized().toRotationMatrix();
        Eigen::Matrix3d R_y = q_y.normalized().toRotationMatrix();
        
        // 计算相对旋转 R_rel = R_y * R_x^T
        Eigen::Matrix3d R_rel = R_y * R_x.transpose();
        
        // 计算相对平移 t_rel = t_y - R_rel * t_x
        Eigen::Vector3d t_rel = t_y - R_rel * t_x;
        
        // 对数映射，严格按照g2o::Sim3::log()
        Eigen::Map<Eigen::Vector6d> result(y_minus_x);
        
        // 计算旋转的对数映射
        double d = 0.5 * (R_rel(0, 0) + R_rel(1, 1) + R_rel(2, 2) - 1.0);
        Eigen::Vector3d omega;
        
        const double eps = 1e-8;
        // 处理接近单位旋转的情况
        if (d > 1.0 - eps) {
            omega = 0.5 * deltaR(R_rel);
        } else {
            double theta = acos(d);
            double scale = theta / (2.0 * sqrt(1.0 - d * d));
            omega = scale * deltaR(R_rel);
        }
        
        // 计算平移的对数映射
        double sigma = 0.0; // 因为尺度固定为1
        double A, B, C;
        C = 1.0; // 因为exp(sigma) = 1.0
        
        double theta = omega.norm();
        Eigen::Matrix3d Omega = skew(omega);
        Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
        
        if (theta < eps) {
            // 小角度情况
            A = 0.5;
            B = 1.0/6.0;
        } else {
            A = (1.0 - cos(theta)) / (theta * theta);
            B = (theta - sin(theta)) / (theta * theta * theta);
        }
        
        // 计算W矩阵的逆
        Eigen::Matrix3d W_inv = I;
        if (theta >= eps) {
            W_inv = I - 0.5 * Omega + 
                (1.0 - theta * cos(theta) / (2.0 * sin(theta))) / 
                (theta * theta) * (Omega * Omega);
        }
        
        Eigen::Vector3d upsilon = W_inv * t_rel;
        
        // 设置结果
        result.segment<3>(0) = omega;
        result.segment<3>(3) = upsilon;
        
        return true;
    }
    
    // 解析计算Minus操作的雅可比矩阵
    bool MinusJacobian(const double* x, double* jacobian) const override {
        Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> J(jacobian);
        J.setZero();
        
        // 提取当前状态
        Eigen::Map<const Eigen::Vector3d> t(x);
        Eigen::Map<const Eigen::Quaterniond> q(x + 3);
        Eigen::Matrix3d R = q.normalized().toRotationMatrix();
        
        // Minus雅可比是Plus雅可比的"逆"
        // 平移部分
        J.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
        J.block<3, 3>(3, 0) = Eigen::Matrix3d::Zero();
        
        // 旋转对omega的影响 - 使用四元数的导数关系
        double qw = q.w(), qx = q.x(), qy = q.y(), qz = q.z();
        
        // 对于小的旋转，从四元数变化到旋转向量的雅可比近似为：
        // 2 * [qw, -qz, qy; qz, qw, -qx; -qy, qx, qw]^T
        Eigen::Matrix<double, 3, 4> domega_dq;
        domega_dq << 2*qw, 2*qx, 2*qy, 2*qz,
                     2*qz, 2*qw, -2*qx, -2*qy,
                     -2*qy, 2*qx, 2*qw, -2*qz;
        
        J.block<3, 4>(0, 3) = domega_dq;
        
        // 旋转对平移的影响 - 在SE3情况下，旋转会影响平移
        J.block<3, 3>(3, 0) = -R.transpose();
        
        // 四元数对平移的影响 - 通过旋转矩阵的变化间接影响
        // 这部分较复杂，实际应用中可能需要数值微分或更详细的解析表达
        // 在此简化处理
        J.block<3, 4>(3, 3) = Eigen::Matrix<double, 3, 4>::Zero();
        
        return true;
    }
};

// SO(3)的对数映射：将旋转矩阵转换为轴角向量 - 严格匹配g2o实现
template<typename T>
Eigen::Matrix<T, 3, 1> LogSO3(const Eigen::Matrix<T, 3, 3>& R) {
    // 计算旋转角度，匹配g2o中的d = 0.5*(R(0,0)+R(1,1)+R(2,2)-1)
    T d = T(0.5) * (R(0, 0) + R(1, 1) + R(2, 2) - T(1.0));
    
    // 限制d在[-1, 1]范围内
    if (d > T(1.0)) d = T(1.0);
    if (d < T(-1.0)) d = T(-1.0);
    
    Eigen::Matrix<T, 3, 1> omega;
    
    // 处理接近单位旋转的情况
    const T eps = T(1e-8);
    if (d > T(1.0) - eps) {
        // 接近单位旋转，使用线性近似
        omega = T(0.5) * SE3Parameterization::deltaR(R);
    } else {
        // 一般情况
        T theta = acos(d);
        T scale = theta / (T(2.0) * sqrt(T(1.0) - d * d));
        omega = scale * SE3Parameterization::deltaR(R);
    }
    
    return omega;
}

// SE3相对姿态约束实现
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
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> t_i(pose_i);
        Eigen::Map<const Eigen::Quaternion<T>> q_i(pose_i + 3);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> t_j(pose_j);
        Eigen::Map<const Eigen::Quaternion<T>> q_j(pose_j + 3);
        
        // 转换为旋转矩阵
        Eigen::Matrix<T, 3, 3> R_i = q_i.normalized().toRotationMatrix();
        Eigen::Matrix<T, 3, 3> R_j = q_j.normalized().toRotationMatrix();
        
        // 预期的相对变换
        Eigen::Matrix<T, 3, 3> R_ij = relative_rotation_.cast<T>();
        Eigen::Matrix<T, 3, 1> t_ij = relative_translation_.cast<T>();
        
        // 按照ORB-SLAM3的公式计算error
        // 旋转error: LogSO3(R_i * R_j^T * R_ij^T)
        Eigen::Matrix<T, 3, 3> R_error_mat = R_i * R_j.transpose() * R_ij.transpose();
        Eigen::Matrix<T, 3, 1> rotation_error = LogSO3(R_error_mat);
        
        // 平移error: R_i * (-R_j^T * t_j) + t_i - t_ij
        Eigen::Matrix<T, 3, 1> t_ji = -R_j.transpose() * t_j;
        Eigen::Matrix<T, 3, 1> translation_error = R_i * t_ji + t_i - t_ij;
        
        // 组合residuals
        Eigen::Map<Eigen::Matrix<T, 6, 1>> residual_map(residuals);
        residual_map.segment<3>(0) = rotation_error;
        residual_map.segment<3>(3) = translation_error;
        
        // 应用信息矩阵
        residual_map = sqrt_information_.cast<T>() * residual_map;
        
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

// ... 其余代码保持不变 ...


// 回环边约束 - 对应g2o的EdgeSim3（用于回环边）
class SE3LoopEdgeCost {
public:
    SE3LoopEdgeCost(const Eigen::Matrix4d& relative_transform, const Eigen::Matrix<double, 6, 6>& information)
        : relative_rotation_(relative_transform.block<3, 3>(0, 0)),
          relative_translation_(relative_transform.block<3, 1>(0, 3)),
          sqrt_information_(information.llt().matrixL()) {
    }
    
    template <typename T>
    bool operator()(const T* const pose_i, const T* const pose_l, T* residuals) const {
        // pose_i: 当前关键帧, pose_l: 回环关键帧
        // 格式: [tx, ty, tz, qx, qy, qz, qw]
        
        // 提取姿态
        Eigen::Matrix<T, 3, 1> t_i(pose_i[0], pose_i[1], pose_i[2]);
        Eigen::Matrix<T, 3, 1> t_l(pose_l[0], pose_l[1], pose_l[2]);
        Eigen::Quaternion<T> q_i(pose_i[6], pose_i[3], pose_i[4], pose_i[5]);
        Eigen::Quaternion<T> q_l(pose_l[6], pose_l[3], pose_l[4], pose_l[5]);
        
        // 转换为旋转矩阵
        Eigen::Matrix<T, 3, 3> R_i = q_i.toRotationMatrix();
        Eigen::Matrix<T, 3, 3> R_l = q_l.toRotationMatrix();
        
        // 计算相对变换 T_li = T_l * T_i^{-1}
        Eigen::Matrix<T, 3, 3> R_li = R_l * R_i.transpose();
        Eigen::Matrix<T, 3, 1> t_li = R_l * (R_i.transpose() * (-t_i)) + t_l;
        
        // 预期的相对变换
        Eigen::Matrix<T, 3, 3> R_expected = relative_rotation_.cast<T>();
        Eigen::Matrix<T, 3, 1> t_expected = relative_translation_.cast<T>();
        
        // 计算旋转误差
        Eigen::Matrix<T, 3, 3> R_error_mat = R_expected.transpose() * R_li;
        Eigen::Matrix<T, 3, 1> rotation_error = LogSO3(R_error_mat);
        
        // 计算平移误差
        Eigen::Matrix<T, 3, 1> translation_error = t_li - t_expected;
        
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
        return new ceres::AutoDiffCostFunction<SE3LoopEdgeCost, 6, 7, 7>(
            new SE3LoopEdgeCost(relative_transform, information));
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

    void AddNormalEdgeConstraints() {
        std::cout << "\n开始添加正常边约束（生成树 + 回环边 + 共视）..." << std::endl;
        
        const int minFeat = 100;
        
        // 信息矩阵
        Eigen::Matrix<double, 6, 6> information = Eigen::Matrix<double, 6, 6>::Identity();
        information(0, 0) = 1e3;
        information(1, 1) = 1e3;
        information(2, 2) = 1e3;
        
        int spanning_tree_edges = 0;
        int loop_edges_count = 0;
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
            
            // 2. Loop edges - 复用T_iw (新添加的部分)
            if (data_.loop_edges.find(nIDi) != data_.loop_edges.end()) {
                for (int nIDl : data_.loop_edges[nIDi]) {
                    // 只处理 ID 较小的边，避免重复
                    if (nIDl >= nIDi) continue;
                    
                    // 检查回环关键帧是否存在且不是坏帧
                    if (data_.keyframes.find(nIDl) == data_.keyframes.end() ||
                        data_.keyframes[nIDl]->is_bad) continue;
                    
                    auto kf_l = data_.keyframes[nIDl];
                    
                    // 计算回环关键帧的变换
                    Eigen::Matrix4d T_wl = Eigen::Matrix4d::Identity();
                    if (data_.non_corrected_poses.find(nIDl) != data_.non_corrected_poses.end()) {
                        const auto& non_corrected_l = data_.non_corrected_poses[nIDl];
                        T_wl.block<3, 3>(0, 0) = non_corrected_l.quaternion.toRotationMatrix();
                        T_wl.block<3, 1>(0, 3) = non_corrected_l.translation;
                    } else {
                        T_wl.block<3, 3>(0, 0) = kf_l->quaternion.toRotationMatrix();
                        T_wl.block<3, 1>(0, 3) = kf_l->translation;
                    }
                    
                    // T_li = T_wl * T_iw (对应ORB-SLAM3的 Sli = Slw * Swi)
                    Eigen::Matrix4d T_li = T_wl * T_iw;
                    
                    ceres::CostFunction* cost_function = SE3LoopEdgeCost::Create(T_li, information);
                    problem_->AddResidualBlock(cost_function, nullptr,
                                             kf_i->se3_state.data(),
                                             kf_l->se3_state.data());
                    loop_edges_count++;
                    
                    // 调试输出
                    if (loop_edges_count <= 5) {
                        std::cout << "添加回环边约束: " << nIDi << " <-> " << nIDl << std::endl;
                    }
                }
            }
            
            // 3. Covisibility graph edges - 复用T_iw
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
                    
                    // 跳过回环边（避免重复约束）
                    if (data_.loop_edges.find(nIDi) != data_.loop_edges.end() &&
                        data_.loop_edges[nIDi].count(nIDn) > 0) continue;
                    
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
        std::cout << "添加回环边约束: " << loop_edges_count << " 个" << std::endl;
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
    
    // 添加回环约束
    optimizer.AddLoopConstraints();
    

    // 添加正常边约束（生成树 + 回环边 + 共视）
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
