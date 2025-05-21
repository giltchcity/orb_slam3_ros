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
// SE3 李代数参数化 - 基于g2o的Sim3实现，但固定尺度为1
class SE3Parameterization : public ceres::Manifold {
public:
    ~SE3Parameterization() {}
    
    // 7维环境空间：[tx, ty, tz, qx, qy, qz, qw] 
    int AmbientSize() const override { return 7; }
    
    // 6维切空间：[rho(3), phi(3)] - 我们实际上是使用Sim3参数化但忽略尺度
    int TangentSize() const override { return 6; }
    
    // 基于g2o::Sim3的指数映射实现
    bool Plus(const double* x, const double* delta, double* x_plus_delta) const override {
        // 提取当前状态
        Eigen::Vector3d t_current(x[0], x[1], x[2]);
        Eigen::Quaterniond q_current(x[6], x[3], x[4], x[5]);
        q_current.normalize();
        
        // 提取李代数增量，按照g2o::Sim3格式：[rotation(3), translation(3), scale(1)]
        // 但我们忽略尺度增量（设为0）
        Eigen::Vector3d omega(delta[0], delta[1], delta[2]);    // 旋转部分
        Eigen::Vector3d upsilon(delta[3], delta[4], delta[5]);  // 平移部分
        double sigma = 0.0;                                     // 固定尺度，增量为0
        
        // ---------- 以下计算逻辑完全复制自g2o::Sim3构造函数 ----------
        
        double theta = omega.norm();
        double s = std::exp(sigma);  // s始终为1（因为sigma=0）
        
        // 计算旋转矩阵
        Eigen::Matrix3d Omega = skew(omega);
        Eigen::Matrix3d Omega2 = Omega * Omega;
        Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
        Eigen::Matrix3d R;
        
        // A, B, C系数，照抄g2o的计算方式
        double eps = 1e-5;
        double A, B, C;
        
        // 处理不同情况（小角度、大角度、尺度接近1等）
        if (fabs(sigma) < eps) {
            // 尺度变化接近于0（我们的情况）
            C = 1;
            if (theta < eps) {
                // 小角度情况
                A = 0.5;
                B = 1.0/6.0;
                R = I + Omega + Omega2 * 0.5;
            } else {
                // 较大角度情况
                double theta2 = theta * theta;
                A = (1.0 - std::cos(theta)) / theta2;
                B = (theta - std::sin(theta)) / (theta2 * theta);
                R = I + std::sin(theta) / theta * Omega + 
                    (1.0 - std::cos(theta)) / (theta * theta) * Omega2;
            }
        } else {
            // 尺度变化明显（不是我们的情况，但保留g2o的完整逻辑）
            C = (s - 1.0) / sigma;
            if (theta < eps) {
                // 尺度变化明显但角度很小
                double sigma2 = sigma * sigma;
                A = ((sigma - 1.0) * s + 1.0) / sigma2;
                B = ((0.5 * sigma2 - sigma + 1.0) * s - 1.0) / (sigma2 * sigma);
                R = I + Omega + Omega2 * 0.5;
            } else {
                // 尺度变化明显且角度明显
                R = I + std::sin(theta) / theta * Omega + 
                    (1.0 - std::cos(theta)) / (theta * theta) * Omega2;
                double a = s * std::sin(theta);
                double b = s * std::cos(theta);
                double theta2 = theta * theta;
                double sigma2 = sigma * sigma;
                double c = theta2 + sigma2;
                A = (a * sigma + (1.0 - b) * theta) / (theta * c);
                B = (C - ((b - 1.0) * sigma + a * theta) / (c)) / (theta2);
            }
        }
        
        // 计算平移增量
        Eigen::Matrix3d W = A * Omega + B * Omega2 + C * I;
        Eigen::Vector3d t_delta = W * upsilon;
        
        // ---------- 结束g2o::Sim3构造函数逻辑复制 ----------
        
        // 应用增量：注意这里使用g2o::Sim3的乘法公式
        Eigen::Matrix3d R_current = q_current.toRotationMatrix();
        Eigen::Matrix3d R_new = R * R_current;          // 旋转更新
        Eigen::Vector3d t_new = s * (R * t_current) + t_delta;  // 平移更新，s=1
        
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
    
    // 辅助函数：计算反对称矩阵
    static Eigen::Matrix3d skew(const Eigen::Vector3d& v) {
        Eigen::Matrix3d m;
        m << 0, -v(2), v(1),
             v(2), 0, -v(0),
             -v(1), v(0), 0;
        return m;
    }
    
    // Sim3的对数映射，用于计算两个位姿之间的差
    bool Minus(const double* y, const double* x, double* y_minus_x) const override {
        // 提取两个位姿
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
        
        // 计算旋转的对数映射（SO3部分）
        Eigen::Vector3d omega;
        double trace = R_rel.trace();
        
        // 处理不同角度情况
        if (trace > 3.0 - 1e-6) {
            // 接近单位矩阵的情况
            omega = 0.5 * Eigen::Vector3d(
                R_rel(2,1) - R_rel(1,2),
                R_rel(0,2) - R_rel(2,0),
                R_rel(1,0) - R_rel(0,1)
            );
        } else {
            double d = 0.5 * (trace - 1.0);
            // 限制d在[-1,1]范围内
            d = d > 1.0 ? 1.0 : (d < -1.0 ? -1.0 : d);
            double angle = std::acos(d);
            // 计算旋转轴
            Eigen::Vector3d axis;
            axis << R_rel(2,1) - R_rel(1,2), 
                    R_rel(0,2) - R_rel(2,0), 
                    R_rel(1,0) - R_rel(0,1);
            
            if (axis.norm() < 1e-10) {
                // 处理接近180度的特殊情况
                // 根据g2o实现查找最大的对角元素
                int max_idx = 0;
                for (int i = 1; i < 3; ++i) {
                    if ((R_rel(i,i) > R_rel(max_idx,max_idx)) && (R_rel(i,i) > 0)) {
                        max_idx = i;
                    }
                }
                
                Eigen::Vector3d col;
                col = R_rel.col(max_idx);
                col.normalize();
                
                omega = angle * col;
                if (omega.norm() < 1e-6) {
                    omega.setZero();
                }
            } else {
                // 正常情况
                axis.normalize();
                omega = angle * axis;
            }
        }
        
        // 计算平移的对数映射（根据g2o的Sim3::log实现）
        double angle = omega.norm();
        double scale = 1.0; // 固定尺度为1
        
        // 计算系数A、B、C（遵照g2o实现）
        double A, B, C;
        double eps = 1e-6;
        
        C = 1.0; // 因为scale=1
        Eigen::Matrix3d Omega = skew(omega);
        Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
        Eigen::Matrix3d W;
        
        if (angle < eps) {
            // 小角度情况
            A = 0.5;
            B = 1.0/6.0;
            W = I + 0.5 * Omega + (1.0/6.0) * Omega * Omega;
        } else {
            // 正常角度情况
            double s = sin(angle);
            double c = cos(angle);
            A = (1.0 - c) / (angle * angle);
            B = (angle - s) / (angle * angle * angle);
            W = I + A * Omega + B * Omega * Omega;
        }
        
        // 计算平移部分的李代数
        Eigen::Vector3d upsilon = W.inverse() * t_rel;
        
        // 设置输出
        y_minus_x[0] = omega(0);
        y_minus_x[1] = omega(1);
        y_minus_x[2] = omega(2);
        y_minus_x[3] = upsilon(0);
        y_minus_x[4] = upsilon(1);
        y_minus_x[5] = upsilon(2);
        
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

    // 添加新的成员变量来存储顶点初始位姿
    std::map<int, Eigen::Matrix4d> vertex_initial_poses_Tcw; // 世界到相机的位姿
    std::map<int, Eigen::Matrix4d> vertex_initial_poses_Twc; // 相机到世界的位姿

};

class ORBSlamLoopOptimizer {
private:
    OptimizationData data_;
    std::unique_ptr<ceres::Problem> problem_;

    // 添加这一行 - 确保两个函数间共享
    std::set<std::pair<long unsigned int, long unsigned int>> sInsertedEdges;
    
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
        
        // 清空之前的位姿存储
        data_.vertex_initial_poses_Tcw.clear();
        data_.vertex_initial_poses_Twc.clear();
        
        // 为每个关键帧添加顶点
        for (auto& kf_pair : data_.keyframes) {
            auto& kf = kf_pair.second;
            
            // 跳过坏的关键帧
            if (kf->is_bad) continue;
            
            // 定义Tcw矩阵 (相机在世界坐标系的位姿)
            Eigen::Matrix4d Tcw = Eigen::Matrix4d::Identity();
            bool useCorrected = false;
            
            // 检查是否有修正的姿态
            if (data_.corrected_poses.find(kf->id) != data_.corrected_poses.end()) {
                // 使用修正的SE3
                const auto& corrected = data_.corrected_poses[kf->id];
                for (int i = 0; i < 7; ++i) {
                    kf->se3_state[i] = corrected.se3_state[i];
                }
                kf->UpdateFromState();
                
                // 保存修正后的姿态作为初始值
                Tcw.block<3, 3>(0, 0) = corrected.quaternion.toRotationMatrix();
                Tcw.block<3, 1>(0, 3) = corrected.translation;
                useCorrected = true;
            } else {
                // 使用当前关键帧的状态
                Tcw.block<3, 3>(0, 0) = kf->quaternion.toRotationMatrix();
                Tcw.block<3, 1>(0, 3) = kf->translation;
            }
            
            // 计算Twc (世界在相机坐标系的位姿)
            Eigen::Matrix4d Twc = Tcw.inverse();
            
            // 保存这些位姿
            data_.vertex_initial_poses_Tcw[kf->id] = Tcw;
            data_.vertex_initial_poses_Twc[kf->id] = Twc;
            
            // 添加参数块到优化问题
            problem_->AddParameterBlock(kf->se3_state.data(), 7);
            
            // 设置SE3参数化（流形）- 类似 ORB-SLAM3 的 VertexSim3Expmap
            problem_->SetManifold(kf->se3_state.data(), new SE3Parameterization());
            
            // 固定初始关键帧
            if (kf->id == data_.init_kf_id || kf->is_fixed) {
                problem_->SetParameterBlockConstant(kf->se3_state.data());
                std::cout << "固定关键帧 " << kf->id << std::endl;
            }
            
            if (kf->id % 100 == 0 || kf->id == data_.init_kf_id || kf->id == data_.current_kf_id || kf->id == data_.loop_kf_id) {
                std::cout << "  添加关键帧顶点 KF" << kf->id << ": "
                         << (useCorrected ? "使用CorrectedPose" : "使用原始位姿") 
                         << " 位置: [" << Tcw.block<3, 1>(0, 3).transpose() << "]" 
                         << std::endl;
            }
        }
        
        std::cout << "添加了 " << data_.keyframes.size() << " 个关键帧顶点" << std::endl;
        std::cout << "保存了 " << data_.vertex_initial_poses_Tcw.size() << " 个初始位姿" << std::endl;
        std::cout << "优化问题设置完成，参数块数量: " << problem_->NumParameterBlocks() << std::endl;
    }


    // 添加回环约束
    // 添加回环约束 - 精确匹配ORB-SLAM3的Set Loop edges部分
    // 修正后的回环约束添加函数 - 精确匹配ORB-SLAM3
    // 修正后的回环约束添加函数 - 解决权重查询问题
    void AddLoopConstraints() {
        std::cout << "\n=== Adding Loop Edges ===" << std::endl;
        
        const int minFeat = 100; // 最小特征点数阈值
        // std::set<std::pair<long unsigned int, long unsigned int>> sInsertedEdges;
        Eigen::Matrix<double, 6, 6> matLambda = Eigen::Matrix<double, 6, 6>::Identity();
        
        int count_loop = 0;
        int attempted_loop = 0;
        
        std::cout << "Minimum features for edge connection: " << minFeat << std::endl;
        std::cout << "\n=== Information Matrix ===" << std::endl;
        std::cout << matLambda << std::endl;
        
        // 检查是否有保存的顶点位姿
        if (data_.vertex_initial_poses_Tcw.empty()) {
            std::cerr << "错误：没有找到保存的顶点初始位姿！" << std::endl;
            return;
        }
        
        // 遍历所有回环连接
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
            
            // 获取关键帧i的变换矩阵
            Eigen::Matrix4d Siw = Eigen::Matrix4d::Identity();
            bool useCorrectedSim3_i = false;
            
            // 优先使用修正位姿
            if (data_.corrected_poses.find(nIDi) != data_.corrected_poses.end()) {
                // 使用修正姿态
                const auto& corrected = data_.corrected_poses[nIDi];
                Siw.block<3, 3>(0, 0) = corrected.quaternion.toRotationMatrix();
                Siw.block<3, 1>(0, 3) = corrected.translation;
                useCorrectedSim3_i = true;
            } 
            else if (data_.vertex_initial_poses_Tcw.find(nIDi) != data_.vertex_initial_poses_Tcw.end()) {
                // 使用保存的初始顶点位姿
                Siw = data_.vertex_initial_poses_Tcw[nIDi];
            }
            else {
                // 使用当前位姿（不应该到达这里）
                Siw.block<3, 3>(0, 0) = pKF->quaternion.toRotationMatrix();
                Siw.block<3, 1>(0, 3) = pKF->translation;
                std::cerr << "警告：KF" << nIDi << "没有找到保存的初始位姿" << std::endl;
            }
            
            // 计算逆变换
            Eigen::Matrix4d Swi = Siw.inverse();
            
            // 遍历连接的关键帧
            for (int nIDj : spConnections) {
                attempted_loop++;
                
                // 检查关键帧j是否有效
                if (data_.keyframes.find(nIDj) == data_.keyframes.end() || 
                    data_.keyframes[nIDj]->is_bad) {
                    continue;
                }
                
                auto pKFj = data_.keyframes[nIDj];
                
                // 权重检查
                bool skipEdge = false;
                int weight = 0;
                
                // 检查是否是主要回环边
                bool isMainLoopEdge = (nIDi == data_.current_kf_id && nIDj == data_.loop_kf_id);
                
                if (!isMainLoopEdge) {
                    // 双向查找权重
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
                    
                    if (weight < minFeat) {
                        skipEdge = true;
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
                
                // 获取关键帧j的变换矩阵
                Eigen::Matrix4d Sjw = Eigen::Matrix4d::Identity();
                bool useCorrectedSim3_j = false;
                
                if (data_.corrected_poses.find(nIDj) != data_.corrected_poses.end()) {
                    // 使用修正姿态
                    const auto& corrected = data_.corrected_poses[nIDj];
                    Sjw.block<3, 3>(0, 0) = corrected.quaternion.toRotationMatrix();
                    Sjw.block<3, 1>(0, 3) = corrected.translation;
                    useCorrectedSim3_j = true;
                } 
                else if (data_.vertex_initial_poses_Tcw.find(nIDj) != data_.vertex_initial_poses_Tcw.end()) {
                    // 使用保存的初始顶点位姿
                    Sjw = data_.vertex_initial_poses_Tcw[nIDj];
                }
                else {
                    // 使用当前位姿（不应该到达这里）
                    Sjw.block<3, 3>(0, 0) = pKFj->quaternion.toRotationMatrix();
                    Sjw.block<3, 1>(0, 3) = pKFj->translation;
                    std::cerr << "警告：KF" << nIDj << "没有找到保存的初始位姿" << std::endl;
                }
                
                // 计算相对变换
                Eigen::Matrix4d Sji = Sjw * Swi;
                
                // 创建优化边
                ceres::CostFunction* cost_function = SE3LoopConstraintCost::Create(Sji, matLambda);
                
                // 添加残差块
                problem_->AddResidualBlock(cost_function,
                                         nullptr,
                                         pKF->se3_state.data(),   // 关键帧i
                                         pKFj->se3_state.data()); // 关键帧j
                
                count_loop++;
                
                // 记录边
                sInsertedEdges.insert(std::make_pair(std::min((long unsigned int)nIDi, (long unsigned int)nIDj), 
                                                   std::max((long unsigned int)nIDi, (long unsigned int)nIDj)));
                
                // 输出详细信息
                Eigen::Vector3d translation = Sji.block<3, 1>(0, 3);
                std::cout << "  Added Loop Edge: KF" << nIDi << " -> KF" << nIDj 
                          << " | Weight: " << weight
                          << " | Translation: [" << translation.transpose() << "]" 
                          << " | Scale: " << 1.0
                          << std::endl;
            }
        }
        
        std::cout << "\nSuccessful Loop Edges: " << count_loop << "/" << attempted_loop << std::endl;
        std::cout << "Unique Edge Pairs: " << sInsertedEdges.size() << std::endl;
    }




    void AddNormalEdgeConstraints() {
        std::cout << "\n=== Adding Normal Edges ===" << std::endl;
        
        // 信息矩阵（与回环约束相同）
        Eigen::Matrix<double, 6, 6> information = Eigen::Matrix<double, 6, 6>::Identity();
        std::cout << "\n=== Information Matrix ===" << std::endl;
        std::cout << information << std::endl;
        
        // 用于记录已添加的边（避免重复）
        // std::set<std::pair<long unsigned int, long unsigned int>> sInsertedEdges;
        
        int normal_edges_added = 0;
        int attempted_normal = 0;
        int validKFCount = 0;  // 有效关键帧计数
        
        // 检查是否有保存的顶点位姿
        if (data_.vertex_initial_poses_Tcw.empty()) {
            std::cerr << "错误：没有找到保存的顶点初始位姿！" << std::endl;
            return;
        }
        
        // 第一阶段：添加生成树边（parent-child edges）
        std::cout << "\n=== Adding Spanning Tree Edges ===" << std::endl;
        
        // 遍历所有关键帧
        for (const auto& kf_pair : data_.keyframes) {
            int kf_id = kf_pair.first;
            auto& kf = kf_pair.second;
            
            // 跳过坏的关键帧
            if (kf->is_bad) continue;
            
            validKFCount++;  // 增加有效关键帧计数
            
            // 每10个关键帧打印一次详细信息
            bool printDetailedInfo = (validKFCount % 10 == 0);
            
            if(printDetailedInfo) {
                std::cout << "Processing KF " << validKFCount << "/" << data_.keyframes.size() 
                          << " (KF" << kf_id << ")";
                
                // 显示父关键帧信息
                if(kf->parent_id >= 0) {
                    std::cout << " | Parent: KF" << kf->parent_id;
                } else {
                    std::cout << " | Parent: None";
                }
                
                std::cout << std::endl;
            }
            
            // 获取父关键帧ID
            int parent_id = kf->parent_id;
            
            // 如果没有父关键帧，或者父关键帧就是自己，跳过
            if (parent_id < 0 || parent_id == kf_id || 
                data_.keyframes.find(parent_id) == data_.keyframes.end() || 
                data_.keyframes[parent_id]->is_bad) {
                if(printDetailedInfo) {
                    std::cout << "  Skipping KF" << kf_id << ": ";
                    if(parent_id < 0) std::cout << "No parent";
                    else if(parent_id == kf_id) std::cout << "Self-loop";
                    else if(data_.keyframes.find(parent_id) == data_.keyframes.end()) std::cout << "Parent not found";
                    else std::cout << "Parent is bad";
                    std::cout << std::endl;
                }
                continue;
            }
            
            // 获取当前关键帧的逆变换 Swi
            Eigen::Matrix4d T_iw = Eigen::Matrix4d::Identity();
            bool usingNonCorrected_i = false;
            
            // 使用NonCorrectedSim3如果存在
            if (data_.non_corrected_poses.find(kf_id) != data_.non_corrected_poses.end()) {
                const auto& non_corrected_i = data_.non_corrected_poses[kf_id];
                Eigen::Matrix4d T_i = Eigen::Matrix4d::Identity();
                T_i.block<3, 3>(0, 0) = non_corrected_i.quaternion.toRotationMatrix();
                T_i.block<3, 1>(0, 3) = non_corrected_i.translation;
                T_iw = T_i.inverse();
                usingNonCorrected_i = true;
            } 
            else if (data_.vertex_initial_poses_Twc.find(kf_id) != data_.vertex_initial_poses_Twc.end()) {
                // 使用保存的初始顶点位姿
                T_iw = data_.vertex_initial_poses_Twc[kf_id];
            }
            else {
                // 应该不会到达这里，因为所有有效顶点都应该有保存的位姿
                std::cerr << "警告：KF" << kf_id << "没有找到保存的初始位姿，使用当前状态" << std::endl;
                Eigen::Matrix4d T_i = Eigen::Matrix4d::Identity();
                T_i.block<3, 3>(0, 0) = kf->quaternion.toRotationMatrix();
                T_i.block<3, 1>(0, 3) = kf->translation;
                T_iw = T_i.inverse();
            }
            
            // 获取父关键帧的变换 Sjw
            Eigen::Matrix4d T_j = Eigen::Matrix4d::Identity();
            bool usingNonCorrected_j = false;
            
            if (data_.non_corrected_poses.find(parent_id) != data_.non_corrected_poses.end()) {
                const auto& non_corrected_parent = data_.non_corrected_poses[parent_id];
                T_j.block<3, 3>(0, 0) = non_corrected_parent.quaternion.toRotationMatrix();
                T_j.block<3, 1>(0, 3) = non_corrected_parent.translation;
                usingNonCorrected_j = true;
            }
            else if (data_.vertex_initial_poses_Tcw.find(parent_id) != data_.vertex_initial_poses_Tcw.end()) {
                // 使用保存的初始顶点位姿
                T_j = data_.vertex_initial_poses_Tcw[parent_id];
            }
            else {
                // 应该不会到达这里
                std::cerr << "警告：父KF" << parent_id << "没有找到保存的初始位姿，使用当前状态" << std::endl;
                T_j.block<3, 3>(0, 0) = data_.keyframes[parent_id]->quaternion.toRotationMatrix();
                T_j.block<3, 1>(0, 3) = data_.keyframes[parent_id]->translation;
            }
            
            // 计算相对变换 Sji = Sjw * Swi
            Eigen::Matrix4d T_ji = T_j * T_iw;
            Eigen::Vector3d translation = T_ji.block<3, 1>(0, 3);
            
            attempted_normal++;
            
            // 添加约束
            ceres::CostFunction* cost_function = SE3LoopConstraintCost::Create(T_ji, information);
            problem_->AddResidualBlock(cost_function, nullptr, 
                                      data_.keyframes[kf_id]->se3_state.data(),      // 子节点
                                      data_.keyframes[parent_id]->se3_state.data()); // 父节点
            
            normal_edges_added++;
            
            // 记录已添加的边对，避免重复
            // sInsertedEdges.insert(std::make_pair(
            //     std::min((long unsigned int)kf_id, (long unsigned int)parent_id),
            //     std::max((long unsigned int)kf_id, (long unsigned int)parent_id)
            // ));
            
            // 打印详细信息
            if(printDetailedInfo) {
                std::cout << "  Added Spanning Tree Edge: KF" << kf_id << " -> KF" << parent_id 
                          << " | Translation: [" << translation.transpose() << "]" 
                          << " | Scale: 1" << std::endl;
                
                // 额外添加的详细信息
                std::string source_i = usingNonCorrected_i ? "NonCorrectedSim3" : "VertexInitialPose";
                std::string source_j = usingNonCorrected_j ? "NonCorrectedSim3" : "VertexInitialPose";
                
                std::cout << "    KF" << kf_id << " using: " << source_i
                          << " | KF" << parent_id << " using: " << source_j 
                          << std::endl;
                
                // 打印相机方向和旋转信息
                Eigen::Vector3d z_vec(0, 0, 1);
                Eigen::Vector3d z_dir_i = T_iw.block<3, 3>(0, 0) * z_vec;
                Eigen::Vector3d z_dir_j = T_j.inverse().block<3, 3>(0, 0) * z_vec;
                
                // 计算相对旋转角度
                double angle = acos(z_dir_i.dot(z_dir_j) / (z_dir_i.norm() * z_dir_j.norm())) * 180.0 / M_PI;
                
                std::cout << "    Camera Dir KF" << kf_id << ": [" << z_dir_i.transpose() << "]" << std::endl;
                std::cout << "    Camera Dir KF" << parent_id << ": [" << z_dir_j.transpose() << "]" << std::endl;
                std::cout << "    Rotation Angle: " << angle << " degrees" << std::endl;
                std::cout << "    Edge Info Trace: " << information.trace() << std::endl;
            }
        }
        
        std::cout << "\nSpanning Tree Edges Added: " << normal_edges_added << "/" << attempted_normal << std::endl;
        
        // 第二阶段：添加共视图边
         std::cout << "\n=== Adding Covisibility Graph Edges ===" << std::endl;
        
        // 添加KF 0特殊关系检查
        std::cout << "检查KF 0的特殊关系:" << std::endl;
        // 检查KF 0与KF 3的关系
        if (data_.keyframes.find(0) != data_.keyframes.end() && 
            data_.keyframes.find(3) != data_.keyframes.end()) {
            bool is_parent = data_.keyframes[3]->parent_id == 0;
            bool is_child = false;
            if (data_.spanning_tree.find(0) != data_.spanning_tree.end()) {
                is_child = std::find(data_.spanning_tree[0].begin(), data_.spanning_tree[0].end(), 3) != data_.spanning_tree[0].end();
            }
            bool is_loop_edge = false;
            if (data_.loop_edges.find(0) != data_.loop_edges.end()) {
                is_loop_edge = data_.loop_edges[0].find(3) != data_.loop_edges[0].end();
            }
            int weight = 0;
            if (data_.covisibility.find(0) != data_.covisibility.end() && 
                data_.covisibility[0].find(3) != data_.covisibility[0].end()) {
                weight = data_.covisibility[0][3];
            }
            std::cout << "KF 0 - KF 3: 父子关系=" << (is_parent || is_child ? "是" : "否")
                      << ", 回环边=" << (is_loop_edge ? "是" : "否")
                      << ", 权重=" << weight << std::endl;
        }
        // 检查KF 0与KF 461的关系
        if (data_.keyframes.find(0) != data_.keyframes.end() && 
            data_.keyframes.find(461) != data_.keyframes.end()) {
            bool is_parent = data_.keyframes[461]->parent_id == 0;
            bool is_child = false;
            if (data_.spanning_tree.find(0) != data_.spanning_tree.end()) {
                is_child = std::find(data_.spanning_tree[0].begin(), data_.spanning_tree[0].end(), 461) != data_.spanning_tree[0].end();
            }
            bool is_loop_edge = false;
            if (data_.loop_edges.find(0) != data_.loop_edges.end()) {
                is_loop_edge = data_.loop_edges[0].find(461) != data_.loop_edges[0].end();
            }
            int weight = 0;
            if (data_.covisibility.find(0) != data_.covisibility.end() && 
                data_.covisibility[0].find(461) != data_.covisibility[0].end()) {
                weight = data_.covisibility[0][461];
            }
            std::cout << "KF 0 - KF 461: 父子关系=" << (is_parent || is_child ? "是" : "否")
                      << ", 回环边=" << (is_loop_edge ? "是" : "否")
                      << ", 权重=" << weight << std::endl;
        }



        
        const int minFeat = 100;  // 最小特征点阈值（与ORB-SLAM3一致）
        int count_covis = 0;      // 添加到图中的共视边计数
        int count_all_valid_covis = 0;  // 所有合格的共视关系（包括未添加的）
        std::vector<int> covis_per_kf(data_.max_kf_id + 1, 0);  // 每个关键帧的共视边数量
        std::map<int, std::vector<std::pair<int, int>>> covis_weights;  // 存储共视关系的权重
        
        std::cout << "Minimum features for covisibility edge: " << minFeat << std::endl;
        
        // 遍历所有关键帧
        for (const auto& kf_pair : data_.keyframes) {
            int nIDi = kf_pair.first;  // 当前关键帧ID
            auto& pKF = kf_pair.second;
            
            // 跳过坏的关键帧
            if (pKF->is_bad) continue;
            
            // 获取当前关键帧的逆变换 Swi
            Eigen::Matrix4d Swi = Eigen::Matrix4d::Identity();
            
            if (data_.non_corrected_poses.find(nIDi) != data_.non_corrected_poses.end()) {
                const auto& non_corrected = data_.non_corrected_poses[nIDi];
                Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
                T.block<3, 3>(0, 0) = non_corrected.quaternion.toRotationMatrix();
                T.block<3, 1>(0, 3) = non_corrected.translation;
                Swi = T.inverse();
            } 
            else if (data_.vertex_initial_poses_Twc.find(nIDi) != data_.vertex_initial_poses_Twc.end()) {
                Swi = data_.vertex_initial_poses_Twc[nIDi];
            }
            else {
                Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
                T.block<3, 3>(0, 0) = pKF->quaternion.toRotationMatrix();
                T.block<3, 1>(0, 3) = pKF->translation;
                Swi = T.inverse();
            }
            
            // 获取父关键帧ID
            int pParentKF = pKF->parent_id;
            
            // 获取与当前关键帧有共视关系且权重>=minFeat的关键帧
            // 这里模拟ORB-SLAM3的GetCovisiblesByWeight函数
            std::vector<std::pair<int, int>> orderedConnections;
            if (data_.covisibility.find(nIDi) != data_.covisibility.end()) {
                // 从共视图中获取所有连接的关键帧
                for (const auto& covis_pair : data_.covisibility[nIDi]) {
                    int connected_id = covis_pair.first;
                    int weight = covis_pair.second;
                    
                    // 只保留权重大于等于minFeat的关键帧
                    if (weight >= minFeat) {
                        orderedConnections.push_back(std::make_pair(connected_id, weight));
                    }
                }
                
                // 按权重降序排序（重要！与ORB-SLAM3保持一致）
                std::sort(orderedConnections.begin(), orderedConnections.end(), 
                    [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
                        return a.second > b.second; // 按权重降序
                    });
            }
            
            // 提取排序后的关键帧ID
            std::vector<int> vpConnectedKFs;
            for (const auto& pair : orderedConnections) {
                vpConnectedKFs.push_back(pair.first);
            }
            
            // 遍历共视关键帧
            for (int nIDj : vpConnectedKFs) {
                // 检查连接的关键帧是否有效
                if (data_.keyframes.find(nIDj) == data_.keyframes.end() || 
                    data_.keyframes[nIDj]->is_bad)
                    continue;
                
                auto pKFn = data_.keyframes[nIDj];
                

                
                // 检查是否满足条件：
                // 1. 不是父关键帧
                // 2. 不是当前关键帧的子关键帧（这需要从spanning_tree判断）
                bool isChild = false;
                if (data_.spanning_tree.find(nIDi) != data_.spanning_tree.end()) {
                    const auto& children = data_.spanning_tree[nIDi];
                    isChild = std::find(children.begin(), children.end(), nIDj) != children.end();
                }
                
                // 检查是否是回环边（从loop_edges判断）
                bool isLoopEdge = false;
                if (data_.loop_edges.find(nIDi) != data_.loop_edges.end()) {
                    isLoopEdge = data_.loop_edges[nIDi].find(nIDj) != data_.loop_edges[nIDi].end();
                }

                // 在这里添加KF 0的调试代码
                if (nIDi == 0 || nIDj == 0) {
                    // 获取共视权重
                    int weight = 0;
                    if (data_.covisibility.find(nIDi) != data_.covisibility.end() &&
                        data_.covisibility[nIDi].find(nIDj) != data_.covisibility[nIDi].end()) {
                        weight = data_.covisibility[nIDi][nIDj];
                    }
                    
                    std::cout << "DEBUG KF0: ";
                    std::cout << "处理 KF" << nIDi << " - KF" << nIDj 
                            << ", 权重=" << weight
                            << ", 父子关系:" << (nIDj == pParentKF || isChild ? "是" : "否")
                            << ", nIDj < nIDi:" << (nIDj < nIDi ? "是" : "否")
                            << ", 已在边集合:" << (sInsertedEdges.count(std::make_pair(std::min(nIDi, nIDj), std::max(nIDi, nIDj))) ? "是" : "否")
                            << std::endl;
                }
                
                



                
                if (nIDj != pParentKF && !isChild) {
                    // 统计所有合格的共视关系
                    count_all_valid_covis++;
                    
                    // 获取共视权重
                    int weight = 0;
                    if (data_.covisibility.find(nIDi) != data_.covisibility.end() &&
                        data_.covisibility[nIDi].find(nIDj) != data_.covisibility[nIDi].end()) {
                        weight = data_.covisibility[nIDi][nIDj];
                    }
                    
                    // 存储共视权重信息
                    covis_weights[nIDi].push_back(std::make_pair(nIDj, weight));
                    
                    // 只处理ID小于当前ID的关键帧（避免重复）
                    if (!pKFn->is_bad && nIDj < nIDi) {
                        // 检查是否已经添加过这条边
                        if (sInsertedEdges.count(std::make_pair(
                            std::min(nIDi, nIDj),
                            std::max(nIDi, nIDj))))
                            continue;
                        
                        // 获取连接关键帧的变换 Snw
                        Eigen::Matrix4d Snw = Eigen::Matrix4d::Identity();
                        
                        if (data_.non_corrected_poses.find(nIDj) != data_.non_corrected_poses.end()) {
                            const auto& non_corrected = data_.non_corrected_poses[nIDj];
                            Snw.block<3, 3>(0, 0) = non_corrected.quaternion.toRotationMatrix();
                            Snw.block<3, 1>(0, 3) = non_corrected.translation;
                        }
                        else if (data_.vertex_initial_poses_Tcw.find(nIDj) != data_.vertex_initial_poses_Tcw.end()) {
                            Snw = data_.vertex_initial_poses_Tcw[nIDj];
                        }
                        else {
                            Snw.block<3, 3>(0, 0) = pKFn->quaternion.toRotationMatrix();
                            Snw.block<3, 1>(0, 3) = pKFn->translation;
                        }
                        
                        // 计算相对变换 Sni = Snw * Swi
                        Eigen::Matrix4d Sni = Snw * Swi;
                        
                        // 添加约束
                        ceres::CostFunction* en = SE3LoopConstraintCost::Create(Sni, information);
                        problem_->AddResidualBlock(en, nullptr, 
                                                 data_.keyframes[nIDi]->se3_state.data(),
                                                 data_.keyframes[nIDj]->se3_state.data());
                        
                        // 在这里添加KF 0的边添加调试代码
                        if (nIDi == 0 || nIDj == 0) {
                            std::cout << "DEBUG KF0: 添加边 KF" << nIDi << " - KF" << nIDj 
                                    << ", KF" << nIDi << "当前边数:" << covis_per_kf[nIDi] + 1
                                    << ", KF" << nIDj << "当前边数:" << covis_per_kf[nIDj] + 1
                                    << std::endl;
                        }
                        
                        // 增加添加到图中的共视边计数
                        count_covis++;
                        covis_per_kf[nIDi]++;
                        covis_per_kf[nIDj]++;
                        normal_edges_added++;  // 包含在总的normal边计数中
                        
                        // 记录已添加的边
                        sInsertedEdges.insert(std::make_pair(
                            std::min(nIDi, nIDj),
                            std::max(nIDi, nIDj)
                        ));
                    }
                }
            }
        }

        // 按ID间隔统计关键帧共视关系
        std::cout << "\n按ID分组的关键帧共视关系统计:" << std::endl;
        for (int id = 0; id <= data_.max_kf_id; id += 20) {
            // 检查是否存在该ID的关键帧
            bool found = false;
            for (const auto& kf_pair : data_.keyframes) {
                if (kf_pair.first == id && !kf_pair.second->is_bad) {
                    found = true;
                    break;
                }
            }
            
            if (found) {
                std::cout << "KF ID " << id << ": 添加到图的共视边数量 = " << covis_per_kf[id] << std::endl;
                
                // 输出该关键帧的所有合格共视关系和权重
                if (covis_weights.find(id) != covis_weights.end()) {
                    std::cout << "  所有合格共视关系 (ID, 权重):" << std::endl;
                    for (const auto& pair : covis_weights[id]) {
                        std::cout << "  -> KF " << pair.first << ", 权重: " << pair.second << std::endl;
                    }
                    std::cout << "  合格共视关系总数: " << covis_weights[id].size() << std::endl;
                } else {
                    std::cout << "  没有合格的共视关系" << std::endl;
                }
            }
        }
        
        std::cout << "\n总共添加到图中的共视边数量: " << count_covis << std::endl;
        std::cout << "所有合格的共视关系总数(包括未添加到图中的): " << count_all_valid_covis << std::endl;

        std::cout << "\nSuccessful Normal Edges: " << normal_edges_added << "/" << (attempted_normal + count_all_valid_covis) << std::endl;
        std::cout << "Total KeyFrames Processed: " << validKFCount << "/" << data_.keyframes.size() << std::endl;
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



    void VerifyVertexPosesConsistency() {
        std::cout << "\n=== 验证顶点位姿一致性 ===" << std::endl;
        
        if (data_.vertex_initial_poses_Tcw.empty()) {
            std::cout << "没有保存的顶点初始位姿，跳过验证" << std::endl;
            return;
        }
        
        int mismatch_count = 0;
        double max_diff = 0.0;
        int max_diff_kf = -1;
        
        for (const auto& kf_pair : data_.keyframes) {
            int kf_id = kf_pair.first;
            const auto& kf = kf_pair.second;
            
            if (kf->is_bad) continue;
            
            // 检查是否有保存的初始位姿
            if (data_.vertex_initial_poses_Tcw.find(kf_id) == data_.vertex_initial_poses_Tcw.end()) 
                continue;
            
            // 获取保存的初始位姿
            const Eigen::Matrix4d& T_saved = data_.vertex_initial_poses_Tcw[kf_id];
            
            // 获取当前关键帧位姿
            Eigen::Matrix4d T_current = Eigen::Matrix4d::Identity();
            T_current.block<3, 3>(0, 0) = kf->quaternion.toRotationMatrix();
            T_current.block<3, 1>(0, 3) = kf->translation;
            
            // 计算位置差异
            Eigen::Vector3d pos_saved = T_saved.block<3, 1>(0, 3);
            Eigen::Vector3d pos_current = T_current.block<3, 1>(0, 3);
            double pos_diff = (pos_saved - pos_current).norm();
            
            // 检查是否有明显差异
            if (pos_diff > 1e-6) {
                mismatch_count++;
                
                if (pos_diff > max_diff) {
                    max_diff = pos_diff;
                    max_diff_kf = kf_id;
                }
                
                // 仅打印一些关键帧的差异
                if (kf_id % 100 == 0 || kf_id == data_.init_kf_id || kf_id == data_.loop_kf_id || kf_id == data_.current_kf_id) {
                    std::cout << "KF" << kf_id << " 位姿差异: " << pos_diff 
                             << " meters | 保存位置: [" << pos_saved.transpose() 
                             << "] | 当前位置: [" << pos_current.transpose() << "]" << std::endl;
                }
            }
        }
        
        std::cout << "总共 " << mismatch_count << " 个关键帧与保存的初始位姿不一致" << std::endl;
        if (max_diff_kf >= 0) {
            std::cout << "最大差异: " << max_diff << " meters, 在KF" << max_diff_kf << std::endl;
        }
    }




    // 获取参数块信息
    void PrintProblemInfo() {
        std::cout << "\n优化问题信息:" << std::endl;
        std::cout << "参数块数量: " << problem_->NumParameterBlocks() << std::endl;
        std::cout << "残差块数量: " << problem_->NumResidualBlocks() << std::endl;
        std::cout << "参数数量: " << problem_->NumParameters() << std::endl;
        std::cout << "残差数量: " << problem_->NumResiduals() << std::endl;
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
    
    // 验证位姿一致性
    optimizer.VerifyVertexPosesConsistency();
    
    // 添加回环约束
    optimizer.AddLoopConstraints();
    

    // 添加正常边约束（生成树）
    optimizer.AddNormalEdgeConstraints();

    
    // 再次验证所有边添加后的位姿一致性
    optimizer.VerifyVertexPosesConsistency();
    
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
