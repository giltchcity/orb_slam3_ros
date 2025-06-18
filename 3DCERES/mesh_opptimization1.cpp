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

// 相机内参
const double FX = 377.535257164;
const double FY = 377.209841379;
const double CX = 328.193371286;
const double CY = 240.426878936;



// 复用你之前的SE3Parameterization类
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


// Ray casting观测数据
struct RayCastObservation {
    int frame_id;
    Eigen::Vector2d pixel;        // (u, v)
    Eigen::Vector3d depth_point;  // 深度投影点
    Eigen::Vector3d mesh_point;   // mesh交点
};

// 3D点结构
struct Point3D {
    int id;
    std::vector<double> position;  // [X, Y, Z] - 用于Ceres优化
    std::vector<RayCastObservation> observations;
    
    Point3D() : position(3, 0.0) {}
    
    void SetPosition(const Eigen::Vector3d& pos) {
        position[0] = pos[0];
        position[1] = pos[1];
        position[2] = pos[2];
    }
    
    Eigen::Vector3d GetPosition() const {
        return Eigen::Vector3d(position[0], position[1], position[2]);
    }
};

// 相机位姿
struct CameraPose {
    int frame_id;
    double timestamp;
    std::vector<double> se3_state;  // [tx, ty, tz, qx, qy, qz, qw]
    
    CameraPose() : se3_state(7, 0.0) {
        se3_state[6] = 1.0;  // qw = 1
    }
    
    void SetFromTwc(const Eigen::Vector3d& t_wc, const Eigen::Quaterniond& q_wc) {
        // 转换Twc到Tcw
        Eigen::Matrix4d T_wc = Eigen::Matrix4d::Identity();
        T_wc.block<3, 3>(0, 0) = q_wc.toRotationMatrix();
        T_wc.block<3, 1>(0, 3) = t_wc;
        
        Eigen::Matrix4d T_cw = T_wc.inverse();
        
        // 提取Tcw的平移和旋转
        Eigen::Vector3d t_cw = T_cw.block<3, 1>(0, 3);
        Eigen::Matrix3d R_cw = T_cw.block<3, 3>(0, 0);
        Eigen::Quaterniond q_cw(R_cw);
        
        // 设置SE3状态
        se3_state[0] = t_cw[0];
        se3_state[1] = t_cw[1];
        se3_state[2] = t_cw[2];
        se3_state[3] = q_cw.x();
        se3_state[4] = q_cw.y();
        se3_state[5] = q_cw.z();
        se3_state[6] = q_cw.w();
    }
};

// 重投影误差代价函数
class ReprojectionCost {
public:
    ReprojectionCost(const Eigen::Vector2d& observation)
        : observed_pixel_(observation) {}
    
    template <typename T>
    bool operator()(const T* const camera_pose,  // [tx, ty, tz, qx, qy, qz, qw]
                    const T* const point_3d,      // [X, Y, Z]
                    T* residuals) const {
        // 提取相机位姿 (Tcw)
        Eigen::Matrix<T, 3, 1> t_cw(camera_pose[0], camera_pose[1], camera_pose[2]);
        Eigen::Quaternion<T> q_cw(camera_pose[6], camera_pose[3], camera_pose[4], camera_pose[5]);
        
        // 3D点（世界坐标系）
        Eigen::Matrix<T, 3, 1> P_w(point_3d[0], point_3d[1], point_3d[2]);
        
        // 转换到相机坐标系
        Eigen::Matrix<T, 3, 1> P_c = q_cw * P_w + t_cw;
        
        // 检查点是否在相机前面
        if (P_c[2] <= T(0.0)) {
            residuals[0] = T(0.0);
            residuals[1] = T(0.0);
            return false;
        }
        
        // 投影到像素平面
        T u = T(FX) * (P_c[0] / P_c[2]) + T(CX);
        T v = T(FY) * (P_c[1] / P_c[2]) + T(CY);
        
        // 计算残差
        residuals[0] = u - T(observed_pixel_[0]);
        residuals[1] = v - T(observed_pixel_[1]);
        
        return true;
    }
    
    static ceres::CostFunction* Create(const Eigen::Vector2d& observation) {
        return new ceres::AutoDiffCostFunction<ReprojectionCost, 2, 7, 3>(
            new ReprojectionCost(observation));
    }
    
private:
    Eigen::Vector2d observed_pixel_;
};

// 位姿先验约束（保持位姿接近初始值）
class PosePriorCost {
public:
    PosePriorCost(const std::vector<double>& prior_pose, double weight)
        : prior_pose_(prior_pose), weight_(weight) {}
    
    template <typename T>
    bool operator()(const T* const pose, T* residuals) const {
        // 位置差异
        for (int i = 0; i < 3; ++i) {
            residuals[i] = weight_ * (pose[i] - T(prior_pose_[i]));
        }
        
        // 旋转差异（四元数）
        Eigen::Quaternion<T> q_current(pose[6], pose[3], pose[4], pose[5]);
        Eigen::Quaternion<T> q_prior;
        q_prior.w() = T(prior_pose_[6]);
        q_prior.x() = T(prior_pose_[3]);
        q_prior.y() = T(prior_pose_[4]);
        q_prior.z() = T(prior_pose_[5]);
        
        // 计算旋转差异
        Eigen::Quaternion<T> q_diff = q_prior.inverse() * q_current;
        
        // 使用轴角表示
        T angle = T(2.0) * acos(q_diff.w());
        residuals[3] = weight_ * angle;
        residuals[4] = T(0.0);
        residuals[5] = T(0.0);
        
        return true;
    }
    
    static ceres::CostFunction* Create(const std::vector<double>& prior_pose, double weight) {
        return new ceres::AutoDiffCostFunction<PosePriorCost, 6, 7>(
            new PosePriorCost(prior_pose, weight));
    }
    
private:
    std::vector<double> prior_pose_;
    double weight_;
};

// Mesh优化器主类
class MeshOptimizer {
private:
    std::map<int, CameraPose> camera_poses_;
    std::map<int, std::shared_ptr<Point3D>> points_;
    std::unique_ptr<ceres::Problem> problem_;
    
    // 用于3D点聚类
    std::vector<std::vector<Eigen::Vector3d>> point_clusters_;
    
public:
    MeshOptimizer() {
        problem_ = std::make_unique<ceres::Problem>();
    }
    
    // 加载优化后的相机位姿（TUM格式）
    bool LoadOptimizedPoses(const std::string& pose_file) {
        std::ifstream file(pose_file);
        if (!file.is_open()) {
            std::cerr << "无法打开位姿文件: " << pose_file << std::endl;
            return false;
        }
        
        std::string line;
        int frame_id = 0;
        
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;
            
            std::istringstream iss(line);
            double timestamp, tx, ty, tz, qx, qy, qz, qw;
            
            if (iss >> timestamp >> tx >> ty >> tz >> qx >> qy >> qz >> qw) {
                CameraPose pose;
                pose.frame_id = frame_id;
                pose.timestamp = timestamp;
                
                // TUM格式是Twc，需要转换
                Eigen::Vector3d t_wc(tx, ty, tz);
                Eigen::Quaterniond q_wc(qw, qx, qy, qz);
                pose.SetFromTwc(t_wc, q_wc);
                
                camera_poses_[frame_id] = pose;
                frame_id++;
            }
        }
        
        std::cout << "加载了 " << camera_poses_.size() << " 个相机位姿" << std::endl;
        return true;
    }
    
    // 加载Ray Casting数据
    bool LoadRayCastData(const std::string& raycast_file) {
        std::ifstream file(raycast_file);
        if (!file.is_open()) {
            std::cerr << "无法打开ray casting文件: " << raycast_file << std::endl;
            return false;
        }
        
        std::string line;
        std::vector<RayCastObservation> all_observations;
        
        // 第一遍：读取所有观测
        while (std::getline(file, line)) {
            if (line.empty()) continue;
            
            std::istringstream iss(line);
            int frame_id;
            double u, v, xd, yd, zd, xm, ym, zm;
            
            if (iss >> frame_id >> u >> v >> xd >> yd >> zd >> xm >> ym >> zm) {
                RayCastObservation obs;
                obs.frame_id = frame_id;
                obs.pixel = Eigen::Vector2d(u, v);
                obs.depth_point = Eigen::Vector3d(xd, yd, zd);
                obs.mesh_point = Eigen::Vector3d(xm, ym, zm);
                
                all_observations.push_back(obs);
            }
        }
        
        std::cout << "读取了 " << all_observations.size() << " 个ray casting观测" << std::endl;
        
        // 第二遍：聚类3D点（合并相近的点）
        const double MERGE_THRESHOLD = 0.01;  // 10mm
        
        for (const auto& obs : all_observations) {
            // 查找是否有相近的已存在点
            int best_cluster = -1;
            double min_dist = MERGE_THRESHOLD;
            
            for (size_t i = 0; i < point_clusters_.size(); ++i) {
                double dist = (point_clusters_[i][0] - obs.mesh_point).norm();
                if (dist < min_dist) {
                    min_dist = dist;
                    best_cluster = i;
                }
            }
            
            if (best_cluster >= 0) {
                // 添加到现有簇
                point_clusters_[best_cluster].push_back(obs.mesh_point);
            } else {
                // 创建新簇
                point_clusters_.push_back({obs.mesh_point});
            }
        }
        
        // 第三遍：创建3D点并关联观测
        for (size_t cluster_id = 0; cluster_id < point_clusters_.size(); ++cluster_id) {
            auto point = std::make_shared<Point3D>();
            point->id = cluster_id;
            
            // 计算簇的平均位置
            Eigen::Vector3d avg_pos = Eigen::Vector3d::Zero();
            for (const auto& pos : point_clusters_[cluster_id]) {
                avg_pos += pos;
            }
            avg_pos /= point_clusters_[cluster_id].size();
            point->SetPosition(avg_pos);
            
            // 关联观测
            for (const auto& obs : all_observations) {
                double dist = (obs.mesh_point - avg_pos).norm();
                if (dist < MERGE_THRESHOLD) {
                    point->observations.push_back(obs);
                }
            }
            
            points_[cluster_id] = point;
        }
        
        std::cout << "创建了 " << points_.size() << " 个唯一3D点" << std::endl;
        
        // 统计信息
        int total_observations = 0;
        for (const auto& p : points_) {
            total_observations += p.second->observations.size();
        }
        std::cout << "平均每个3D点有 " << (double)total_observations / points_.size() 
                  << " 个观测" << std::endl;
        
        return true;
    }
    
    // 设置优化问题
    void SetupOptimization() {
        std::cout << "\n设置优化问题..." << std::endl;
        
        // 1. 添加相机位姿参数
        for (auto& pose_pair : camera_poses_) {
            auto& pose = pose_pair.second;
            
            // 添加位姿参数块
            problem_->AddParameterBlock(pose.se3_state.data(), 7);
            problem_->SetManifold(pose.se3_state.data(), new SE3Parameterization());
            
            // 添加位姿先验（强约束，避免位姿大幅变化）
            std::vector<double> prior_pose = pose.se3_state;
            ceres::CostFunction* prior_cost = PosePriorCost::Create(prior_pose, 100.0);
            problem_->AddResidualBlock(prior_cost, nullptr, pose.se3_state.data());
        }
        
        std::cout << "添加了 " << camera_poses_.size() << " 个相机位姿参数" << std::endl;
        
        // 2. 添加3D点参数和重投影约束
        int constraint_count = 0;
        for (auto& point_pair : points_) {
            auto& point = point_pair.second;
            
            // 添加3D点参数块
            problem_->AddParameterBlock(point->position.data(), 3);
            
            // 为每个观测添加重投影约束
            for (const auto& obs : point->observations) {
                // 检查是否有对应的相机位姿
                if (camera_poses_.find(obs.frame_id) == camera_poses_.end()) {
                    continue;
                }
                
                // 创建重投影约束
                ceres::CostFunction* cost = ReprojectionCost::Create(obs.pixel);
                
                // 使用Huber损失函数处理外点
                ceres::LossFunction* loss = new ceres::HuberLoss(1.0);
                
                // 添加残差块
                problem_->AddResidualBlock(
                    cost, 
                    loss,
                    camera_poses_[obs.frame_id].se3_state.data(),
                    point->position.data()
                );
                
                constraint_count++;
            }
        }
        
        std::cout << "添加了 " << points_.size() << " 个3D点参数" << std::endl;
        std::cout << "添加了 " << constraint_count << " 个重投影约束" << std::endl;
    }
    
    // 执行优化
    bool Optimize() {
        std::cout << "\n开始优化..." << std::endl;
        
        // 配置求解器
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::SPARSE_SCHUR;
        options.minimizer_progress_to_stdout = true;
        options.max_num_iterations = 100;
        options.num_threads = 8;
        
        // 求解
        ceres::Solver::Summary summary;
        ceres::Solve(options, problem_.get(), &summary);
        
        std::cout << summary.FullReport() << std::endl;
        
        return summary.IsSolutionUsable();
    }
    
  // 替换原来的OutputOptimizedPoints函数
  void OutputOptimizedPointsComparison(const std::string& output_file) {
      std::ofstream file(output_file);
      if (!file.is_open()) {
          std::cerr << "无法创建输出文件: " << output_file << std::endl;
          return;
      }
      
      // 写入头部
      file << "# Optimized 3D points comparison" << std::endl;
      file << "# Format: point_id x_before y_before z_before x_after y_after z_after movement_distance num_observations" << std::endl;
      file << std::fixed << std::setprecision(6);
      
      // 统计信息
      double total_movement = 0.0;
      double max_movement = 0.0;
      int min_observations = INT_MAX;
      int max_observations = 0;
      
      // 输出每个点的优化前后对比
      for (const auto& point_pair : points_) {
          const auto& point = point_pair.second;
          int point_id = point->id;
          
          // 获取初始位置（从第一个观测的mesh_point）
          Eigen::Vector3d initial_pos = point_clusters_[point_id][0];
          
          // 获取优化后位置
          Eigen::Vector3d final_pos = point->GetPosition();
          
          // 计算移动距离
          double movement = (final_pos - initial_pos).norm();
          total_movement += movement;
          max_movement = std::max(max_movement, movement);
          
          // 统计观测数量
          int num_obs = point->observations.size();
          min_observations = std::min(min_observations, num_obs);
          max_observations = std::max(max_observations, num_obs);
          
          // 写入文件
          file << point_id << " "
               << initial_pos[0] << " " << initial_pos[1] << " " << initial_pos[2] << " "
               << final_pos[0] << " " << final_pos[1] << " " << final_pos[2] << " "
               << movement << " "
               << num_obs << std::endl;
      }
      
      file.close();
      
      // 输出统计摘要
      std::cout << "\n=== 3D点优化统计 ===" << std::endl;
      std::cout << "总共优化了 " << points_.size() << " 个采样3D点" << std::endl;
      std::cout << "平均移动距离: " << total_movement / points_.size() << " 米" << std::endl;
      std::cout << "最大移动距离: " << max_movement << " 米" << std::endl;
      std::cout << "观测数量范围: [" << min_observations << ", " << max_observations << "]" << std::endl;
      std::cout << "结果已保存到: " << output_file << std::endl;
  }
  
  // 额外输出一个简化版本，只包含优化后的位置
  void OutputOptimizedPointsSimple(const std::string& output_file) {
      std::ofstream file(output_file);
      if (!file.is_open()) {
          std::cerr << "无法创建输出文件: " << output_file << std::endl;
          return;
      }
      
      // 写入头部
      file << "# Optimized 3D points" << std::endl;
      file << "# Format: x y z" << std::endl;
      file << std::fixed << std::setprecision(6);
      
      // 按ID排序输出
      std::vector<int> sorted_ids;
      for (const auto& p : points_) {
          sorted_ids.push_back(p.first);
      }
      std::sort(sorted_ids.begin(), sorted_ids.end());
      
      for (int id : sorted_ids) {
          const auto& point = points_[id];
          Eigen::Vector3d pos = point->GetPosition();
          file << pos[0] << " " << pos[1] << " " << pos[2] << std::endl;
      }
      
      std::cout << "优化后的3D点位置已保存到: " << output_file << std::endl;
  }
  
  // 输出详细的观测信息（用于验证）
  void OutputDetailedObservations(const std::string& output_file) {
      std::ofstream file(output_file);
      if (!file.is_open()) {
          std::cerr << "无法创建输出文件: " << output_file << std::endl;
          return;
      }
      
      file << "# Detailed observations for each 3D point" << std::endl;
      file << "# Format: point_id frame_id u v reprojection_error_before reprojection_error_after" << std::endl;
      file << std::fixed << std::setprecision(6);
      
      for (const auto& point_pair : points_) {
          const auto& point = point_pair.second;
          Eigen::Vector3d initial_pos = point_clusters_[point->id][0];
          Eigen::Vector3d final_pos = point->GetPosition();
          
          for (const auto& obs : point->observations) {
              if (camera_poses_.find(obs.frame_id) == camera_poses_.end()) continue;
              
              const auto& pose = camera_poses_[obs.frame_id];
              
              // 计算优化前的重投影误差
              double error_before = ComputeReprojectionError(
                  initial_pos, pose.se3_state, obs.pixel);
              
              // 计算优化后的重投影误差
              double error_after = ComputeReprojectionError(
                  final_pos, pose.se3_state, obs.pixel);
              
              file << point->id << " "
                   << obs.frame_id << " "
                   << obs.pixel[0] << " " << obs.pixel[1] << " "
                   << error_before << " "
                   << error_after << std::endl;
          }
      }
      
      std::cout << "详细观测信息已保存到: " << output_file << std::endl;
  }
  
  private:
      // 辅助函数：计算重投影误差
      double ComputeReprojectionError(const Eigen::Vector3d& point_3d,
                                     const std::vector<double>& camera_pose,
                                     const Eigen::Vector2d& observed_pixel) {
          // 提取相机位姿
          Eigen::Vector3d t_cw(camera_pose[0], camera_pose[1], camera_pose[2]);
          Eigen::Quaterniond q_cw(camera_pose[6], camera_pose[3], camera_pose[4], camera_pose[5]);
          
          // 转换到相机坐标系
          Eigen::Vector3d P_c = q_cw * point_3d + t_cw;
          
          if (P_c[2] <= 0) return std::numeric_limits<double>::max();
          
          // 投影
          double u = FX * (P_c[0] / P_c[2]) + CX;
          double v = FY * (P_c[1] / P_c[2]) + CY;
          
          // 计算像素误差
          return std::sqrt(std::pow(u - observed_pixel[0], 2) + 
                          std::pow(v - observed_pixel[1], 2));
      }
    
    // 输出统计信息
    void PrintStatistics() {
        std::cout << "\n=== 优化统计 ===" << std::endl;
        
        // 计算3D点的平均移动距离
        double total_movement = 0.0;
        double max_movement = 0.0;
        
        for (size_t i = 0; i < point_clusters_.size(); ++i) {
            if (points_.find(i) != points_.end()) {
                Eigen::Vector3d initial_pos = point_clusters_[i][0];  // 初始位置
                Eigen::Vector3d final_pos = points_[i]->GetPosition();
                double movement = (final_pos - initial_pos).norm();
                
                total_movement += movement;
                max_movement = std::max(max_movement, movement);
            }
        }
        
        std::cout << "3D点平均移动距离: " << total_movement / points_.size() << " 米" << std::endl;
        std::cout << "3D点最大移动距离: " << max_movement << " 米" << std::endl;
        
        // 输出一些示例点的优化前后对比
        std::cout << "\n示例3D点优化前后对比:" << std::endl;
        int count = 0;
        for (const auto& point_pair : points_) {
            if (count++ >= 5) break;  // 只显示前5个
            
            const auto& point = point_pair.second;
            Eigen::Vector3d initial_pos = point_clusters_[point->id][0];
            Eigen::Vector3d final_pos = point->GetPosition();
            
            std::cout << "点 " << point->id << ":" << std::endl;
            std::cout << "  初始位置: [" << initial_pos.transpose() << "]" << std::endl;
            std::cout << "  优化后位置: [" << final_pos.transpose() << "]" << std::endl;
            std::cout << "  移动距离: " << (final_pos - initial_pos).norm() << " 米" << std::endl;
            std::cout << "  观测数量: " << point->observations.size() << std::endl;
        }
    }
};





// 主函数
int main() {
    // 文件路径
    std::string optimized_poses = "/Datasets/CERES_Work/output/trajectory_after_optimization.txt";
    std::string raycast_data = "/Datasets/CERES_Work/3DPintput/raycast_combined_points_no_loop.txt";  // 替换为实际路径
    std::string output_dir = "/Datasets/CERES_Work/output/mesh_optimization";
    
    // 创建输出目录
    system(("mkdir -p " + output_dir).c_str());
    
    // 创建优化器
    MeshOptimizer optimizer;
    
    // 加载数据
    if (!optimizer.LoadOptimizedPoses(optimized_poses)) {
        std::cerr << "加载优化后的位姿失败" << std::endl;
        return -1;
    }
    
    if (!optimizer.LoadRayCastData(raycast_data)) {
        std::cerr << "加载ray casting数据失败" << std::endl;
        return -1;
    }
    
    // 设置优化问题
    optimizer.SetupOptimization();
    
    // 执行优化
    // 在main函数中，替换输出部分
    if (optimizer.Optimize()) {
        std::cout << "\n优化成功完成！" << std::endl;
        
        // 输出多种格式的结果
        optimizer.OutputOptimizedPointsComparison(output_dir + "/points_comparison.txt");
        optimizer.OutputOptimizedPointsSimple(output_dir + "/points_optimized.txt");
        optimizer.OutputDetailedObservations(output_dir + "/observations_detail.txt");
        
        std::cout << "\n输出文件说明：" << std::endl;
        std::cout << "1. points_comparison.txt - 3D点优化前后对比" << std::endl;
        std::cout << "2. points_optimized.txt - 仅包含优化后的3D点坐标" << std::endl;
        std::cout << "3. observations_detail.txt - 详细的观测和重投影误差" << std::endl;
    } else {
        std::cerr << "优化失败" << std::endl;
        return -1;
    }
    
    return 0;
}
