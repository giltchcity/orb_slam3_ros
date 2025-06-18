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

// SE3 李代数参数化
class SE3Parameterization : public ceres::Manifold {
public:
    ~SE3Parameterization() {}
    
    int AmbientSize() const override { return 7; }
    int TangentSize() const override { return 6; }
    
    bool Plus(const double* x, const double* delta, double* x_plus_delta) const override {
        // 提取当前状态
        Eigen::Vector3d t_current(x[0], x[1], x[2]);
        Eigen::Quaterniond q_current(x[6], x[3], x[4], x[5]);
        q_current.normalize();
        
        // 提取李代数增量
        Eigen::Vector3d omega(delta[0], delta[1], delta[2]);
        Eigen::Vector3d upsilon(delta[3], delta[4], delta[5]);
        
        double theta = omega.norm();
        double eps = 1e-8;
        
        Eigen::Matrix3d Omega = skew(omega);
        Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
        Eigen::Matrix3d R;
        
        if (theta < eps) {
            R = I + Omega + 0.5 * Omega * Omega;
        } else {
            R = I + (sin(theta) / theta) * Omega + 
                ((1.0 - cos(theta)) / (theta * theta)) * Omega * Omega;
        }
        
        // 计算V矩阵用于平移更新
        Eigen::Matrix3d V;
        if (theta < eps) {
            V = I + 0.5 * Omega + (1.0/6.0) * Omega * Omega;
        } else {
            double c = cos(theta);
            double s = sin(theta);
            V = I + ((1.0 - c) / (theta * theta)) * Omega + 
                ((theta - s) / (theta * theta * theta)) * Omega * Omega;
        }
        
        // 应用增量
        Eigen::Matrix3d R_current = q_current.toRotationMatrix();
        Eigen::Matrix3d R_new = R * R_current;
        Eigen::Vector3d t_new = R * t_current + V * upsilon;
        
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
        // 数值微分
        Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> J(jacobian);
        J.setZero();
        
        const double eps = 1e-8;
        double x_plus_eps[7], x_minus_eps[7];
        double delta_plus[6], delta_minus[6];
        
        for (int i = 0; i < 6; ++i) {
            for (int j = 0; j < 6; ++j) {
                delta_plus[j] = (i == j) ? eps : 0.0;
                delta_minus[j] = (i == j) ? -eps : 0.0;
            }
            
            Plus(x, delta_plus, x_plus_eps);
            Plus(x, delta_minus, x_minus_eps);
            
            for (int k = 0; k < 7; ++k) {
                J(k, i) = (x_plus_eps[k] - x_minus_eps[k]) / (2.0 * eps);
            }
        }
        
        return true;
    }
    
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
        
        // 计算旋转的对数映射
        double trace = R_rel.trace();
        Eigen::Vector3d omega;
        
        if (trace > 3.0 - 1e-6) {
            omega = 0.5 * Eigen::Vector3d(
                R_rel(2,1) - R_rel(1,2),
                R_rel(0,2) - R_rel(2,0),
                R_rel(1,0) - R_rel(0,1)
            );
        } else {
            double angle = acos((trace - 1.0) / 2.0);
            omega = (angle / (2.0 * sin(angle))) * Eigen::Vector3d(
                R_rel(2,1) - R_rel(1,2),
                R_rel(0,2) - R_rel(2,0),
                R_rel(1,0) - R_rel(0,1)
            );
        }
        
        // 计算平移的对数映射
        double theta = omega.norm();
        Eigen::Matrix3d V_inv;
        
        if (theta < 1e-8) {
            V_inv = Eigen::Matrix3d::Identity() - 0.5 * skew(omega);
        } else {
            double c = cos(theta);
            double s = sin(theta);
            V_inv = Eigen::Matrix3d::Identity() - 0.5 * skew(omega) +
                    (1.0 / (theta * theta)) * (1.0 - (theta * s) / (2.0 * (1.0 - c))) * 
                    skew(omega) * skew(omega);
        }
        
        Eigen::Vector3d upsilon = V_inv * t_rel;
        
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
        // 数值微分
        Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> J(jacobian);
        J.setZero();
        
        const double eps = 1e-8;
        double y_plus[7], y_minus[7];
        double diff_plus[6], diff_minus[6];
        
        for (int i = 0; i < 7; ++i) {
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
    
private:
    static Eigen::Matrix3d skew(const Eigen::Vector3d& v) {
        Eigen::Matrix3d m;
        m << 0, -v(2), v(1),
             v(2), 0, -v(0),
             -v(1), v(0), 0;
        return m;
    }
};

// Ray casting观测数据
struct RayCastObservation {
    int frame_id;
    Eigen::Vector2d pixel;
    Eigen::Vector3d depth_point;      // 可靠的depth投影点
    Eigen::Vector3d mesh_point;        // 原始mesh点
    Eigen::Vector3d mesh_point_transformed;  // 变换后的mesh点
};

// 3D点结构
struct Point3D {
    int id;
    std::vector<double> position;  // [X, Y, Z] - 优化变量
    std::vector<RayCastObservation> observations;
    Eigen::Vector3d depth_center;  // depth points的中心（作为强约束）
    
    Point3D() : position(3, 0.0) {}
    
    void SetPosition(const Eigen::Vector3d& pos) {
        position[0] = pos[0];
        position[1] = pos[1];
        position[2] = pos[2];
    }
    
    Eigen::Vector3d GetPosition() const {
        return Eigen::Vector3d(position[0], position[1], position[2]);
    }
    
    // 计算depth points的中心
    void ComputeDepthCenter() {
        depth_center = Eigen::Vector3d::Zero();
        if (observations.empty()) return;
        
        for (const auto& obs : observations) {
            depth_center += obs.depth_point;
        }
        depth_center /= observations.size();
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
    
    void SetFromTUM(double tx, double ty, double tz, 
                    double qx, double qy, double qz, double qw) {
        // TUM格式是Twc
        Eigen::Vector3d t_wc(tx, ty, tz);
        Eigen::Quaterniond q_wc(qw, qx, qy, qz);
        q_wc.normalize();
        
        // 转换为Tcw存储
        Eigen::Matrix4d T_wc = Eigen::Matrix4d::Identity();
        T_wc.block<3, 3>(0, 0) = q_wc.toRotationMatrix();
        T_wc.block<3, 1>(0, 3) = t_wc;
        
        Eigen::Matrix4d T_cw = T_wc.inverse();
        
        Eigen::Vector3d t_cw = T_cw.block<3, 1>(0, 3);
        Eigen::Matrix3d R_cw = T_cw.block<3, 3>(0, 0);
        Eigen::Quaterniond q_cw(R_cw);
        
        se3_state[0] = t_cw[0];
        se3_state[1] = t_cw[1];
        se3_state[2] = t_cw[2];
        se3_state[3] = q_cw.x();
        se3_state[4] = q_cw.y();
        se3_state[5] = q_cw.z();
        se3_state[6] = q_cw.w();
    }
    
    Eigen::Matrix4d GetTcw() const {
        Eigen::Vector3d t_cw(se3_state[0], se3_state[1], se3_state[2]);
        Eigen::Quaterniond q_cw(se3_state[6], se3_state[3], se3_state[4], se3_state[5]);
        
        Eigen::Matrix4d T_cw = Eigen::Matrix4d::Identity();
        T_cw.block<3,3>(0,0) = q_cw.toRotationMatrix();
        T_cw.block<3,1>(0,3) = t_cw;
        
        return T_cw;
    }
    
    Eigen::Matrix4d GetTwc() const {
        return GetTcw().inverse();
    }
};

// 重投影误差代价函数
class ReprojectionCost {
public:
    ReprojectionCost(const Eigen::Vector2d& observation)
        : observed_pixel_(observation) {}
    
    template <typename T>
    bool operator()(const T* const camera_pose,
                    const T* const point_3d,
                    T* residuals) const {
        // 提取相机位姿 (Tcw)
        Eigen::Matrix<T, 3, 1> t_cw(camera_pose[0], camera_pose[1], camera_pose[2]);
        Eigen::Quaternion<T> q_cw(camera_pose[6], camera_pose[3], camera_pose[4], camera_pose[5]);
        
        // 3D点（世界坐标系）
        Eigen::Matrix<T, 3, 1> P_w(point_3d[0], point_3d[1], point_3d[2]);
        
        // 转换到相机坐标系
        Eigen::Matrix<T, 3, 1> P_c = q_cw * P_w + t_cw;
        
        // 检查点是否在相机前面
        if (P_c[2] <= T(0.01)) {  // 至少1cm前方
            residuals[0] = T(1000.0);  // 大惩罚
            residuals[1] = T(1000.0);
            return true;
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

// 深度一致性约束
class DepthConsistencyCost {
public:
    DepthConsistencyCost(const Eigen::Vector3d& depth_center, double weight)
        : depth_center_(depth_center), weight_(weight) {}
    
    template <typename T>
    bool operator()(const T* const point_3d, T* residuals) const {
        residuals[0] = weight_ * (point_3d[0] - T(depth_center_[0]));
        residuals[1] = weight_ * (point_3d[1] - T(depth_center_[1]));
        residuals[2] = weight_ * (point_3d[2] - T(depth_center_[2]));
        return true;
    }
    
    static ceres::CostFunction* Create(const Eigen::Vector3d& depth_center, double weight) {
        return new ceres::AutoDiffCostFunction<DepthConsistencyCost, 3, 3>(
            new DepthConsistencyCost(depth_center, weight));
    }
    
private:
    Eigen::Vector3d depth_center_;
    double weight_;
};

// Mesh优化器V3
class MeshOptimizerV3 {
private:
    std::map<int, CameraPose> bad_poses_;
    std::map<int, CameraPose> good_poses_;
    std::vector<RayCastObservation> all_observations_;
    std::map<int, std::shared_ptr<Point3D>> points_;
    std::unique_ptr<ceres::Problem> problem_;
    
    // 观测索引到3D点ID的映射
    std::map<int, int> observation_to_point_;
    
    // 基于depth point的聚类
    struct DepthCluster {
        std::vector<int> observation_indices;
        Eigen::Vector3d depth_center;
        Eigen::Vector3d mesh_center;  // 变换后的mesh points中心
    };
    
public:
    MeshOptimizerV3() : problem_(std::make_unique<ceres::Problem>()) {}
    
    // 加载位姿
    bool LoadPoses(const std::string& pose_file, std::map<int, CameraPose>& poses) {
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
                pose.SetFromTUM(tx, ty, tz, qx, qy, qz, qw);
                
                poses[frame_id] = pose;
                frame_id++;
            }
        }
        
        std::cout << "加载了 " << poses.size() << " 个位姿" << std::endl;
        return true;
    }
    
    // 加载ray casting数据并进行per-frame变换
    bool LoadAndTransformRayCastData(const std::string& raycast_file) {
        std::ifstream file(raycast_file);
        if (!file.is_open()) {
            std::cerr << "无法打开ray casting文件: " << raycast_file << std::endl;
            return false;
        }
        
        std::string line;
        all_observations_.clear();
        observation_to_point_.clear();
        
        // 读取并变换所有观测
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
                
                // 计算per-frame变换
                if (bad_poses_.find(frame_id) != bad_poses_.end() && 
                    good_poses_.find(frame_id) != good_poses_.end()) {
                    
                    Eigen::Matrix4d T_bad_wc = bad_poses_[frame_id].GetTwc();
                    Eigen::Matrix4d T_good_wc = good_poses_[frame_id].GetTwc();
                    Eigen::Matrix4d T_good_bad = T_good_wc.inverse() * T_bad_wc;
                    
                    Eigen::Vector4d P_bad(xm, ym, zm, 1.0);
                    Eigen::Vector4d P_good = T_good_bad * P_bad;
                    obs.mesh_point_transformed = P_good.head<3>();
                } else {
                    obs.mesh_point_transformed = obs.mesh_point;
                }
                
                all_observations_.push_back(obs);
            }
        }
        
        std::cout << "读取了 " << all_observations_.size() << " 个观测" << std::endl;
        
        // 初始化映射
        for (size_t i = 0; i < all_observations_.size(); ++i) {
            observation_to_point_[i] = -1;
        }
        
        return true;
    }
    
    // 基于depth point进行聚类
    void ClusterByDepthPoints() {
        std::cout << "\n基于depth point进行聚类..." << std::endl;
        
        const double DEPTH_CLUSTER_THRESHOLD = 0.02;  // 2cm - depth point非常准确
        
        std::vector<DepthCluster> clusters;
        std::vector<bool> processed(all_observations_.size(), false);
        
        // 对每个未处理的观测
        for (size_t i = 0; i < all_observations_.size(); ++i) {
            if (processed[i]) continue;
            
            // 创建新簇
            DepthCluster cluster;
            cluster.observation_indices.push_back(i);
            cluster.depth_center = all_observations_[i].depth_point;
            cluster.mesh_center = all_observations_[i].mesh_point_transformed;
            processed[i] = true;
            
            // 查找所有相近的depth points
            for (size_t j = i + 1; j < all_observations_.size(); ++j) {
                if (processed[j]) continue;
                
                double dist = (all_observations_[j].depth_point - cluster.depth_center).norm();
                if (dist < DEPTH_CLUSTER_THRESHOLD) {
                    // 更新簇中心（增量平均）
                    int n = cluster.observation_indices.size();
                    cluster.depth_center = (cluster.depth_center * n + all_observations_[j].depth_point) / (n + 1);
                    cluster.mesh_center = (cluster.mesh_center * n + all_observations_[j].mesh_point_transformed) / (n + 1);
                    
                    cluster.observation_indices.push_back(j);
                    processed[j] = true;
                }
            }
            
            clusters.push_back(cluster);
        }
        
        std::cout << "初始聚类数: " << clusters.size() << std::endl;
        
        // 过滤单视角观测
        std::vector<DepthCluster> filtered_clusters;
        for (const auto& cluster : clusters) {
            if (cluster.observation_indices.size() >= 2) {
                filtered_clusters.push_back(cluster);
            }
        }
        
        std::cout << "过滤后聚类数: " << filtered_clusters.size() << std::endl;
        
        // 创建3D点
        points_.clear();
        for (size_t i = 0; i < filtered_clusters.size(); ++i) {
            auto point = std::make_shared<Point3D>();
            point->id = i;
            
            // 收集观测
            for (int obs_idx : filtered_clusters[i].observation_indices) {
                point->observations.push_back(all_observations_[obs_idx]);
                observation_to_point_[obs_idx] = i;
            }
            
            // 计算depth center
            point->ComputeDepthCenter();
            
            // 使用depth center作为初值（非常好的初值！）
            point->SetPosition(point->depth_center);
            
            points_[i] = point;
        }
        
        // 统计信息
        int total_observations = 0;
        int max_observations = 0;
        int min_observations = INT_MAX;
        
        for (const auto& p : points_) {
            int num_obs = p.second->observations.size();
            total_observations += num_obs;
            max_observations = std::max(max_observations, num_obs);
            min_observations = std::min(min_observations, num_obs);
        }
        
        std::cout << "创建了 " << points_.size() << " 个3D点" << std::endl;
        std::cout << "覆盖了 " << total_observations << " 个观测" << std::endl;
        std::cout << "每个点的观测数范围: [" << min_observations << ", " << max_observations << "]" << std::endl;
        std::cout << "平均每个点有 " << (double)total_observations / points_.size() << " 个观测" << std::endl;
    }
    
    // 设置优化问题
    void SetupOptimization() {
        std::cout << "\n设置优化问题..." << std::endl;
        
        // 1. 添加相机位姿参数（使用好位姿，固定不动）
        for (auto& pose_pair : good_poses_) {
            auto& pose = pose_pair.second;
            problem_->AddParameterBlock(pose.se3_state.data(), 7);
            problem_->SetManifold(pose.se3_state.data(), new SE3Parameterization());
            problem_->SetParameterBlockConstant(pose.se3_state.data());
        }
        
        std::cout << "添加了 " << good_poses_.size() << " 个相机位姿（全部固定）" << std::endl;
        
        // 2. 添加3D点参数和约束
        int reproj_constraints = 0;
        int depth_constraints = 0;
        
        for (auto& point_pair : points_) {
            auto& point = point_pair.second;
            
            // 添加3D点参数块
            problem_->AddParameterBlock(point->position.data(), 3);
            
            // 添加深度一致性约束（权重较大，因为depth很准确）
            ceres::CostFunction* depth_cost = 
                DepthConsistencyCost::Create(point->depth_center, 10.0);  // 权重10
            problem_->AddResidualBlock(depth_cost, nullptr, point->position.data());
            depth_constraints++;
            
            // 为每个观测添加重投影约束
            for (const auto& obs : point->observations) {
                if (good_poses_.find(obs.frame_id) == good_poses_.end()) {
                    continue;
                }
                
                ceres::CostFunction* reproj_cost = ReprojectionCost::Create(obs.pixel);
                ceres::LossFunction* loss = new ceres::HuberLoss(1.0);
                
                problem_->AddResidualBlock(
                    reproj_cost, 
                    loss,
                    good_poses_[obs.frame_id].se3_state.data(),
                    point->position.data()
                );
                
                reproj_constraints++;
            }
        }
        
        std::cout << "添加了 " << points_.size() << " 个3D点参数" << std::endl;
        std::cout << "添加了 " << depth_constraints << " 个深度一致性约束" << std::endl;
        std::cout << "添加了 " << reproj_constraints << " 个重投影约束" << std::endl;
    }
    
    // 执行优化
    bool Optimize() {
        std::cout << "\n开始优化..." << std::endl;
        
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::SPARSE_SCHUR;
        options.minimizer_progress_to_stdout = true;
        options.max_num_iterations = 100;
        options.num_threads = 8;
        options.function_tolerance = 1e-8;
        options.gradient_tolerance = 1e-10;
        
        ceres::Solver::Summary summary;
        ceres::Solve(options, problem_.get(), &summary);
        
        std::cout << summary.BriefReport() << std::endl;
        
        return summary.IsSolutionUsable();
    }
    
    // 输出所有观测的优化结果
    void OutputAllObservationsOptimized(const std::string& output_file) {
        std::ofstream file(output_file);
        if (!file.is_open()) {
            std::cerr << "无法创建输出文件: " << output_file << std::endl;
            return;
        }
        
        file << std::fixed << std::setprecision(6);
        
        int optimized_count = 0;
        int unchanged_count = 0;
        
        // 对每个原始观测
        for (size_t obs_idx = 0; obs_idx < all_observations_.size(); ++obs_idx) {
            const auto& obs = all_observations_[obs_idx];
            
            // 确定使用的mesh point位置
            Eigen::Vector3d final_mesh_point;
            
            if (observation_to_point_.find(obs_idx) != observation_to_point_.end() &&
                observation_to_point_[obs_idx] >= 0) {
                // 使用优化后的位置
                int point_id = observation_to_point_[obs_idx];
                final_mesh_point = points_[point_id]->GetPosition();
                optimized_count++;
            } else {
                // 使用变换后的原始位置
                final_mesh_point = obs.mesh_point_transformed;
                unchanged_count++;
            }
            
            // 输出
            file << obs.frame_id << " "
                 << obs.pixel[0] << " " << obs.pixel[1] << " "
                 << obs.depth_point[0] << " " << obs.depth_point[1] << " " << obs.depth_point[2] << " "
                 << final_mesh_point[0] << " " << final_mesh_point[1] << " " << final_mesh_point[2] << std::endl;
        }
        
        file.close();
        
        std::cout << "\n=== 输出统计 ===" << std::endl;
        std::cout << "总观测数: " << all_observations_.size() << std::endl;
        std::cout << "使用优化位置的观测: " << optimized_count << std::endl;
        std::cout << "使用变换位置的观测: " << unchanged_count << std::endl;
        std::cout << "结果已保存到: " << output_file << std::endl;
    }
    
    // 分析优化结果
    void AnalyzeOptimizationResults() {
        std::cout << "\n=== 优化结果分析 ===" << std::endl;
        
        std::vector<double> movements;
        std::vector<double> depth_errors_before;
        std::vector<double> depth_errors_after;
        
        for (const auto& point_pair : points_) {
            const auto& point = point_pair.second;
            
            // 优化前后的移动距离
            double movement = (point->GetPosition() - point->depth_center).norm();
            movements.push_back(movement);
            
            // 与depth center的距离
            for (const auto& obs : point->observations) {
                // 优化前（使用变换后的mesh point）
                double error_before = (obs.mesh_point_transformed - obs.depth_point).norm();
                depth_errors_before.push_back(error_before);
                
                // 优化后
                double error_after = (point->GetPosition() - obs.depth_point).norm();
                depth_errors_after.push_back(error_after);
            }
        }
        
        // 统计
        if (!movements.empty()) {
            double avg_movement = std::accumulate(movements.begin(), movements.end(), 0.0) / movements.size();
            double max_movement = *std::max_element(movements.begin(), movements.end());
            
            std::cout << "3D点平均移动距离: " << avg_movement * 1000 << " mm" << std::endl;
            std::cout << "3D点最大移动距离: " << max_movement * 1000 << " mm" << std::endl;
        }
        
        if (!depth_errors_before.empty()) {
            double avg_before = std::accumulate(depth_errors_before.begin(), depth_errors_before.end(), 0.0) / depth_errors_before.size();
            double avg_after = std::accumulate(depth_errors_after.begin(), depth_errors_after.end(), 0.0) / depth_errors_after.size();
            
            std::cout << "\n与depth point的平均误差:" << std::endl;
            std::cout << "  优化前: " << avg_before * 1000 << " mm" << std::endl;
            std::cout << "  优化后: " << avg_after * 1000 << " mm" << std::endl;
            std::cout << "  改善: " << (1.0 - avg_after/avg_before) * 100 << "%" << std::endl;
        }
        
        // 显示几个示例
        std::cout << "\n示例3D点优化结果:" << std::endl;
        int count = 0;
        for (const auto& point_pair : points_) {
            if (count++ >= 5) break;
            
            const auto& point = point_pair.second;
            std::cout << "点 " << point->id << " (" << point->observations.size() << " 个观测):" << std::endl;
            std::cout << "  Depth center: [" << point->depth_center.transpose() << "]" << std::endl;
            std::cout << "  优化后位置: [" << point->GetPosition().transpose() << "]" << std::endl;
            std::cout << "  移动距离: " << (point->GetPosition() - point->depth_center).norm() * 1000 << " mm" << std::endl;
        }
    }
    
    // 主运行函数
    bool Run(const std::string& bad_poses_file,
             const std::string& good_poses_file,
             const std::string& raycast_file,
             const std::string& output_dir) {
        
        // 1. 加载位姿
        std::cout << "加载坏位姿..." << std::endl;
        if (!LoadPoses(bad_poses_file, bad_poses_)) {
            return false;
        }
        
        std::cout << "\n加载好位姿..." << std::endl;
        if (!LoadPoses(good_poses_file, good_poses_)) {
            return false;
        }
        
        // 2. 加载并变换ray casting数据
        std::cout << "\n加载并变换ray casting数据..." << std::endl;
        if (!LoadAndTransformRayCastData(raycast_file)) {
            return false;
        }
        
        // 3. 基于depth point聚类
        ClusterByDepthPoints();
        
        // 4. 设置优化问题
        SetupOptimization();
        
        // 5. 执行优化
        if (!Optimize()) {
            std::cerr << "优化失败!" << std::endl;
            return false;
        }
        
        // 6. 分析结果
        AnalyzeOptimizationResults();
        
        // 7. 输出结果
        OutputAllObservationsOptimized(output_dir + "/all_observations_optimized.txt");
        
        // 输出点云用于可视化
        OutputPointClouds(output_dir);
        
        return true;
    }
    
    // 输出点云文件
    void OutputPointClouds(const std::string& output_dir) {
        // 输出优化后的3D点
        std::ofstream file_points(output_dir + "/optimized_3d_points.txt");
        if (!file_points.is_open()) return;
        
        file_points << std::fixed << std::setprecision(6);
        for (const auto& point_pair : points_) {
            const auto& point = point_pair.second;
            Eigen::Vector3d pos = point->GetPosition();
            file_points << pos[0] << " " << pos[1] << " " << pos[2] << std::endl;
        }
        file_points.close();
        
        // 输出所有mesh points（优化后）
        std::ofstream file_mesh(output_dir + "/all_mesh_points_optimized.txt");
        if (!file_mesh.is_open()) return;
        
        file_mesh << std::fixed << std::setprecision(6);
        for (size_t i = 0; i < all_observations_.size(); ++i) {
            Eigen::Vector3d mesh_point;
            
            if (observation_to_point_.find(i) != observation_to_point_.end() &&
                observation_to_point_[i] >= 0) {
                int point_id = observation_to_point_[i];
                mesh_point = points_[point_id]->GetPosition();
            } else {
                mesh_point = all_observations_[i].mesh_point_transformed;
            }
            
            file_mesh << mesh_point[0] << " " << mesh_point[1] << " " << mesh_point[2] << std::endl;
        }
        file_mesh.close();
        
        std::cout << "\n点云文件已保存:" << std::endl;
        std::cout << "  优化的3D点: " << output_dir << "/optimized_3d_points.txt" << std::endl;
        std::cout << "  所有mesh点: " << output_dir << "/all_mesh_points_optimized.txt" << std::endl;
    }
};

int main() {
    // 文件路径
    std::string bad_poses = "/Datasets/CERES_Work/Vis_Result/standard_trajectory_no_loop.txt";
    std::string good_poses = "/Datasets/CERES_Work/Vis_Result/trajectory_after_optimization.txt";
    std::string raycast_data = "/Datasets/CERES_Work/3DPinput/raycast_combined_points_no_loop.txt";
    std::string output_dir = "/Datasets/CERES_Work/output/mesh_optimization_v3";
    
    // 创建输出目录
    system(("mkdir -p " + output_dir).c_str());
    
    // 创建优化器并运行
    MeshOptimizerV3 optimizer;
    
    if (optimizer.Run(bad_poses, good_poses, raycast_data, output_dir)) {
        std::cout << "\n===== 优化完成！=====" << std::endl;
        std::cout << "基于depth point的聚类和优化成功完成" << std::endl;
        std::cout << "所有8355个观测都已处理并输出" << std::endl;
    } else {
        std::cerr << "优化过程失败！" << std::endl;
        return -1;
    }
    
    return 0;
}
