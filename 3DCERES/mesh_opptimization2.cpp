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

// Camera intrinsics
const double FX = 377.535257164;
const double FY = 377.209841379;
const double CX = 328.193371286;
const double CY = 240.426878936;

// SE3 Lie algebra parameterization
class SE3Parameterization : public ceres::Manifold {
public:
    ~SE3Parameterization() {}
    
    int AmbientSize() const override { return 7; }
    int TangentSize() const override { return 6; }
    
    bool Plus(const double* x, const double* delta, double* x_plus_delta) const override {
        // Extract current state
        Eigen::Vector3d t_current(x[0], x[1], x[2]);
        Eigen::Quaterniond q_current(x[6], x[3], x[4], x[5]);
        q_current.normalize();
        
        // Extract Lie algebra increment
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
        
        // Compute V matrix for translation update
        Eigen::Matrix3d V;
        if (theta < eps) {
            V = I + 0.5 * Omega + (1.0/6.0) * Omega * Omega;
        } else {
            double c = cos(theta);
            double s = sin(theta);
            V = I + ((1.0 - c) / (theta * theta)) * Omega + 
                ((theta - s) / (theta * theta * theta)) * Omega * Omega;
        }
        
        // Apply increment
        Eigen::Matrix3d R_current = q_current.toRotationMatrix();
        Eigen::Matrix3d R_new = R * R_current;
        Eigen::Vector3d t_new = R * t_current + V * upsilon;
        
        // Convert back to quaternion
        Eigen::Quaterniond q_new(R_new);
        q_new.normalize();
        
        // Output new state
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
        // Numerical differentiation
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
        // Extract poses
        Eigen::Vector3d t_x(x[0], x[1], x[2]);
        Eigen::Vector3d t_y(y[0], y[1], y[2]);
        Eigen::Quaterniond q_x(x[6], x[3], x[4], x[5]);
        Eigen::Quaterniond q_y(y[6], y[3], y[4], y[5]);
        
        q_x.normalize();
        q_y.normalize();
        
        // Compute relative transform
        Eigen::Matrix3d R_x = q_x.toRotationMatrix();
        Eigen::Matrix3d R_y = q_y.toRotationMatrix();
        Eigen::Matrix3d R_rel = R_x.transpose() * R_y;
        Eigen::Vector3d t_rel = R_x.transpose() * (t_y - t_x);
        
        // Compute rotation log mapping
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
        
        // Compute translation log mapping
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
        
        // Set output
        y_minus_x[0] = omega(0);
        y_minus_x[1] = omega(1);
        y_minus_x[2] = omega(2);
        y_minus_x[3] = upsilon(0);
        y_minus_x[4] = upsilon(1);
        y_minus_x[5] = upsilon(2);
        
        return true;
    }
    
    bool MinusJacobian(const double* x, double* jacobian) const override {
        // Numerical differentiation
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

// Ray casting observation data
struct RayCastObservation {
    int frame_id;
    Eigen::Vector2d pixel;
    Eigen::Vector3d depth_point;      // Reliable depth-projected point
    Eigen::Vector3d mesh_point;        // Original mesh point
    Eigen::Vector3d mesh_point_transformed;  // Transformed mesh point
};

// 3D point structure
struct Point3D {
    int id;
    std::vector<double> position;  // [X, Y, Z] - optimization variables
    std::vector<RayCastObservation> observations;
    Eigen::Vector3d depth_center;  // Center of depth points (strong constraint)
    
    Point3D() : position(3, 0.0) {}
    
    void SetPosition(const Eigen::Vector3d& pos) {
        position[0] = pos[0];
        position[1] = pos[1];
        position[2] = pos[2];
    }
    
    Eigen::Vector3d GetPosition() const {
        return Eigen::Vector3d(position[0], position[1], position[2]);
    }
    
    // Compute center of depth points
    void ComputeDepthCenter() {
        depth_center = Eigen::Vector3d::Zero();
        if (observations.empty()) return;
        
        for (const auto& obs : observations) {
            depth_center += obs.depth_point;
        }
        depth_center /= observations.size();
    }
};

// Camera pose
struct CameraPose {
    int frame_id;
    double timestamp;
    std::vector<double> se3_state;  // [tx, ty, tz, qx, qy, qz, qw]
    
    CameraPose() : se3_state(7, 0.0) {
        se3_state[6] = 1.0;  // qw = 1
    }
    
    void SetFromTUM(double tx, double ty, double tz, 
                    double qx, double qy, double qz, double qw) {
        // TUM format is Twc
        Eigen::Vector3d t_wc(tx, ty, tz);
        Eigen::Quaterniond q_wc(qw, qx, qy, qz);
        q_wc.normalize();
        
        // Convert to Tcw for storage
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

// Reprojection error cost function
class ReprojectionCost {
public:
    ReprojectionCost(const Eigen::Vector2d& observation)
        : observed_pixel_(observation) {}
    
    template <typename T>
    bool operator()(const T* const camera_pose,
                    const T* const point_3d,
                    T* residuals) const {
        // Extract camera pose (Tcw)
        Eigen::Matrix<T, 3, 1> t_cw(camera_pose[0], camera_pose[1], camera_pose[2]);
        Eigen::Quaternion<T> q_cw(camera_pose[6], camera_pose[3], camera_pose[4], camera_pose[5]);
        
        // 3D point (world coordinates)
        Eigen::Matrix<T, 3, 1> P_w(point_3d[0], point_3d[1], point_3d[2]);
        
        // Transform to camera coordinates
        Eigen::Matrix<T, 3, 1> P_c = q_cw * P_w + t_cw;
        
        // Check if point is in front of camera
        if (P_c[2] <= T(0.01)) {  // At least 1cm in front
            residuals[0] = T(1000.0);
            residuals[1] = T(1000.0);
            return true;
        }
        
        // Project to pixel plane
        T u = T(FX) * (P_c[0] / P_c[2]) + T(CX);
        T v = T(FY) * (P_c[1] / P_c[2]) + T(CY);
        
        // Compute residuals
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

// Depth consistency constraint
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

// Mesh Optimizer V3
class MeshOptimizerV3 {
private:
    std::map<int, CameraPose> bad_poses_;
    std::map<int, CameraPose> good_poses_;
    std::vector<RayCastObservation> all_observations_;
    std::map<int, std::shared_ptr<Point3D>> points_;
    std::unique_ptr<ceres::Problem> problem_;
    
    // Observation index to 3D point ID mapping
    std::map<int, int> observation_to_point_;
    
    // Depth point-based clustering
    struct DepthCluster {
        std::vector<int> observation_indices;
        Eigen::Vector3d depth_center;
        Eigen::Vector3d mesh_center;  // Center of transformed mesh points
    };
    
public:
    MeshOptimizerV3() : problem_(std::make_unique<ceres::Problem>()) {}
    
    // Load poses
    bool LoadPoses(const std::string& pose_file, std::map<int, CameraPose>& poses) {
        std::ifstream file(pose_file);
        if (!file.is_open()) {
            std::cerr << "Cannot open pose file: " << pose_file << std::endl;
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
        
        std::cout << "Loaded " << poses.size() << " poses" << std::endl;
        return true;
    }
    
    // Load ray casting data and perform per-frame transformation
    bool LoadAndTransformRayCastData(const std::string& raycast_file) {
        std::ifstream file(raycast_file);
        if (!file.is_open()) {
            std::cerr << "Cannot open ray casting file: " << raycast_file << std::endl;
            return false;
        }
        
        std::string line;
        all_observations_.clear();
        observation_to_point_.clear();
        
        // Read and transform all observations
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
                
                // Compute per-frame transformation
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
        
        std::cout << "Read " << all_observations_.size() << " observations" << std::endl;
        
        // Initialize mapping
        for (size_t i = 0; i < all_observations_.size(); ++i) {
            observation_to_point_[i] = -1;
        }
        
        return true;
    }
    
    // Cluster by depth points
    void ClusterByDepthPoints() {
        std::cout << "\nClustering by depth points..." << std::endl;
        
        const double DEPTH_CLUSTER_THRESHOLD = 0.02;  // 2cm - depth points are very accurate
        
        std::vector<DepthCluster> clusters;
        std::vector<bool> processed(all_observations_.size(), false);
        
        // For each unprocessed observation
        for (size_t i = 0; i < all_observations_.size(); ++i) {
            if (processed[i]) continue;
            
            // Create new cluster
            DepthCluster cluster;
            cluster.observation_indices.push_back(i);
            cluster.depth_center = all_observations_[i].depth_point;
            cluster.mesh_center = all_observations_[i].mesh_point_transformed;
            processed[i] = true;
            
            // Find all nearby depth points
            for (size_t j = i + 1; j < all_observations_.size(); ++j) {
                if (processed[j]) continue;
                
                double dist = (all_observations_[j].depth_point - cluster.depth_center).norm();
                if (dist < DEPTH_CLUSTER_THRESHOLD) {
                    // Update cluster center (incremental average)
                    int n = cluster.observation_indices.size();
                    cluster.depth_center = (cluster.depth_center * n + all_observations_[j].depth_point) / (n + 1);
                    cluster.mesh_center = (cluster.mesh_center * n + all_observations_[j].mesh_point_transformed) / (n + 1);
                    
                    cluster.observation_indices.push_back(j);
                    processed[j] = true;
                }
            }
            
            clusters.push_back(cluster);
        }
        
        std::cout << "Initial clusters: " << clusters.size() << std::endl;
        
        // Filter single-view observations
        std::vector<DepthCluster> filtered_clusters;
        for (const auto& cluster : clusters) {
            if (cluster.observation_indices.size() >= 2) {
                filtered_clusters.push_back(cluster);
            }
        }
        
        std::cout << "Filtered clusters: " << filtered_clusters.size() << std::endl;
        
        // Create 3D points
        points_.clear();
        for (size_t i = 0; i < filtered_clusters.size(); ++i) {
            auto point = std::make_shared<Point3D>();
            point->id = i;
            
            // Collect observations
            for (int obs_idx : filtered_clusters[i].observation_indices) {
                point->observations.push_back(all_observations_[obs_idx]);
                observation_to_point_[obs_idx] = i;
            }
            
            // Compute depth center
            point->ComputeDepthCenter();
            
            // Use depth center as initial value (excellent initial guess!)
            point->SetPosition(point->depth_center);
            
            points_[i] = point;
        }
        
        // Statistics
        int total_observations = 0;
        int max_observations = 0;
        int min_observations = INT_MAX;
        
        for (const auto& p : points_) {
            int num_obs = p.second->observations.size();
            total_observations += num_obs;
            max_observations = std::max(max_observations, num_obs);
            min_observations = std::min(min_observations, num_obs);
        }
        
        std::cout << "Created " << points_.size() << " 3D points" << std::endl;
        std::cout << "Covering " << total_observations << " observations" << std::endl;
        std::cout << "Observation count range: [" << min_observations << ", " << max_observations << "]" << std::endl;
        std::cout << "Average observations per point: " << (double)total_observations / points_.size() << std::endl;
    }
    
    // Setup optimization problem
    void SetupOptimization() {
        std::cout << "\nSetting up optimization problem..." << std::endl;
        
        // 1. Add camera pose parameters (using good poses, fixed)
        for (auto& pose_pair : good_poses_) {
            auto& pose = pose_pair.second;
            problem_->AddParameterBlock(pose.se3_state.data(), 7);
            problem_->SetManifold(pose.se3_state.data(), new SE3Parameterization());
            problem_->SetParameterBlockConstant(pose.se3_state.data());
        }
        
        std::cout << "Added " << good_poses_.size() << " camera poses (all fixed)" << std::endl;
        
        // 2. Add 3D point parameters and constraints
        int reproj_constraints = 0;
        int depth_constraints = 0;
        
        for (auto& point_pair : points_) {
            auto& point = point_pair.second;
            
            // Add 3D point parameter block
            problem_->AddParameterBlock(point->position.data(), 3);
            
            // Add depth consistency constraint (high weight because depth is accurate)
            ceres::CostFunction* depth_cost = 
                DepthConsistencyCost::Create(point->depth_center, 10.0);  // Weight 10
            problem_->AddResidualBlock(depth_cost, nullptr, point->position.data());
            depth_constraints++;
            
            // Add reprojection constraint for each observation
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
        
        std::cout << "Added " << points_.size() << " 3D point parameters" << std::endl;
        std::cout << "Added " << depth_constraints << " depth consistency constraints" << std::endl;
        std::cout << "Added " << reproj_constraints << " reprojection constraints" << std::endl;
    }
    
    // Execute optimization
    bool Optimize() {
        std::cout << "\nStarting optimization..." << std::endl;
        
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
    
    // Output all observations with optimized results
    void OutputAllObservationsOptimized(const std::string& output_file) {
        std::ofstream file(output_file);
        if (!file.is_open()) {
            std::cerr << "Cannot create output file: " << output_file << std::endl;
            return;
        }
        
        file << std::fixed << std::setprecision(6);
        
        int optimized_count = 0;
        int unchanged_count = 0;
        
        // For each original observation
        for (size_t obs_idx = 0; obs_idx < all_observations_.size(); ++obs_idx) {
            const auto& obs = all_observations_[obs_idx];
            
            // Determine which mesh point position to use
            Eigen::Vector3d final_mesh_point;
            
            if (observation_to_point_.find(obs_idx) != observation_to_point_.end() &&
                observation_to_point_[obs_idx] >= 0) {
                // Use optimized position
                int point_id = observation_to_point_[obs_idx];
                final_mesh_point = points_[point_id]->GetPosition();
                optimized_count++;
            } else {
                // Use transformed original position
                final_mesh_point = obs.mesh_point_transformed;
                unchanged_count++;
            }
            
            // Output
            file << obs.frame_id << " "
                 << obs.pixel[0] << " " << obs.pixel[1] << " "
                 << obs.depth_point[0] << " " << obs.depth_point[1] << " " << obs.depth_point[2] << " "
                 << final_mesh_point[0] << " " << final_mesh_point[1] << " " << final_mesh_point[2] << std::endl;
        }
        
        file.close();
        
        std::cout << "\n=== Output Statistics ===" << std::endl;
        std::cout << "Total observations: " << all_observations_.size() << std::endl;
        std::cout << "Observations using optimized positions: " << optimized_count << std::endl;
        std::cout << "Observations using transformed positions: " << unchanged_count << std::endl;
        std::cout << "Results saved to: " << output_file << std::endl;
    }
    
    // Analyze optimization results
    void AnalyzeOptimizationResults() {
        std::cout << "\n=== Optimization Results Analysis ===" << std::endl;
        
        std::vector<double> movements;
        std::vector<double> depth_errors_before;
        std::vector<double> depth_errors_after;
        
        for (const auto& point_pair : points_) {
            const auto& point = point_pair.second;
            
            // Movement distance before and after optimization
            double movement = (point->GetPosition() - point->depth_center).norm();
            movements.push_back(movement);
            
            // Distance to depth center
            for (const auto& obs : point->observations) {
                // Before optimization (using transformed mesh point)
                double error_before = (obs.mesh_point_transformed - obs.depth_point).norm();
                depth_errors_before.push_back(error_before);
                
                // After optimization
                double error_after = (point->GetPosition() - obs.depth_point).norm();
                depth_errors_after.push_back(error_after);
            }
        }
        
        // Statistics
        if (!movements.empty()) {
            double avg_movement = std::accumulate(movements.begin(), movements.end(), 0.0) / movements.size();
            double max_movement = *std::max_element(movements.begin(), movements.end());
            
            std::cout << "Average 3D point movement: " << avg_movement * 1000 << " mm" << std::endl;
            std::cout << "Maximum 3D point movement: " << max_movement * 1000 << " mm" << std::endl;
        }
        
        if (!depth_errors_before.empty()) {
            double avg_before = std::accumulate(depth_errors_before.begin(), depth_errors_before.end(), 0.0) / depth_errors_before.size();
            double avg_after = std::accumulate(depth_errors_after.begin(), depth_errors_after.end(), 0.0) / depth_errors_after.size();
            
            std::cout << "\nAverage error to depth points:" << std::endl;
            std::cout << "  Before optimization: " << avg_before * 1000 << " mm" << std::endl;
            std::cout << "  After optimization: " << avg_after * 1000 << " mm" << std::endl;
            std::cout << "  Improvement: " << (1.0 - avg_after/avg_before) * 100 << "%" << std::endl;
        }
        
        // Show sample results
        std::cout << "\nSample 3D point optimization results:" << std::endl;
        int count = 0;
        for (const auto& point_pair : points_) {
            if (count++ >= 5) break;
            
            const auto& point = point_pair.second;
            std::cout << "Point " << point->id << " (" << point->observations.size() << " observations):" << std::endl;
            std::cout << "  Depth center: [" << point->depth_center.transpose() << "]" << std::endl;
            std::cout << "  Optimized position: [" << point->GetPosition().transpose() << "]" << std::endl;
            std::cout << "  Movement: " << (point->GetPosition() - point->depth_center).norm() * 1000 << " mm" << std::endl;
        }
    }
    
    // Main run function
    bool Run(const std::string& bad_poses_file,
             const std::string& good_poses_file,
             const std::string& raycast_file,
             const std::string& output_dir) {
        
        // 1. Load poses
        std::cout << "Loading bad poses..." << std::endl;
        if (!LoadPoses(bad_poses_file, bad_poses_)) {
            return false;
        }
        
        std::cout << "\nLoading good poses..." << std::endl;
        if (!LoadPoses(good_poses_file, good_poses_)) {
            return false;
        }
        
        // 2. Load and transform ray casting data
        std::cout << "\nLoading and transforming ray casting data..." << std::endl;
        if (!LoadAndTransformRayCastData(raycast_file)) {
            return false;
        }
        
        // 3. Cluster by depth points
        ClusterByDepthPoints();
        
        // 4. Setup optimization problem
        SetupOptimization();
        
        // 5. Execute optimization
        if (!Optimize()) {
            std::cerr << "Optimization failed!" << std::endl;
            return false;
        }
        
        // 6. Analyze results
        AnalyzeOptimizationResults();
        
        // 7. Output results
        OutputAllObservationsOptimized(output_dir + "/all_observations_optimized.txt");
        
        // Output point clouds for visualization
        OutputPointClouds(output_dir);
        
        return true;
    }
    
    // Output point cloud files
    void OutputPointClouds(const std::string& output_dir) {
        // Output optimized 3D points
        std::ofstream file_points(output_dir + "/optimized_3d_points.txt");
        if (!file_points.is_open()) return;
        
        file_points << std::fixed << std::setprecision(6);
        for (const auto& point_pair : points_) {
            const auto& point = point_pair.second;
            Eigen::Vector3d pos = point->GetPosition();
            file_points << pos[0] << " " << pos[1] << " " << pos[2] << std::endl;
        }
        file_points.close();
        
        // Output all mesh points (optimized)
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
        
        std::cout << "\nPoint cloud files saved:" << std::endl;
        std::cout << "  Optimized 3D points: " << output_dir << "/optimized_3d_points.txt" << std::endl;
        std::cout << "  All mesh points: " << output_dir << "/all_mesh_points_optimized.txt" << std::endl;
    }
};

int main() {
    // File paths
    std::string bad_poses = "/Datasets/CERES_Work/Vis_Result/standard_trajectory_no_loop.txt";
    std::string good_poses = "/Datasets/CERES_Work/Vis_Result/trajectory_after_optimization.txt";
    std::string raycast_data = "/Datasets/CERES_Work/3DPinput/raycast_combined_points_no_loop.txt";
    std::string output_dir = "/Datasets/CERES_Work/output/mesh_optimization_v3";
    
    // Create output directory
    system(("mkdir -p " + output_dir).c_str());
    
    // Create optimizer and run
    MeshOptimizerV3 optimizer;
    
    if (optimizer.Run(bad_poses, good_poses, raycast_data, output_dir)) {
        std::cout << "\n===== Optimization Complete! =====" << std::endl;
        std::cout << "Depth point-based clustering and optimization successfully completed" << std::endl;
        std::cout << "All 8355 observations have been processed and output" << std::endl;
    } else {
        std::cerr << "Optimization process failed!" << std::endl;
        return -1;
    }
    
    return 0;
}
