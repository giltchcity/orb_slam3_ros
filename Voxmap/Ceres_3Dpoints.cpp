#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <memory>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

// Camera intrinsics
const double FX = 377.535257164;
const double FY = 377.209841379;
const double CX = 328.193371286;
const double CY = 240.426878936;

// Camera pose
struct CameraPose {
    int frame_id;
    double timestamp;
    Eigen::Matrix4d T_wc;  // World to Camera
    Eigen::Matrix4d T_cw;  // Camera to World
    std::vector<double> pose_data;  // For Ceres optimization [tx, ty, tz, qx, qy, qz, qw]
    
    CameraPose() : pose_data(7, 0.0) {
        pose_data[6] = 1.0;  // qw = 1
    }
    
    void SetFromTUM(double tx, double ty, double tz, 
                    double qx, double qy, double qz, double qw) {
        // TUM format is T_wc
        Eigen::Vector3d t_wc(tx, ty, tz);
        Eigen::Quaterniond q_wc(qw, qx, qy, qz);
        q_wc.normalize();
        
        T_wc = Eigen::Matrix4d::Identity();
        T_wc.block<3, 3>(0, 0) = q_wc.toRotationMatrix();
        T_wc.block<3, 1>(0, 3) = t_wc;
        
        T_cw = T_wc.inverse();
        
        // Also fill pose_data (T_cw format) for Ceres
        Eigen::Vector3d t_cw = T_cw.block<3, 1>(0, 3);
        Eigen::Matrix3d R_cw = T_cw.block<3, 3>(0, 0);
        Eigen::Quaterniond q_cw(R_cw);
        
        pose_data[0] = t_cw[0];
        pose_data[1] = t_cw[1];
        pose_data[2] = t_cw[2];
        pose_data[3] = q_cw.x();
        pose_data[4] = q_cw.y();
        pose_data[5] = q_cw.z();
        pose_data[6] = q_cw.w();
    }
};

// Voxblox observation data
struct VoxbloxObservation {
    int kf_idx;
    Eigen::Vector2d pixel;
    double depth_measured;
    Eigen::Vector3d P_cam;  // 3D position in camera coordinate system
    double weight;          // Observation weight
    double confidence;      // Confidence
};

// 3D point
struct Point3D {
    int id;
    Eigen::Vector3d position_original;     // Original position (world coordinates under bad pose)
    Eigen::Vector3d position_transformed;  // Position after per-frame transform
    std::vector<double> position_optimized;  // Ceres optimized position [x, y, z]
    std::vector<VoxbloxObservation> observations;
    
    Point3D() : position_optimized(3, 0.0) {}
    
    void SetOptimizedPosition(const Eigen::Vector3d& pos) {
        position_optimized[0] = pos[0];
        position_optimized[1] = pos[1];
        position_optimized[2] = pos[2];
    }
    
    Eigen::Vector3d GetOptimizedPosition() const {
        return Eigen::Vector3d(position_optimized[0], position_optimized[1], position_optimized[2]);
    }
};

// Reprojection error cost function
class ReprojectionCost {
public:
    ReprojectionCost(const Eigen::Vector2d& observation, double weight = 1.0)
        : observed_pixel_(observation), weight_(sqrt(weight)) {}
    
    template <typename T>
    bool operator()(const T* const camera_pose,
                    const T* const point_3d,
                    T* residuals) const {
        // Extract camera pose (T_cw)
        Eigen::Matrix<T, 3, 1> t_cw(camera_pose[0], camera_pose[1], camera_pose[2]);
        Eigen::Quaternion<T> q_cw(camera_pose[6], camera_pose[3], camera_pose[4], camera_pose[5]);
        
        // 3D point (world coordinate system)
        Eigen::Matrix<T, 3, 1> P_w(point_3d[0], point_3d[1], point_3d[2]);
        
        // Transform to camera coordinate system
        Eigen::Matrix<T, 3, 1> P_c = q_cw * P_w + t_cw;
        
        // Check if point is in front of camera
        if (P_c[2] <= T(0.01)) {
            residuals[0] = T(100.0) * T(weight_);
            residuals[1] = T(100.0) * T(weight_);
            return true;
        }
        
        // Project to pixel plane
        T u = T(FX) * (P_c[0] / P_c[2]) + T(CX);
        T v = T(FY) * (P_c[1] / P_c[2]) + T(CY);
        
        // Calculate weighted residuals
        residuals[0] = T(weight_) * (u - T(observed_pixel_[0]));
        residuals[1] = T(weight_) * (v - T(observed_pixel_[1]));
        
        return true;
    }
    
    static ceres::CostFunction* Create(const Eigen::Vector2d& observation, double weight = 1.0) {
        return new ceres::AutoDiffCostFunction<ReprojectionCost, 2, 7, 3>(
            new ReprojectionCost(observation, weight));
    }
    
private:
    Eigen::Vector2d observed_pixel_;
    double weight_;
};

// Depth consistency cost function
class DepthConsistencyCost {
public:
    DepthConsistencyCost(const Eigen::Vector3d& P_cam_target, double depth_weight = 10.0)
        : P_cam_target_(P_cam_target), depth_weight_(depth_weight) {}
    
    template <typename T>
    bool operator()(const T* const camera_pose,
                    const T* const point_3d,
                    T* residuals) const {
        // Extract camera pose (T_cw)
        Eigen::Matrix<T, 3, 1> t_cw(camera_pose[0], camera_pose[1], camera_pose[2]);
        Eigen::Quaternion<T> q_cw(camera_pose[6], camera_pose[3], camera_pose[4], camera_pose[5]);
        
        // 3D point (world coordinate system)
        Eigen::Matrix<T, 3, 1> P_w(point_3d[0], point_3d[1], point_3d[2]);
        
        // Transform to camera coordinate system
        Eigen::Matrix<T, 3, 1> P_c = q_cw * P_w + t_cw;
        
        // Only constrain depth (z direction), x and y are constrained by reprojection
        residuals[0] = T(depth_weight_) * (P_c[2] - T(P_cam_target_[2]));
        
        return true;
    }
    
    static ceres::CostFunction* Create(const Eigen::Vector3d& P_cam_target, double depth_weight = 10.0) {
        return new ceres::AutoDiffCostFunction<DepthConsistencyCost, 1, 7, 3>(
            new DepthConsistencyCost(P_cam_target, depth_weight));
    }
    
private:
    Eigen::Vector3d P_cam_target_;
    double depth_weight_;
};

// Initial position regularizer (prevent points from drifting too far)
class InitialPositionRegularizer {
public:
    InitialPositionRegularizer(const Eigen::Vector3d& initial_pos, double weight = 0.1)
        : initial_position_(initial_pos), weight_(weight) {}
    
    template <typename T>
    bool operator()(const T* const point_3d, T* residuals) const {
        residuals[0] = T(weight_) * (point_3d[0] - T(initial_position_[0]));
        residuals[1] = T(weight_) * (point_3d[1] - T(initial_position_[1]));
        residuals[2] = T(weight_) * (point_3d[2] - T(initial_position_[2]));
        return true;
    }
    
    static ceres::CostFunction* Create(const Eigen::Vector3d& initial_pos, double weight = 0.1) {
        return new ceres::AutoDiffCostFunction<InitialPositionRegularizer, 3, 3>(
            new InitialPositionRegularizer(initial_pos, weight));
    }
    
private:
    Eigen::Vector3d initial_position_;
    double weight_;
};

// Mesh transformer and optimizer class
class MeshTransformerOptimizer {
private:
    std::map<int, CameraPose> poses_before_;  // Poses before loop closure
    std::map<int, CameraPose> poses_after_;   // Poses after loop closure
    std::map<int, std::shared_ptr<Point3D>> points_;
    std::unique_ptr<ceres::Problem> problem_;
    
public:
    MeshTransformerOptimizer() : problem_(std::make_unique<ceres::Problem>()) {}
    
    // Load pose file
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
    
    // Load Voxblox correspondence data
    bool LoadVoxbloxData(const std::string& data_file) {
        std::ifstream file(data_file);
        if (!file.is_open()) {
            std::cerr << "Cannot open data file: " << data_file << std::endl;
            return false;
        }
        
        std::string line;
        points_.clear();
        
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;
            
            // Check if this is a point definition line
            if (line[0] != ' ' && line[0] != '\t') {
                std::istringstream iss(line);
                int point_id, num_obs;
                double x, y, z;
                
                if (iss >> point_id >> x >> y >> z >> num_obs) {
                    auto point = std::make_shared<Point3D>();
                    point->id = point_id;
                    point->position_original = Eigen::Vector3d(x, y, z);
                    
                    // Read observation data
                    for (int i = 0; i < num_obs; ++i) {
                        if (std::getline(file, line)) {
                            // Remove leading spaces
                            size_t first = line.find_first_not_of(" \t");
                            if (first != std::string::npos) {
                                line = line.substr(first);
                            }
                            
                            std::istringstream obs_iss(line);
                            VoxbloxObservation obs;
                            double depth_proj;
                            
                            if (obs_iss >> obs.kf_idx >> obs.pixel[0] >> obs.pixel[1] 
                                >> obs.depth_measured >> depth_proj >> obs.weight >> obs.confidence) {
                                
                                // Calculate 3D position in camera coordinate system
                                double z = obs.depth_measured;
                                double x = (obs.pixel[0] - CX) * z / FX;
                                double y = (obs.pixel[1] - CY) * z / FY;
                                obs.P_cam = Eigen::Vector3d(x, y, z);
                                
                                point->observations.push_back(obs);
                            }
                        }
                    }
                    
                    if (!point->observations.empty()) {
                        points_[point_id] = point;
                    }
                }
            }
        }
        
        std::cout << "Loaded " << points_.size() << " 3D points" << std::endl;
        return true;
    }
    
    // Step 1: Perform per-frame transform (provide good initial values)
    void PerformPerFrameTransform() {
        std::cout << "\nStep 1: Performing per-frame transform (providing optimization initial values)..." << std::endl;
        
        std::vector<double> movements;
        
        for (auto& point_pair : points_) {
            auto& point = point_pair.second;
            
            // Use the first observation's KF for transformation
            for (const auto& obs : point->observations) {
                int kf_id = obs.kf_idx;
                
                if (poses_before_.find(kf_id) != poses_before_.end() && 
                    poses_after_.find(kf_id) != poses_after_.end()) {
                    
                    // Use precise depth to calculate camera coordinates
                    Eigen::Vector3d P_cam = obs.P_cam;
                    
                    // Use new camera pose to calculate world coordinates
                    Eigen::Matrix4d T_wc_after = poses_after_[kf_id].T_wc;
                    Eigen::Vector4d P_cam_homo(P_cam[0], P_cam[1], P_cam[2], 1.0);
                    Eigen::Vector4d P_world_new = T_wc_after * P_cam_homo;
                    
                    point->position_transformed = P_world_new.head<3>();
                    
                    // Set as initial value for optimization
                    point->SetOptimizedPosition(point->position_transformed);
                    
                    double movement = (point->position_transformed - point->position_original).norm();
                    movements.push_back(movement);
                    
                    break;  // Use the first valid observation
                }
            }
        }
        
        if (!movements.empty()) {
            double avg_movement = std::accumulate(movements.begin(), movements.end(), 0.0) / movements.size();
            std::cout << "  Per-frame transform completed, average movement: " << avg_movement << " m" << std::endl;
        }
    }
    
    // Step 2: Setup Ceres optimization problem
    void SetupOptimization() {
        std::cout << "\nStep 2: Setting up Ceres optimization problem..." << std::endl;
        
        // Add camera pose parameters (fixed)
        for (auto& pose_pair : poses_after_) {
            auto& pose = pose_pair.second;
            problem_->AddParameterBlock(pose.pose_data.data(), 7);
            problem_->SetParameterBlockConstant(pose.pose_data.data());
        }
        
        int reproj_constraints = 0;
        int depth_constraints = 0;
        int regularizer_constraints = 0;
        
        // Add constraints for each 3D point
        for (auto& point_pair : points_) {
            auto& point = point_pair.second;
            
            // Add 3D point parameter block
            problem_->AddParameterBlock(point->position_optimized.data(), 3);
            
            // 1. Add soft regularization constraint (prevent points from drifting too far)
            ceres::CostFunction* regularizer_cost = 
                InitialPositionRegularizer::Create(point->position_transformed, 0.1);  // Small weight
            problem_->AddResidualBlock(regularizer_cost, nullptr, point->position_optimized.data());
            regularizer_constraints++;
            
            // 2. Add constraints for each observation
            for (const auto& obs : point->observations) {
                if (poses_after_.find(obs.kf_idx) == poses_after_.end()) continue;
                
                // 2.1 Reprojection error constraint
                double reproj_weight = std::max(0.5, obs.weight);  // Use Voxblox weight
                ceres::CostFunction* reproj_cost = 
                    ReprojectionCost::Create(obs.pixel, reproj_weight);
                ceres::LossFunction* reproj_loss = new ceres::HuberLoss(2.0);  // Robust loss function
                
                problem_->AddResidualBlock(
                    reproj_cost,
                    reproj_loss,
                    poses_after_[obs.kf_idx].pose_data.data(),
                    point->position_optimized.data()
                );
                reproj_constraints++;
                
                // 2.2 Depth consistency constraint (strong constraint)
                double depth_weight = 20.0;  // Depth is very accurate, large weight
                ceres::CostFunction* depth_cost = 
                    DepthConsistencyCost::Create(obs.P_cam, depth_weight);
                
                problem_->AddResidualBlock(
                    depth_cost,
                    nullptr,  // No robust loss for depth, it's a hard constraint
                    poses_after_[obs.kf_idx].pose_data.data(),
                    point->position_optimized.data()
                );
                depth_constraints++;
            }
        }
        
        std::cout << "  Added " << points_.size() << " 3D point parameters" << std::endl;
        std::cout << "  Reprojection constraints: " << reproj_constraints << std::endl;
        std::cout << "  Depth constraints: " << depth_constraints << std::endl;
        std::cout << "  Regularizer constraints: " << regularizer_constraints << std::endl;
    }
    
    // Step 3: Execute optimization
    bool Optimize() {
        std::cout << "\nStep 3: Executing Ceres optimization (fine-tuning)..." << std::endl;
        
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::SPARSE_SCHUR;
        options.minimizer_progress_to_stdout = true;  // Show optimization progress
        options.max_num_iterations = 20;  // Don't need many iterations with good initial values
        options.num_threads = 8;
        options.function_tolerance = 1e-6;
        
        ceres::Solver::Summary summary;
        ceres::Solve(options, problem_.get(), &summary);
        
        std::cout << "\n  Optimization completed: " << summary.BriefReport() << std::endl;
        std::cout << "  Initial cost: " << summary.initial_cost << std::endl;
        std::cout << "  Final cost: " << summary.final_cost << std::endl;
        std::cout << "  Number of iterations: " << summary.iterations.size() << std::endl;
        std::cout << "  Cost reduction: " << (summary.initial_cost - summary.final_cost) 
                  << " (" << (100.0 * (summary.initial_cost - summary.final_cost) / summary.initial_cost) 
                  << "%)" << std::endl;
        
        return summary.IsSolutionUsable();
    }
    
    // Analyze optimization results
    void AnalyzeResults() {
        std::cout << "\n=== Optimization Results Analysis ===" << std::endl;
        
        std::vector<double> transform_to_optimize;  // Movement from per-frame to optimized
        std::vector<double> original_to_optimize;   // Total movement from original to optimized
        std::vector<double> reproj_errors;
        std::vector<double> depth_errors;
        
        for (const auto& point_pair : points_) {
            const auto& point = point_pair.second;
            Eigen::Vector3d pos_opt = point->GetOptimizedPosition();
            
            // Calculate movements
            double move1 = (pos_opt - point->position_transformed).norm();
            double move2 = (pos_opt - point->position_original).norm();
            transform_to_optimize.push_back(move1);
            original_to_optimize.push_back(move2);
            
            // Calculate errors
            for (const auto& obs : point->observations) {
                if (poses_after_.find(obs.kf_idx) == poses_after_.end()) continue;
                
                // Calculate optimized reprojection and depth
                Eigen::Matrix4d T_cw = poses_after_[obs.kf_idx].T_cw;
                Eigen::Vector4d P_w(pos_opt[0], pos_opt[1], pos_opt[2], 1.0);
                Eigen::Vector4d P_c = T_cw * P_w;
                
                // Reprojection error
                double u = FX * (P_c[0] / P_c[2]) + CX;
                double v = FY * (P_c[1] / P_c[2]) + CY;
                double reproj_err = sqrt(pow(u - obs.pixel[0], 2) + pow(v - obs.pixel[1], 2));
                reproj_errors.push_back(reproj_err);
                
                // Depth error
                double depth_err = std::abs(P_c[2] - obs.depth_measured);
                depth_errors.push_back(depth_err);
            }
        }
        
        // Statistics
        if (!transform_to_optimize.empty()) {
            double avg_microadjust = std::accumulate(transform_to_optimize.begin(), 
                                                    transform_to_optimize.end(), 0.0) / transform_to_optimize.size();
            double max_microadjust = *std::max_element(transform_to_optimize.begin(), transform_to_optimize.end());
            
            std::cout << "Optimization fine-tuning (relative to per-frame transform):" << std::endl;
            std::cout << "  Average: " << avg_microadjust * 1000 << " mm" << std::endl;
            std::cout << "  Maximum: " << max_microadjust * 1000 << " mm" << std::endl;
        }
        
        if (!reproj_errors.empty()) {
            double avg_reproj = std::accumulate(reproj_errors.begin(), reproj_errors.end(), 0.0) / reproj_errors.size();
            std::cout << "\nReprojection error:" << std::endl;
            std::cout << "  Average: " << avg_reproj << " pixels" << std::endl;
        }
        
        if (!depth_errors.empty()) {
            double avg_depth = std::accumulate(depth_errors.begin(), depth_errors.end(), 0.0) / depth_errors.size();
            std::cout << "\nDepth error:" << std::endl;
            std::cout << "  Average: " << avg_depth * 1000 << " mm" << std::endl;
        }
    }
    
    // Save results
    void SaveResults(const std::string& output_dir) {
        // Save optimized points
        std::string output_file = output_dir + "/optimized_points_final.txt";
        std::ofstream file(output_file);
        if (!file.is_open()) {
            std::cerr << "Cannot create output file: " << output_file << std::endl;
            return;
        }
        
        file << std::fixed << std::setprecision(6);
        file << "# Final optimized 3D points (per-frame transform + Ceres refinement)\n";
        file << "# Format: point_id x_original y_original z_original x_optimized y_optimized z_optimized movement_mm\n";
        
        for (const auto& point_pair : points_) {
            const auto& point = point_pair.second;
            Eigen::Vector3d pos_opt = point->GetOptimizedPosition();
            double movement = (pos_opt - point->position_original).norm();
            
            file << point->id << " "
                 << point->position_original[0] << " " 
                 << point->position_original[1] << " " 
                 << point->position_original[2] << " "
                 << pos_opt[0] << " " 
                 << pos_opt[1] << " " 
                 << pos_opt[2] << " "
                 << movement * 1000 << std::endl;
        }
        
        file.close();
        std::cout << "\nSaved final optimization results to: " << output_file << std::endl;
    }
    
    // Main run function
    bool Run(const std::string& poses_before_file,
             const std::string& poses_after_file,
             const std::string& voxblox_data_file,
             const std::string& output_dir) {
        
        // 1. Load data
        std::cout << "=== Loading Data ===" << std::endl;
        if (!LoadPoses(poses_before_file, poses_before_)) return false;
        if (!LoadPoses(poses_after_file, poses_after_)) return false;
        if (!LoadVoxbloxData(voxblox_data_file)) return false;
        
        // 2. Per-frame transform (provide good initial values)
        PerformPerFrameTransform();
        
        // 3. Setup optimization problem
        SetupOptimization();
        
        // 4. Execute optimization
        if (!Optimize()) {
            std::cerr << "Optimization failed!" << std::endl;
            return false;
        }
        
        // 5. Analyze results
        AnalyzeResults();
        
        // 6. Save results
        SaveResults(output_dir);
        
        return true;
    }
};

int main() {
    // Set file paths
    std::string base_path = "/Datasets/Voxmap/";
    std::string poses_before = base_path + "standard_trajectory_no_loop.txt";
    std::string poses_after = base_path + "standard_trajectory_with_loop.txt";
    std::string voxblox_data = base_path + "optimization_data.txt";
    std::string output_dir = base_path + "output";
    
    // Create output directory
    system(("mkdir -p " + output_dir).c_str());
    
    // Create optimizer and run
    MeshTransformerOptimizer optimizer;
    
    if (optimizer.Run(poses_before, poses_after, voxblox_data, output_dir)) {
        std::cout << "\n===== Complete! =====" << std::endl;
        std::cout << "1. Per-frame transform provided good initial values (keeping depth unchanged)" << std::endl;
        std::cout << "2. Ceres optimization performed fine-tuning (considering both reprojection and depth)" << std::endl;
        std::cout << "3. Final results saved at: " << output_dir << "/optimized_points_final.txt" << std::endl;
    } else {
        std::cerr << "Processing failed!" << std::endl;
        return -1;
    }
    
    return 0;
}
