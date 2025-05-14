#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <set>
#include <string>
#include <sstream>
#include <algorithm>
#include <iomanip>
#include <ceres/ceres.h>
#include <ceres/manifold.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <sophus/se3.hpp>

// SE3 Manifold for Ceres 2.2
class SE3Manifold : public ceres::Manifold {
public:
    // Fixed scale version (for RGBD/stereo)
    SE3Manifold(bool fix_scale = true) : _fix_scale(fix_scale) {}

    // Ambient space: 7D for quaternion + translation (qx, qy, qz, qw, tx, ty, tz)
    int AmbientSize() const override { return 7; }
    
    // Tangent space: 6D for rotation + translation
    int TangentSize() const override { return 6; }

    // Plus operation: x_plus_delta = [q * exp(delta_rot), t + delta_trans]
    bool Plus(const double* x, const double* delta, double* x_plus_delta) const override {
        // Extract the quaternion and translation from x
        Eigen::Quaterniond q(x[3], x[0], x[1], x[2]); // w, x, y, z
        Eigen::Vector3d t(x[4], x[5], x[6]);

        // Map delta rotation (first 3 elements) to a quaternion using exponential map
        Eigen::Vector3d delta_rot(delta[0], delta[1], delta[2]);
        double norm_delta = delta_rot.norm();
        
        Eigen::Quaterniond delta_q;
        if (norm_delta > 1e-10) {
            double half_angle = norm_delta / 2.0;
            delta_q = Eigen::Quaterniond(
                cos(half_angle),
                delta_rot[0] / norm_delta * sin(half_angle),
                delta_rot[1] / norm_delta * sin(half_angle),
                delta_rot[2] / norm_delta * sin(half_angle)
            );
        } else {
            // Small rotation approximation
            delta_q = Eigen::Quaterniond(1.0, delta_rot[0]/2.0, delta_rot[1]/2.0, delta_rot[2]/2.0);
            delta_q.normalize();
        }

        // Apply delta rotation to the original rotation
        Eigen::Quaterniond q_plus = q * delta_q;
        q_plus.normalize();

        // Apply delta translation
        Eigen::Vector3d t_plus = t + Eigen::Vector3d(delta[3], delta[4], delta[5]);

        // Store the result in x_plus_delta
        x_plus_delta[0] = q_plus.x();
        x_plus_delta[1] = q_plus.y();
        x_plus_delta[2] = q_plus.z();
        x_plus_delta[3] = q_plus.w();
        x_plus_delta[4] = t_plus[0];
        x_plus_delta[5] = t_plus[1];
        x_plus_delta[6] = t_plus[2];

        return true;
    }

    // Jacobian of the Plus operation
    bool PlusJacobian(const double* x, double* jacobian) const override {
        // Initialize Jacobian to zero
        Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> J(jacobian);
        J.setZero();

        // Extract the quaternion
        Eigen::Quaterniond q(x[3], x[0], x[1], x[2]); // w, x, y, z
        
        // Compute the quaternion derivative part (3x3 block for qx, qy, qz)
        // This is the differential of quaternion multiplication by the exponential map
        Eigen::Matrix<double, 3, 3> R = q.toRotationMatrix();
        
        // For small delta, the Jacobian of quaternion rotation is approximately:
        // θ = theta angle, u = rotation axis
        // q * exp(delta) ≈ q * [cos(θ/2), u*sin(θ/2)]
        // near zero, cos(θ/2) ≈ 1, sin(θ/2) ≈ θ/2
        // so the derivative of q.x, q.y, q.z w.r.t. delta is approximately 0.5 * R
        J.block<3, 3>(0, 0) = 0.5 * Eigen::Matrix3d::Identity();
        
        // The rotation affects the translation part too, but for small deltas
        // we can ignore this effect as a reasonable approximation
        
        // The translation part has a simple 3x3 identity block
        J.block<3, 3>(4, 3) = Eigen::Matrix3d::Identity();

        return true;
    }

    // Minus operation: y_minus_x computes the delta that takes x to y
    bool Minus(const double* y, const double* x, double* y_minus_x) const override {
        // Extract the quaternions and translations
        Eigen::Quaterniond q_x(x[3], x[0], x[1], x[2]); // w, x, y, z
        Eigen::Vector3d t_x(x[4], x[5], x[6]);
        
        Eigen::Quaterniond q_y(y[3], y[0], y[1], y[2]); // w, x, y, z
        Eigen::Vector3d t_y(y[4], y[5], y[6]);

        // Compute the rotation difference: q_diff = q_y * q_x^(-1)
        Eigen::Quaterniond q_diff = q_y * q_x.conjugate();
        q_diff.normalize();
        
        // Convert to axis-angle representation (logarithmic map)
        Eigen::AngleAxisd angle_axis(q_diff);
        double angle = angle_axis.angle();
        Eigen::Vector3d axis = angle_axis.axis();
        
        // The rotation part of y_minus_x
        Eigen::Vector3d delta_rot = axis * angle;
        
        // The translation part of y_minus_x
        Eigen::Vector3d delta_trans = t_y - t_x;

        // Store the result
        y_minus_x[0] = delta_rot[0];
        y_minus_x[1] = delta_rot[1];
        y_minus_x[2] = delta_rot[2];
        y_minus_x[3] = delta_trans[0];
        y_minus_x[4] = delta_trans[1];
        y_minus_x[5] = delta_trans[2];

        return true;
    }

    // Compute the Jacobian of the Minus operation with respect to y
    bool MinusJacobian(const double* x, double* jacobian) const override {
        // Initialize Jacobian to zero
        Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> J(jacobian);
        J.setZero();

        // For small angle differences, the Jacobian of the rotation part
        // has a 3x3 block for the quaternion part (qx, qy, qz)
        // The Jacobian of the logarithmic map is approximately identity for small rotations
        J.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
        
        // The translation part has an identity block
        J.block<3, 3>(3, 4) = Eigen::Matrix3d::Identity();

        return true;
    }

private:
    bool _fix_scale;
};

// Cost function for SE3 relative pose constraints between keyframes
class SE3RelativePoseCostFunction : public ceres::CostFunction {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    SE3RelativePoseCostFunction(const Eigen::Matrix4d& T_ij, double weight = 1.0)
        : m_T_ij(T_ij), m_weight(weight) {
        
        // Extract the rotation and translation from T_ij
        m_R_ij = m_T_ij.block<3, 3>(0, 0);
        m_t_ij = m_T_ij.block<3, 1>(0, 3);
        
        // This cost function has 6 residuals and takes two SE3 poses
        set_num_residuals(6);
        mutable_parameter_block_sizes()->push_back(7);  // First SE3 pose
        mutable_parameter_block_sizes()->push_back(7);  // Second SE3 pose
    }

    bool Evaluate(double const* const* parameters,
                  double* residuals,
                  double** jacobians) const {
        
        // Extract SE3 parameters (vertex i and j)
        const double* pose_i = parameters[0];
        const double* pose_j = parameters[1];
        
        // Extract quaternions and translations
        Eigen::Quaterniond q_i(pose_i[3], pose_i[0], pose_i[1], pose_i[2]); // w, x, y, z
        Eigen::Vector3d t_i(pose_i[4], pose_i[5], pose_i[6]);
        
        Eigen::Quaterniond q_j(pose_j[3], pose_j[0], pose_j[1], pose_j[2]); // w, x, y, z
        Eigen::Vector3d t_j(pose_j[4], pose_j[5], pose_j[6]);

        // Compute the predicted relative transformation
        Eigen::Matrix3d R_i = q_i.toRotationMatrix();
        Eigen::Matrix3d R_j = q_j.toRotationMatrix();
        
        // Predicted relative rotation: R_ij_pred = R_i.transpose() * R_j
        Eigen::Matrix3d R_ij_pred = R_i.transpose() * R_j;
        
        // Predicted relative translation: t_ij_pred = R_i.transpose() * (t_j - t_i)
        Eigen::Vector3d t_ij_pred = R_i.transpose() * (t_j - t_i);
        
        // Compute the error between measured and predicted
        // Rotation error: R_error = R_ij.transpose() * R_ij_pred
        // This gives us a rotation matrix representing how much R_ij_pred differs from R_ij
        Eigen::Matrix3d R_error = m_R_ij.transpose() * R_ij_pred;
        
        // Convert rotation error to angle-axis representation
        Eigen::AngleAxisd angle_axis_error(R_error);
        Eigen::Vector3d rot_error = angle_axis_error.angle() * angle_axis_error.axis();
        
        // Translation error: t_error = t_ij_pred - t_ij
        Eigen::Vector3d trans_error = t_ij_pred - m_t_ij;
        
        // Apply weight and fill in residuals
        for (int i = 0; i < 3; ++i) {
            residuals[i] = rot_error[i] * m_weight;
            residuals[i + 3] = trans_error[i] * m_weight;
        }
        
        // Compute jacobians if requested
        if (jacobians) {
            if (jacobians[0]) {
                // Jacobian with respect to pose_i
                Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> J_i(jacobians[0]);
                J_i.setZero();
                
                // For rotation error w.r.t. pose_i
                // Approximation: small changes in quaternion params -> small changes in rotation
                J_i.block<3, 3>(0, 0) = -R_j * R_i.transpose();
                
                // For translation error w.r.t. pose_i translation
                J_i.block<3, 3>(3, 4) = -R_i.transpose();
                
                // For translation error w.r.t. pose_i rotation
                // This is more complex as rotation affects the translation error
                // We can use an approximation for small changes:
                Eigen::Matrix3d skew_t_diff;
                Eigen::Vector3d t_diff = t_j - t_i;
                skew_t_diff << 0, -t_diff.z(), t_diff.y(),
                               t_diff.z(), 0, -t_diff.x(),
                               -t_diff.y(), t_diff.x(), 0;
                J_i.block<3, 3>(3, 0) = R_i.transpose() * skew_t_diff;
                
                // Apply weight
                J_i *= m_weight;
            }
            
            if (jacobians[1]) {
                // Jacobian with respect to pose_j
                Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> J_j(jacobians[1]);
                J_j.setZero();
                
                // For rotation error w.r.t. pose_j
                J_j.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
                
                // For translation error w.r.t. pose_j translation
                J_j.block<3, 3>(3, 4) = R_i.transpose();
                
                // Apply weight
                J_j *= m_weight;
            }
        }
        
        return true;
    }

private:
    Eigen::Matrix4d m_T_ij;
    Eigen::Matrix3d m_R_ij;
    Eigen::Vector3d m_t_ij;
    double m_weight;
};

// Helper function to convert a 4x4 matrix stored in row-major order as a string to an Eigen::Matrix4d
Eigen::Matrix4d parseMatrix4d(const std::string& matrix_str) {
    std::istringstream iss(matrix_str);
    std::vector<double> values;
    double value;
    
    while (iss >> value) {
        values.push_back(value);
    }
    
    Eigen::Matrix4d mat = Eigen::Matrix4d::Identity();
    if (values.size() == 16) {
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                mat(i, j) = values[i * 4 + j];
            }
        }
    }
    
    return mat;
}

// Helper function to create quaternion from pose data
Eigen::Quaterniond getQuaternionFromPose(const std::vector<double>& pose) {
    return Eigen::Quaterniond(pose[3], pose[0], pose[1], pose[2]); // w, x, y, z
}

// Helper function to create translation from pose data
Eigen::Vector3d getTranslationFromPose(const std::vector<double>& pose) {
    return Eigen::Vector3d(pose[4], pose[5], pose[6]);
}

// Function to optimize the pose graph using Ceres for loop closure
void OptimizeEssentialGraph(
    const std::string& data_dir,
    bool bFixScale = true) {
    
    std::cout << "Optimizing Essential Graph for Loop Closure..." << std::endl;
    
    // 1. Read keyframe poses
    std::map<int, std::vector<double>> keyframe_poses;
    std::ifstream kf_poses_file(data_dir + "keyframe_poses.txt");
    if (!kf_poses_file.is_open()) {
        std::cerr << "Error: Could not open keyframe_poses.txt" << std::endl;
        return;
    }
    
    std::string line;
    std::getline(kf_poses_file, line); // Skip the header line
    int kf_id;
    double timestamp, tx, ty, tz, qx, qy, qz, qw;
    
    while (kf_poses_file >> kf_id >> timestamp >> tx >> ty >> tz >> qx >> qy >> qz >> qw) {
        std::vector<double> pose = {qx, qy, qz, qw, tx, ty, tz}; // qx, qy, qz, qw, tx, ty, tz format
        keyframe_poses[kf_id] = pose;
    }
    
    std::cout << "Read poses for " << keyframe_poses.size() << " keyframes" << std::endl;

    // 2. Read loop match information
    std::ifstream loop_match_file(data_dir + "loop_match.txt");
    if (!loop_match_file.is_open()) {
        std::cerr << "Error: Could not open loop_match.txt" << std::endl;
        return;
    }
    
    int current_kf_id, loop_kf_id;
    
    // Skip header lines
    std::getline(loop_match_file, line); // Skip first line (header)
    std::getline(loop_match_file, line); // Skip second line (comment)
    
    // Read current KF and loop KF IDs
    loop_match_file >> current_kf_id >> loop_kf_id;
    
    // Read the transformation matrix
    std::string matrix_line;
    std::getline(loop_match_file, matrix_line); // Clear the newline after reading IDs
    std::getline(loop_match_file, matrix_line); // Read the actual matrix line
    Eigen::Matrix4d loop_constraint = parseMatrix4d(matrix_line);
    
    std::cout << "Loop constraint between KF " << current_kf_id << " and KF " << loop_kf_id << std::endl;
    
    // 3. Read keyframe info including max kf id
    int init_kf_id = 0;
    int max_kf_id = 0;
    std::ifstream map_info_file(data_dir + "map_info.txt");
    if (map_info_file.is_open()) {
        std::string key;
        map_info_file >> key >> key; // MAP_ID
        map_info_file >> key >> init_kf_id; // INIT_KF_ID
        map_info_file >> key >> max_kf_id; // MAX_KF_ID
        std::cout << "Init KF: " << init_kf_id << ", Max KF: " << max_kf_id << std::endl;
    } else {
        // If we can't read the file, estimate max_kf_id from keyframe_poses
        for (const auto& kf_pair : keyframe_poses) {
            max_kf_id = std::max(max_kf_id, kf_pair.first);
        }
        std::cout << "Could not read map_info.txt, estimated Max KF: " << max_kf_id << std::endl;
    }
    
    // 4. Read corrected Sim3 poses (for RGBD/stereo, these are SE3 with fixed scale=1)
    std::map<int, std::vector<double>> corrected_poses;
    std::ifstream corrected_file(data_dir + "corrected_sim3.txt");
    if (corrected_file.is_open()) {
        std::getline(corrected_file, line); // Skip the header line
        int kf_id;
        double scale, tx, ty, tz, qx, qy, qz, qw;
        
        while (corrected_file >> kf_id >> scale >> tx >> ty >> tz >> qx >> qy >> qz >> qw) {
            std::vector<double> pose = {qx, qy, qz, qw, tx, ty, tz}; // qx, qy, qz, qw, tx, ty, tz format
            corrected_poses[kf_id] = pose;
        }
        
        std::cout << "Read corrected poses for " << corrected_poses.size() << " keyframes" << std::endl;
    }
    
    // 5. Read non-corrected Sim3 poses (original poses for corrected keyframes)
    std::map<int, std::vector<double>> non_corrected_poses;
    std::ifstream non_corrected_file(data_dir + "non_corrected_sim3.txt");
    if (non_corrected_file.is_open()) {
        std::getline(non_corrected_file, line); // Skip the header line
        int kf_id;
        double scale, tx, ty, tz, qx, qy, qz, qw;
        
        while (non_corrected_file >> kf_id >> scale >> tx >> ty >> tz >> qx >> qy >> qz >> qw) {
            std::vector<double> pose = {qx, qy, qz, qw, tx, ty, tz}; // qx, qy, qz, qw, tx, ty, tz format
            non_corrected_poses[kf_id] = pose;
        }
        
        std::cout << "Read non-corrected poses for " << non_corrected_poses.size() << " keyframes" << std::endl;
    }
    
    // 6. Read spanning tree information (child -> parent mapping)
    std::map<int, int> spanning_tree; // child -> parent
    std::ifstream spanning_tree_file(data_dir + "spanning_tree.txt");
    if (!spanning_tree_file.is_open()) {
        std::cerr << "Error: Could not open spanning_tree.txt" << std::endl;
        return;
    }
    
    std::getline(spanning_tree_file, line); // Skip the header line
    int child_id, parent_id;
    
    while (spanning_tree_file >> child_id >> parent_id) {
        spanning_tree[child_id] = parent_id;
    }
    
    std::cout << "Read " << spanning_tree.size() << " spanning tree edges" << std::endl;
    
    // 7. Read covisibility graph (for strong edges between keyframes)
    const int minFeat = 100; // Minimum features for strong edges
    std::map<std::pair<int, int>, int> covisibility_weights; // (kf1_id, kf2_id) -> weight
    std::ifstream covisibility_file(data_dir + "covisibility.txt");
    if (!covisibility_file.is_open()) {
        std::cerr << "Error: Could not open covisibility.txt" << std::endl;
        return;
    }
    
    std::getline(covisibility_file, line); // Skip the header line
    int kf1_id, kf2_id, weight;
    
    while (covisibility_file >> kf1_id >> kf2_id >> weight) {
        covisibility_weights[std::make_pair(kf1_id, kf2_id)] = weight;
    }
    
    std::cout << "Read covisibility information for " << covisibility_weights.size() << " edges" << std::endl;

    // 8. Read loop connections if available
    std::map<int, std::set<int>> loop_connections;
    std::ifstream loop_conn_file(data_dir + "loop_connections.txt");
    if (loop_conn_file.is_open()) {
        std::getline(loop_conn_file, line); // Skip the header line
        
        while (std::getline(loop_conn_file, line)) {
            if (line.empty() || line[0] == '#') continue;
            
            std::istringstream iss(line);
            int kf_id;
            if (!(iss >> kf_id)) continue;
            
            std::set<int> connections;
            int connected_kf;
            while (iss >> connected_kf) {
                connections.insert(connected_kf);
            }
            
            if (!connections.empty()) {
                loop_connections[kf_id] = connections;
            }
        }
        
        std::cout << "Read loop connections for " << loop_connections.size() << " keyframes" << std::endl;
    } else {
        // If loop_connections.txt doesn't exist, create a default one with just the loop closure pair
        std::set<int> loop_kfs;
        loop_kfs.insert(loop_kf_id);
        loop_connections[current_kf_id] = loop_kfs;
        std::cout << "Created default loop connection between KF " << current_kf_id << " and KF " << loop_kf_id << std::endl;
    }

    // Helper function to get covisibility weight between two keyframes
    auto getCovisibilityWeight = [&](int kf1_id, int kf2_id) -> int {
        auto it = covisibility_weights.find(std::make_pair(kf1_id, kf2_id));
        if (it != covisibility_weights.end()) {
            return it->second;
        }
        
        // Check the reverse order
        it = covisibility_weights.find(std::make_pair(kf2_id, kf1_id));
        if (it != covisibility_weights.end()) {
            return it->second;
        }
        
        return 0; // No covisibility found
    };

    // 9. Set up the optimization problem
    ceres::Problem problem;
    
    // Create the SE3 manifold
    ceres::Manifold* se3_manifold = new SE3Manifold(bFixScale);
    
    // Storage for the optimized poses
    std::map<int, double*> optimized_poses;
    
    // Vector for keeping track of allocated memory
    std::vector<double*> allocated_memory;
    
    // 10. Add the keyframe poses as variables to optimize
    for (auto& kf_pose : keyframe_poses) {
        int kf_id = kf_pose.first;
        
        // Allocate memory for the pose
        double* pose = new double[7];
        allocated_memory.push_back(pose);
        
        // Check if this keyframe has a corrected pose
        if (corrected_poses.find(kf_id) != corrected_poses.end()) {
            // Use the corrected pose for initial values
            for (int i = 0; i < 7; i++) {
                pose[i] = corrected_poses[kf_id][i];
            }
        } else {
            // Use the original pose
            for (int i = 0; i < 7; i++) {
                pose[i] = kf_pose.second[i];
            }
        }
        
        optimized_poses[kf_id] = pose;
        
        // Add the variable to the optimization problem
        problem.AddParameterBlock(optimized_poses[kf_id], 7, se3_manifold);
        
        // Fix the initial keyframe
        if (kf_id == init_kf_id) {
            problem.SetParameterBlockConstant(optimized_poses[kf_id]);
            std::cout << "Fixed KF " << kf_id << " (initial keyframe)" << std::endl;
        }
    }
    
    // 11. Add loop closure constraints
    std::set<std::pair<int, int>> inserted_edges;
    
    // Add the main loop closure constraint first
    if (optimized_poses.find(current_kf_id) != optimized_poses.end() && 
        optimized_poses.find(loop_kf_id) != optimized_poses.end()) {
        
        // Add the residual with high weight (1000.0) for loop closure
        ceres::CostFunction* loop_cost_function = 
            new SE3RelativePoseCostFunction(loop_constraint, 1000.0);
        
        problem.AddResidualBlock(
            loop_cost_function,
            nullptr, // No robust loss function
            optimized_poses[current_kf_id],
            optimized_poses[loop_kf_id]
        );
        
        // Mark this edge as inserted
        inserted_edges.insert(std::make_pair(std::min(current_kf_id, loop_kf_id), 
                                            std::max(current_kf_id, loop_kf_id)));
        
        std::cout << "Added loop closure constraint between KF " << current_kf_id 
                << " and KF " << loop_kf_id << " with weight 1000.0" << std::endl;
    }
    
    // Loop through all loop connections
    for (const auto& kf_pair : loop_connections) {
        int kf1_id = kf_pair.first;
        
        // Skip if this keyframe is not in our optimized_poses
        if (optimized_poses.find(kf1_id) == optimized_poses.end()) {
            continue;
        }
        
        const std::set<int>& connected_kfs = kf_pair.second;
        
        for (int kf2_id : connected_kfs) {
            // Skip if the connected keyframe is not in our optimized_poses
            if (optimized_poses.find(kf2_id) == optimized_poses.end()) {
                continue;
            }
            
            // Skip if we've already added this edge
            if (inserted_edges.count(std::make_pair(std::min(kf1_id, kf2_id), std::max(kf1_id, kf2_id)))) {
                continue;
            }
            
            // Implement ORB-SLAM3's filtering rule:
            // if((nIDi!=pCurKF->mnId || nIDj!=pLoopKF->mnId) && pKF->GetWeight(*sit)<minFeat)
            //     continue;
            bool isMainLoopEdge = (kf1_id == current_kf_id && kf2_id == loop_kf_id) || 
                                 (kf1_id == loop_kf_id && kf2_id == current_kf_id);
                                 
            // Skip low-weight edges that are not the main loop edge
            if (!isMainLoopEdge) {
                int covis_weight = getCovisibilityWeight(kf1_id, kf2_id);
                if (covis_weight < minFeat) {
                    continue; // Skip if weight is below minimum
                }
            }
            
            // First, determine if both keyframes are in the corrected region
            bool kf1_corrected = corrected_poses.find(kf1_id) != corrected_poses.end();
            bool kf2_corrected = corrected_poses.find(kf2_id) != corrected_poses.end();
            
            // Set up the transformation constraint
            Eigen::Matrix4d T_kf1_kf2 = Eigen::Matrix4d::Identity();
            
            if (kf1_corrected && kf2_corrected) {
                // Both in corrected region - use corrected poses
                Eigen::Quaterniond q1 = getQuaternionFromPose(corrected_poses[kf1_id]);
                Eigen::Vector3d t1 = getTranslationFromPose(corrected_poses[kf1_id]);
                
                Eigen::Quaterniond q2 = getQuaternionFromPose(corrected_poses[kf2_id]);
                Eigen::Vector3d t2 = getTranslationFromPose(corrected_poses[kf2_id]);
                
                Eigen::Matrix3d R1 = q1.toRotationMatrix();
                Eigen::Matrix3d R2 = q2.toRotationMatrix();
                
                Eigen::Matrix3d R12 = R1.transpose() * R2;
                Eigen::Vector3d t12 = R1.transpose() * (t2 - t1);
                
                T_kf1_kf2.block<3, 3>(0, 0) = R12;
                T_kf1_kf2.block<3, 1>(0, 3) = t12;
            } else {
                // At least one is uncorrected - use original poses
                Eigen::Quaterniond q1 = getQuaternionFromPose(keyframe_poses[kf1_id]);
                Eigen::Vector3d t1 = getTranslationFromPose(keyframe_poses[kf1_id]);
                
                Eigen::Quaterniond q2 = getQuaternionFromPose(keyframe_poses[kf2_id]);
                Eigen::Vector3d t2 = getTranslationFromPose(keyframe_poses[kf2_id]);
                
                Eigen::Matrix3d R1 = q1.toRotationMatrix();
                Eigen::Matrix3d R2 = q2.toRotationMatrix();
                
                Eigen::Matrix3d R12 = R1.transpose() * R2;
                Eigen::Vector3d t12 = R1.transpose() * (t2 - t1);
                
                T_kf1_kf2.block<3, 3>(0, 0) = R12;
                T_kf1_kf2.block<3, 1>(0, 3) = t12;
            }
            
            // Add the constraint with a standard weight (100.0)
            ceres::CostFunction* loop_conn_cost = new SE3RelativePoseCostFunction(T_kf1_kf2, 100.0);
            
            problem.AddResidualBlock(
                loop_conn_cost,
                nullptr, // No robust loss function
                optimized_poses[kf1_id],
                optimized_poses[kf2_id]
            );
            
            std::cout << "Added loop connection constraint between KF " << kf1_id 
                    << " and KF " << kf2_id << " with weight 100.0" << std::endl;
            
            // Mark this edge as inserted
            inserted_edges.insert(std::make_pair(std::min(kf1_id, kf2_id), std::max(kf1_id, kf2_id)));
        }
    }

    // 12. Add spanning tree constraints
    for (auto& edge : spanning_tree) {
        int child_id = edge.first;
        int parent_id = edge.second;
        
        // Skip if either keyframe is not in our optimized_poses
        if (optimized_poses.find(child_id) == optimized_poses.end() || 
            optimized_poses.find(parent_id) == optimized_poses.end()) {
            continue;
        }
        
        // Skip if we've already added this edge
        if (inserted_edges.count(std::make_pair(std::min(child_id, parent_id), std::max(child_id, parent_id)))) {
            continue;
        }
        
        // Check if this is a boundary between corrected and uncorrected regions
        bool child_corrected = corrected_poses.find(child_id) != corrected_poses.end();
        bool parent_corrected = corrected_poses.find(parent_id) != corrected_poses.end();
        bool is_boundary = (child_corrected != parent_corrected);
        
        // Setup the correct constraint depending on the region
        Eigen::Matrix4d T_parent_child = Eigen::Matrix4d::Identity();
        
        if (is_boundary) {
            // For boundary constraints, use pre-correction and original poses
            int corrected_id = parent_corrected ? parent_id : child_id;
            int uncorrected_id = parent_corrected ? child_id : parent_id;
            
            if (non_corrected_poses.find(corrected_id) != non_corrected_poses.end()) {
                // Get the pre-correction pose for the corrected keyframe
                Eigen::Quaterniond q_corrected_pre = getQuaternionFromPose(non_corrected_poses.at(corrected_id));
                Eigen::Vector3d t_corrected_pre = getTranslationFromPose(non_corrected_poses.at(corrected_id));
                
                // Get the original pose for the uncorrected keyframe
                Eigen::Quaterniond q_uncorrected = getQuaternionFromPose(keyframe_poses.at(uncorrected_id));
                Eigen::Vector3d t_uncorrected = getTranslationFromPose(keyframe_poses.at(uncorrected_id));
                
                // Compute the true relative transform that should be maintained
                Eigen::Matrix3d R_corrected_pre = q_corrected_pre.toRotationMatrix();
                Eigen::Matrix3d R_uncorrected = q_uncorrected.toRotationMatrix();
                
                if (corrected_id == parent_id) {
                    // Parent is corrected, child is uncorrected
                    Eigen::Matrix3d R_rel = R_corrected_pre.transpose() * R_uncorrected;
                    Eigen::Vector3d t_rel = R_corrected_pre.transpose() * (t_uncorrected - t_corrected_pre);
                    
                    T_parent_child.block<3, 3>(0, 0) = R_rel;
                    T_parent_child.block<3, 1>(0, 3) = t_rel;
                } else {
                    // Child is corrected, parent is uncorrected
                    Eigen::Matrix3d R_rel = R_uncorrected.transpose() * R_corrected_pre;
                    Eigen::Vector3d t_rel = R_uncorrected.transpose() * (t_corrected_pre - t_uncorrected);
                    
                    T_parent_child.block<3, 3>(0, 0) = R_rel;
                    T_parent_child.block<3, 1>(0, 3) = t_rel;
                }
                
                std::cout << "Used TRUE relative pose for boundary constraint between KF " 
                          << parent_id << " and KF " << child_id << std::endl;
            } else {
                // Fall back to standard approach if no pre-correction data available
                std::cerr << "WARNING: No pre-correction data for boundary keyframe " << corrected_id 
                          << ", using potentially incorrect constraint!" << std::endl;
                
                // Use the current poses (which might lead to inconsistent constraints)
                Eigen::Quaterniond q_parent, q_child;
                Eigen::Vector3d t_parent, t_child;
                
                if (parent_corrected) {
                    q_parent = getQuaternionFromPose(corrected_poses[parent_id]);
                    t_parent = getTranslationFromPose(corrected_poses[parent_id]);
                    q_child = getQuaternionFromPose(keyframe_poses[child_id]);
                    t_child = getTranslationFromPose(keyframe_poses[child_id]);
                } else {
                    q_parent = getQuaternionFromPose(keyframe_poses[parent_id]);
                    t_parent = getTranslationFromPose(keyframe_poses[parent_id]);
                    q_child = getQuaternionFromPose(corrected_poses[child_id]);
                    t_child = getTranslationFromPose(corrected_poses[child_id]);
                }
                
                Eigen::Matrix3d R_parent = q_parent.toRotationMatrix();
                Eigen::Matrix3d R_child = q_child.toRotationMatrix();
                
                Eigen::Matrix3d R_parent_child = R_parent.transpose() * R_child;
                Eigen::Vector3d t_parent_child = R_parent.transpose() * (t_child - t_parent);
                
                T_parent_child.block<3, 3>(0, 0) = R_parent_child;
                T_parent_child.block<3, 1>(0, 3) = t_parent_child;
            }
        } else if (parent_corrected && child_corrected) {
            // Both in corrected region - use corrected poses
            Eigen::Quaterniond q_parent = getQuaternionFromPose(corrected_poses[parent_id]);
            Eigen::Vector3d t_parent = getTranslationFromPose(corrected_poses[parent_id]);
            
            Eigen::Quaterniond q_child = getQuaternionFromPose(corrected_poses[child_id]);
            Eigen::Vector3d t_child = getTranslationFromPose(corrected_poses[child_id]);
            
            Eigen::Matrix3d R_parent = q_parent.toRotationMatrix();
            Eigen::Matrix3d R_child = q_child.toRotationMatrix();
            
            Eigen::Matrix3d R_parent_child = R_parent.transpose() * R_child;
            Eigen::Vector3d t_parent_child = R_parent.transpose() * (t_child - t_parent);
            
            T_parent_child.block<3, 3>(0, 0) = R_parent_child;
            T_parent_child.block<3, 1>(0, 3) = t_parent_child;
        } else {
            // Both in uncorrected region - use original poses
            Eigen::Quaterniond q_parent = getQuaternionFromPose(keyframe_poses[parent_id]);
            Eigen::Vector3d t_parent = getTranslationFromPose(keyframe_poses[parent_id]);
            
            Eigen::Quaterniond q_child = getQuaternionFromPose(keyframe_poses[child_id]);
            Eigen::Vector3d t_child = getTranslationFromPose(keyframe_poses[child_id]);
            
            Eigen::Matrix3d R_parent = q_parent.toRotationMatrix();
            Eigen::Matrix3d R_child = q_child.toRotationMatrix();
            
            Eigen::Matrix3d R_parent_child = R_parent.transpose() * R_child;
            Eigen::Vector3d t_parent_child = R_parent.transpose() * (t_child - t_parent);
            
            T_parent_child.block<3, 3>(0, 0) = R_parent_child;
            T_parent_child.block<3, 1>(0, 3) = t_parent_child;
        }
        
        // Determine weight - higher weight for boundary constraints
        double constraint_weight = is_boundary ? 500.0 : 100.0;
        
        // Add the residual
        ceres::CostFunction* spanning_tree_cost_function = 
            new SE3RelativePoseCostFunction(T_parent_child, constraint_weight);
        
        problem.AddResidualBlock(
            spanning_tree_cost_function,
            nullptr, // No robust loss function
            optimized_poses[parent_id],
            optimized_poses[child_id]
        );
        
        // If this is a boundary constraint, print a message
        if (is_boundary) {
            std::cout << "Added boundary constraint between KF " << parent_id 
                      << " and KF " << child_id << " with weight " << constraint_weight << std::endl;
        }
        
        // Mark this edge as inserted
        inserted_edges.insert(std::make_pair(std::min(child_id, parent_id), std::max(child_id, parent_id)));
    }

    // 13. Add covisibility graph constraints
    for (const auto& weight_entry : covisibility_weights) {
        int kf1_id = weight_entry.first.first;
        int kf2_id = weight_entry.first.second;
        int weight = weight_entry.second;
        
        // Skip low-weight edges
        if (weight < minFeat) {
            continue;
        }
        
        // Skip if either keyframe is not in our optimized_poses
        if (optimized_poses.find(kf1_id) == optimized_poses.end() || 
            optimized_poses.find(kf2_id) == optimized_poses.end()) {
            continue;
        }
        
        // Skip if we've already added this edge
        if (inserted_edges.count(std::make_pair(std::min(kf1_id, kf2_id), std::max(kf1_id, kf2_id)))) {
            continue;
        }
        
        // Skip if this is a parent-child relationship in the spanning tree
        if (spanning_tree[kf1_id] == kf2_id || spanning_tree[kf2_id] == kf1_id) {
            continue;
        }
        
        // Check if these keyframes are both in the corrected region or both in the uncorrected region
        bool kf1_corrected = corrected_poses.find(kf1_id) != corrected_poses.end();
        bool kf2_corrected = corrected_poses.find(kf2_id) != corrected_poses.end();
        
        // Skip edges that cross the boundary (one corrected, one uncorrected)
        if (kf1_corrected != kf2_corrected) {
            continue;
        }
        
        // Setup the transformation constraint
        Eigen::Matrix4d T_kf1_kf2 = Eigen::Matrix4d::Identity();
        
        if (kf1_corrected && kf2_corrected) {
            // Both in corrected region - use corrected poses
            Eigen::Quaterniond q1 = getQuaternionFromPose(corrected_poses[kf1_id]);
            Eigen::Vector3d t1 = getTranslationFromPose(corrected_poses[kf1_id]);
            
            Eigen::Quaterniond q2 = getQuaternionFromPose(corrected_poses[kf2_id]);
            Eigen::Vector3d t2 = getTranslationFromPose(corrected_poses[kf2_id]);
            
            Eigen::Matrix3d R1 = q1.toRotationMatrix();
            Eigen::Matrix3d R2 = q2.toRotationMatrix();
            
            Eigen::Matrix3d R12 = R1.transpose() * R2;
            Eigen::Vector3d t12 = R1.transpose() * (t2 - t1);
            
            T_kf1_kf2.block<3, 3>(0, 0) = R12;
            T_kf1_kf2.block<3, 1>(0, 3) = t12;
        } else {
            // Both in uncorrected region - use original poses
            Eigen::Quaterniond q1 = getQuaternionFromPose(keyframe_poses[kf1_id]);
            Eigen::Vector3d t1 = getTranslationFromPose(keyframe_poses[kf1_id]);
            
            Eigen::Quaterniond q2 = getQuaternionFromPose(keyframe_poses[kf2_id]);
            Eigen::Vector3d t2 = getTranslationFromPose(keyframe_poses[kf2_id]);
            
            Eigen::Matrix3d R1 = q1.toRotationMatrix();
            Eigen::Matrix3d R2 = q2.toRotationMatrix();
            
            Eigen::Matrix3d R12 = R1.transpose() * R2;
            Eigen::Vector3d t12 = R1.transpose() * (t2 - t1);
            
            T_kf1_kf2.block<3, 3>(0, 0) = R12;
            T_kf1_kf2.block<3, 1>(0, 3) = t12;
        }
        
        // Scale the weight according to covisibility
        double weight_factor = std::min(2.0, weight / 100.0);
        double constraint_weight = 10.0 * weight_factor;
        
        ceres::CostFunction* covisibility_cost = new SE3RelativePoseCostFunction(T_kf1_kf2, constraint_weight);
        
        problem.AddResidualBlock(
            covisibility_cost,
            nullptr, // No robust loss function
            optimized_poses[kf1_id],
            optimized_poses[kf2_id]
        );
        
        // Mark this edge as inserted
        inserted_edges.insert(std::make_pair(std::min(kf1_id, kf2_id), std::max(kf1_id, kf2_id)));
    }

    // 14. Configure the solver
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 20;
    options.function_tolerance = 1e-6;
    options.gradient_tolerance = 1e-10;
    options.parameter_tolerance = 1e-8;
    
    // 15. Run the solver
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    
    std::cout << summary.BriefReport() << std::endl;
    
    // 16. Write optimized poses to file (convert from Tcw to Twc format)
    // Create the output directory if it doesn't exist
    std::string output_dir = "/Datasets/CERES_Work/output/";
    system(("mkdir -p " + output_dir).c_str());
    
    std::string output_file = output_dir + "optimized_poses.txt";
    std::ofstream out_file(output_file);
    
    if (!out_file.is_open()) {
        std::cerr << "Error: Could not open output file: " << output_file << std::endl;
        return;
    }
    
    // Function to convert from Tcw to Twc format
    auto convertTcwToTwc = [](const double* tcw_pose, double* twc_pose) {
        // Extract quaternion and translation
        Eigen::Quaterniond q_cw(tcw_pose[3], tcw_pose[0], tcw_pose[1], tcw_pose[2]); // w, x, y, z
        Eigen::Vector3d t_cw(tcw_pose[4], tcw_pose[5], tcw_pose[6]);
        
        // Convert to Sophus::SE3d
        Sophus::SE3d Tcw(q_cw, t_cw);
        
        // Compute the inverse (Twc = Tcw^(-1))
        Sophus::SE3d Twc = Tcw.inverse();
        
        // Extract quaternion and translation from Twc
        Eigen::Quaterniond q_wc = Twc.unit_quaternion();
        Eigen::Vector3d t_wc = Twc.translation();
        
        // Store in array format [qx, qy, qz, qw, tx, ty, tz]
        twc_pose[0] = q_wc.x();
        twc_pose[1] = q_wc.y();
        twc_pose[2] = q_wc.z();
        twc_pose[3] = q_wc.w();
        twc_pose[4] = t_wc.x();
        twc_pose[5] = t_wc.y();
        twc_pose[6] = t_wc.z();
    };
    
    // Write in TUM format: timestamp tx ty tz qx qy qz qw (in Twc format)
    out_file << "# timestamp tx ty tz qx qy qz qw" << std::endl;
    
    // Vector to store all timestamps and keyframe IDs for sorting
    std::vector<std::pair<double, int>> timestamps_kfids;
    
    // Collect timestamps for all keyframes
    for (const auto& kf_pair : keyframe_poses) {
        int kf_id = kf_pair.first;
        double timestamp = 0.0;
        
        // Try to get timestamp from the original keyframe_poses file
        std::ifstream ts_file(data_dir + "keyframe_poses.txt");
        std::string ts_line;
        std::getline(ts_file, ts_line); // Skip header
        
        int id;
        double ts;
        double dummy1, dummy2, dummy3, dummy4, dummy5, dummy6, dummy7;
        while (ts_file >> id >> ts >> dummy1 >> dummy2 >> dummy3 >> dummy4 >> dummy5 >> dummy6 >> dummy7) {
            if (id == kf_id) {
                timestamp = ts;
                timestamps_kfids.push_back(std::make_pair(timestamp, kf_id));
                break;
            }
        }
    }
    
    // Sort by timestamp
    std::sort(timestamps_kfids.begin(), timestamps_kfids.end());
    
    // Write in chronological order (important for visualization and evaluation)
    for (const auto& ts_kf : timestamps_kfids) {
        double timestamp = ts_kf.first;
        int kf_id = ts_kf.second;
        
        if (optimized_poses.find(kf_id) != optimized_poses.end()) {
            const double* tcw_pose = optimized_poses[kf_id];
            double twc_pose[7]; // Temporary storage for Twc pose
            
            // Convert from Tcw to Twc format
            convertTcwToTwc(tcw_pose, twc_pose);
            
            // Write in TUM format: timestamp tx ty tz qx qy qz qw
            out_file << std::fixed << std::setprecision(9) << timestamp << " "
                     << twc_pose[4] << " " << twc_pose[5] << " " << twc_pose[6] << " "
                     << twc_pose[0] << " " << twc_pose[1] << " " << twc_pose[2] << " " << twc_pose[3] << std::endl;
        }
    }
    
    out_file.close();
    
    std::cout << "Optimized poses saved to: " << output_file << " (in Twc format for visualization)" << std::endl;
    

    
    std::cout << "Essential Graph Optimization complete!" << std::endl;
}

int main(int argc, char** argv) {
    std::string data_dir = "/Datasets/CERES_Work/input/optimization_data/";
    
    // Check if a data directory was provided
    if (argc > 1) {
        data_dir = argv[1];
        // Make sure the directory ends with a slash
        if (data_dir.back() != '/') {
            data_dir += '/';
        }
    }
    
    // For RGBD/stereo, bFixScale should be true
    bool bFixScale = true;
    
    // Run the optimization
    OptimizeEssentialGraph(data_dir, bFixScale);
    
    // Compare with ground truth if available
    std::string groundtruth_file = "/Datasets/CERES_Work/Vis_Result/standard_trajectory_with_loop.txt";
    std::string optimized_file = "/Datasets/CERES_Work/output/optimized_poses.txt";
    
    // Structure to store pose data for comparison
    struct PoseData {
        double timestamp;
        Eigen::Vector3d position;
        Eigen::Quaterniond rotation;
    };
    
    // Function to read a trajectory file in TUM format
    auto readTrajectory = [](const std::string& filename) {
        std::vector<PoseData> trajectory;
        std::ifstream file(filename);
        
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file " << filename << std::endl;
            return trajectory;
        }
        
        std::string line;
        while (std::getline(file, line)) {
            // Skip comment or empty lines
            if (line.empty() || line[0] == '#') {
                continue;
            }
            
            std::istringstream iss(line);
            PoseData pose;
            double tx, ty, tz, qx, qy, qz, qw;
            
            // Read timestamp and pose data
            if (!(iss >> pose.timestamp >> tx >> ty >> tz >> qx >> qy >> qz >> qw)) {
                continue; // Skip malformed lines
            }
            
            pose.position = Eigen::Vector3d(tx, ty, tz);
            pose.rotation = Eigen::Quaterniond(qw, qx, qy, qz); // Note: Eigen uses w, x, y, z order
            pose.rotation.normalize();
            
            trajectory.push_back(pose);
        }
        
        std::cout << "Read " << trajectory.size() << " poses from " << filename << std::endl;
        return trajectory;
    };
    
    // Function to calculate translation error between two poses
    auto calculateTranslationError = [](const Eigen::Vector3d& p1, const Eigen::Vector3d& p2) {
        return (p1 - p2).norm();
    };
    
    // Function to calculate rotation error between two quaternions (in degrees)
    auto calculateRotationError = [](const Eigen::Quaterniond& q1, const Eigen::Quaterniond& q2) {
        // Make sure the quaternions are normalized
        Eigen::Quaterniond q1_norm = q1.normalized();
        Eigen::Quaterniond q2_norm = q2.normalized();
        
        // Calculate the angular distance
        double dot_product = std::abs(q1_norm.dot(q2_norm));
        if (dot_product > 1.0) dot_product = 1.0; // Clamp to prevent numerical issues
        
        double angle_rad = 2.0 * std::acos(dot_product);
        double angle_deg = angle_rad * 180.0 / M_PI;
        
        return angle_deg;
    };
    
    // Read trajectories
    std::vector<PoseData> optimized_traj = readTrajectory(optimized_file);
    std::vector<PoseData> groundtruth_traj = readTrajectory(groundtruth_file);
    
    if (optimized_traj.empty() || groundtruth_traj.empty()) {
        std::cerr << "Error: One or both trajectory files could not be read." << std::endl;
        return -1;
    }
    
    // Check if trajectories have the same number of poses
    if (optimized_traj.size() != groundtruth_traj.size()) {
        std::cerr << "Warning: Trajectories have different numbers of poses. Using the smaller size." << std::endl;
    }
    
    // Vector to store errors for each associated pose
    std::vector<std::pair<double, double>> errors; // (translation_error, rotation_error)
    
    // Compare poses directly by index (keyframe-to-keyframe)
    size_t min_size = std::min(optimized_traj.size(), groundtruth_traj.size());
    for (size_t i = 0; i < min_size; i++) {
        double trans_error = calculateTranslationError(optimized_traj[i].position, groundtruth_traj[i].position);
        double rot_error = calculateRotationError(optimized_traj[i].rotation, groundtruth_traj[i].rotation);
        
        errors.push_back(std::make_pair(trans_error, rot_error));
    }
    
    std::cout << "\n=== Trajectory Comparison Results ===" << std::endl;
    std::cout << "Directly compared " << errors.size() << " poses by keyframe order" << std::endl;
    
    // Calculate average error for every 5 keyframes
    std::cout << "\n=== Error Analysis (Average Every 5 KFs) ===" << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    
    const int group_size = 5;
    int num_groups = (errors.size() + group_size - 1) / group_size; // Ceiling division
    
    std::cout << "+-------------+------------------+------------------+" << std::endl;
    std::cout << "| KF Group    | Translation (m)  | Rotation (deg)   |" << std::endl;
    std::cout << "+-------------+------------------+------------------+" << std::endl;
    
    double total_trans_error = 0.0;
    double total_rot_error = 0.0;
    
    for (int i = 0; i < num_groups; ++i) {
        int start_idx = i * group_size;
        int end_idx = std::min<int>(start_idx + group_size, errors.size());
        
        double avg_trans_error = 0.0;
        double avg_rot_error = 0.0;
        int count = 0;
        
        for (int j = start_idx; j < end_idx; ++j) {
            avg_trans_error += errors[j].first;
            avg_rot_error += errors[j].second;
            count++;
        }
        
        avg_trans_error /= count;
        avg_rot_error /= count;
        
        total_trans_error += avg_trans_error;
        total_rot_error += avg_rot_error;
        
        std::cout << "| " << std::setw(5) << start_idx << "-" << std::setw(5) << end_idx - 1 
                  << " | " << std::setw(16) << avg_trans_error 
                  << " | " << std::setw(16) << avg_rot_error << " |" << std::endl;
    }
    
    double overall_avg_trans_error = total_trans_error / num_groups;
    double overall_avg_rot_error = total_rot_error / num_groups;
    
    std::cout << "+-------------+------------------+------------------+" << std::endl;
    std::cout << "| Overall     | " << std::setw(16) << overall_avg_trans_error 
              << " | " << std::setw(16) << overall_avg_rot_error << " |" << std::endl;
    std::cout << "+-------------+------------------+------------------+" << std::endl;
    
    // Calculate RMSE
    double rmse_trans = 0.0;
    double rmse_rot = 0.0;
    
    for (const auto& error : errors) {
        rmse_trans += error.first * error.first;
        rmse_rot += error.second * error.second;
    }
    
    rmse_trans = std::sqrt(rmse_trans / errors.size());
    rmse_rot = std::sqrt(rmse_rot / errors.size());
    
    std::cout << "\n=== Overall Error Statistics ===" << std::endl;
    std::cout << "Number of matched poses: " << errors.size() << std::endl;
    std::cout << "RMSE Translation: " << rmse_trans << " m" << std::endl;
    std::cout << "RMSE Rotation: " << rmse_rot << " deg" << std::endl;
    
    return 0;
}
