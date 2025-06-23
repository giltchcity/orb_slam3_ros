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

// SE3 Lie Algebraic Parameterization - VertexSim3Expmap mimicking ORB-SLAM3 
// SE3 Lie Algebraic Parameterization - Sim3 implementation based on g2o but with fixed scale of 1
class SE3Parameterization : public ceres::Manifold {
public:
    ~SE3Parameterization() {}

    // 7-dimensional ambient space: [tx, ty, tz, qx, qy, qz, qw]
    int AmbientSize() const override { return 7; }
    
    // 6-dimensional tangent space: [rho(3), phi(3)] - we're actually using the Sim3 parameterization but ignoring the scale
    int TangentSize() const override { return 6; }
    
    // Exponential map implementation based on g2o::Sim3
    bool Plus(const double* x, const double* delta, double* x_plus_delta) const override {
        // Extract current state
        Eigen::Vector3d t_current(x[0], x[1], x[2]);
        Eigen::Quaterniond q_current(x[6], x[3], x[4], x[5]);
        q_current.normalize();
        
        // Extract Lie algebra increment in g2o::Sim3 format: [rotation(3), translation(3), scale(1)]
        // But we ignore scale increment (set to 0)
        Eigen::Vector3d omega(delta[0], delta[1], delta[2]);    // Rotation part
        Eigen::Vector3d upsilon(delta[3], delta[4], delta[5]);  // Translation part
        double sigma = 0.0;                                     // Fixed scale, increment is 0
        
        
        double theta = omega.norm();
        double s = std::exp(sigma);  // s is always 1 (since sigma=0)
        
        // Compute rotation matrix
        Eigen::Matrix3d Omega = skew(omega);
        Eigen::Matrix3d Omega2 = Omega * Omega;
        Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
        Eigen::Matrix3d R;
        
        // A, B, C coefficients, following g2o's calculation
        double eps = 1e-5;
        double A, B, C;
        
        // Handle different cases (small angle, large angle, scale near 1, etc.)
        if (fabs(sigma) < eps) {
            // Scale change close to 0 (our case)
            C = 1;
            if (theta < eps) {
                // Small angle case
                A = 0.5;
                B = 1.0/6.0;
                R = I + Omega + Omega2 * 0.5;
            } else {
                // Larger angle case
                double theta2 = theta * theta;
                A = (1.0 - std::cos(theta)) / theta2;
                B = (theta - std::sin(theta)) / (theta2 * theta);
                R = I + std::sin(theta) / theta * Omega + 
                    (1.0 - std::cos(theta)) / (theta * theta) * Omega2;
            }
        } else {
            // Significant scale change (not our case, but keep g2o's complete logic)
            C = (s - 1.0) / sigma;
            if (theta < eps) {
                // Significant scale change but small angle
                double sigma2 = sigma * sigma;
                A = ((sigma - 1.0) * s + 1.0) / sigma2;
                B = ((0.5 * sigma2 - sigma + 1.0) * s - 1.0) / (sigma2 * sigma);
                R = I + Omega + Omega2 * 0.5;
            } else {
                // Significant scale change and angle
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
        
        // Compute translation increment
        Eigen::Matrix3d W = A * Omega + B * Omega2 + C * I;
        Eigen::Vector3d t_delta = W * upsilon;
        
        
        // Apply increment: note using g2o::Sim3's multiplication formula
        Eigen::Matrix3d R_current = q_current.toRotationMatrix();
        Eigen::Matrix3d R_new = R * R_current;          // Rotation update
        Eigen::Vector3d t_new = s * (R * t_current) + t_delta;  // Translation update, s=1
        
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
        // Compute Jacobian of Plus operation
        Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> J(jacobian);
        J.setZero();
        
        // Numerical differentiation for Jacobian (simplified implementation)
        const double eps = 1e-8;
        double x_plus_eps[7], x_minus_eps[7];
        double delta_plus[6], delta_minus[6];
        
        for (int i = 0; i < 6; ++i) {
            // Forward perturbation
            for (int j = 0; j < 6; ++j) {
                delta_plus[j] = (i == j) ? eps : 0.0;
                delta_minus[j] = (i == j) ? -eps : 0.0;
            }
            
            Plus(x, delta_plus, x_plus_eps);
            Plus(x, delta_minus, x_minus_eps);
            
            // Numerical differentiation
            for (int k = 0; k < 7; ++k) {
                J(k, i) = (x_plus_eps[k] - x_minus_eps[k]) / (2.0 * eps);
            }
        }
        
        return true;
    }
    
    // Helper function: compute skew-symmetric matrix
    static Eigen::Matrix3d skew(const Eigen::Vector3d& v) {
        Eigen::Matrix3d m;
        m << 0, -v(2), v(1),
             v(2), 0, -v(0),
             -v(1), v(0), 0;
        return m;
    }
    
    // Sim3 logarithmic map for computing difference between two poses
    bool Minus(const double* y, const double* x, double* y_minus_x) const override {
        // Extract two poses
        Eigen::Vector3d t_x(x[0], x[1], x[2]);
        Eigen::Vector3d t_y(y[0], y[1], y[2]);
        Eigen::Quaterniond q_x(x[6], x[3], x[4], x[5]);
        Eigen::Quaterniond q_y(y[6], y[3], y[4], y[5]);
        
        q_x.normalize();
        q_y.normalize();
        
        // Compute relative transformation
        Eigen::Matrix3d R_x = q_x.toRotationMatrix();
        Eigen::Matrix3d R_y = q_y.toRotationMatrix();
        Eigen::Matrix3d R_rel = R_x.transpose() * R_y;
        Eigen::Vector3d t_rel = R_x.transpose() * (t_y - t_x);
        
        // Compute logarithmic map of rotation (SO3 part)
        Eigen::Vector3d omega;
        double trace = R_rel.trace();
        
        // Handle different angle cases
        if (trace > 3.0 - 1e-6) {
            // Case close to identity matrix
            omega = 0.5 * Eigen::Vector3d(
                R_rel(2,1) - R_rel(1,2),
                R_rel(0,2) - R_rel(2,0),
                R_rel(1,0) - R_rel(0,1)
            );
        } else {
            double d = 0.5 * (trace - 1.0);
            // Limit d to [-1,1] range
            d = d > 1.0 ? 1.0 : (d < -1.0 ? -1.0 : d);
            double angle = std::acos(d);
            // Compute rotation axis
            Eigen::Vector3d axis;
            axis << R_rel(2,1) - R_rel(1,2), 
                    R_rel(0,2) - R_rel(2,0), 
                    R_rel(1,0) - R_rel(0,1);
            
            if (axis.norm() < 1e-10) {
                // Handle special case near 180 degrees
                // Find largest diagonal element per g2o implementation
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
                // Normal case
                axis.normalize();
                omega = angle * axis;
            }
        }
        
        // Compute logarithmic map of translation (following g2o's Sim3::log implementation)
        double angle = omega.norm();
        double scale = 1.0; // Fixed scale is 1
        
        // Compute coefficients A, B, C (following g2o implementation)
        double A, B, C;
        double eps = 1e-6;
        
        C = 1.0; // Since scale=1
        Eigen::Matrix3d Omega = skew(omega);
        Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
        Eigen::Matrix3d W;
        
        if (angle < eps) {
            // Small angle case
            A = 0.5;
            B = 1.0/6.0;
            W = I + 0.5 * Omega + (1.0/6.0) * Omega * Omega;
        } else {
            // Normal angle case
            double s = sin(angle);
            double c = cos(angle);
            A = (1.0 - c) / (angle * angle);
            B = (angle - s) / (angle * angle * angle);
            W = I + A * Omega + B * Omega * Omega;
        }
        
        // Compute Lie algebra of translation part
        Eigen::Vector3d upsilon = W.inverse() * t_rel;
        
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
        // Numerical differentiation for MinusJacobian
        Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> J(jacobian);
        J.setZero();
        
        const double eps = 1e-8;
        double y_plus[7], y_minus[7];
        double diff_plus[6], diff_minus[6];
        
        for (int i = 0; i < 7; ++i) {
            // Perturb the i-th component of y
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


// SO(3) logarithmic map: convert rotation matrix to axis-angle vector
template<typename T>
Eigen::Matrix<T, 3, 1> LogSO3(const Eigen::Matrix<T, 3, 3>& R) {
    // Compute rotation angle
    T trace = R.trace();
    T cos_angle = (trace - T(1.0)) * T(0.5);
    
    // Limit cos_angle to [-1, 1] range
    if (cos_angle > T(1.0)) cos_angle = T(1.0);
    if (cos_angle < T(-1.0)) cos_angle = T(-1.0);
    
    T angle = acos(cos_angle);
    
    Eigen::Matrix<T, 3, 1> omega;
    
    // Handle small angle case
    if (angle < T(1e-6)) {
        // For small angles, use first-order approximation
        T factor = T(0.5) * (T(1.0) + trace * trace / T(12.0));
        omega << factor * (R(2, 1) - R(1, 2)),
                 factor * (R(0, 2) - R(2, 0)),
                 factor * (R(1, 0) - R(0, 1));
    } else if (angle > T(M_PI - 1e-6)) {
        // Handle case near 180 degrees
        Eigen::Matrix<T, 3, 3> A = (R + R.transpose()) * T(0.5);
        A.diagonal().array() -= T(1.0);
        
        // Find largest diagonal element
        int max_idx = 0;
        T max_val = ceres::abs(A(0, 0));
        for (int i = 1; i < 3; ++i) {
            if (ceres::abs(A(i, i)) > max_val) {
                max_val = ceres::abs(A(i, i));
                max_idx = i;
            }
        }
        
        // Compute axis vector
        Eigen::Matrix<T, 3, 1> axis;
        axis[max_idx] = sqrt(A(max_idx, max_idx));
        for (int i = 0; i < 3; ++i) {
            if (i != max_idx) {
                axis[i] = A(max_idx, i) / axis[max_idx];
            }
        }
        axis.normalize();
        
        // Determine correct sign
        if ((R(2, 1) - R(1, 2)) * axis[0] + 
            (R(0, 2) - R(2, 0)) * axis[1] + 
            (R(1, 0) - R(0, 1)) * axis[2] < T(0.0)) {
            axis = -axis;
        }
        
        omega = angle * axis;
    } else {
        // General case
        T sin_angle = sin(angle);
        T factor = angle / (T(2.0) * sin_angle);
        omega << factor * (R(2, 1) - R(1, 2)),
                 factor * (R(0, 2) - R(2, 0)),
                 factor * (R(1, 0) - R(0, 1));
    }
    
    return omega;
}



// Class specifically for loop constraints - corresponding to g2o's EdgeSim3
class SE3LoopConstraintCost {
public:
    SE3LoopConstraintCost(const Eigen::Matrix4d& relative_transform, const Eigen::Matrix<double, 6, 6>& information)
        : relative_rotation_(relative_transform.block<3, 3>(0, 0)),
          relative_translation_(relative_transform.block<3, 1>(0, 3)),
          sqrt_information_(information.llt().matrixL()) {
    }
    
    template <typename T>
    bool operator()(const T* const pose_i, const T* const pose_j, T* residuals) const {
        // pose_i: first keyframe, pose_j: second keyframe
        // Format: [tx, ty, tz, qx, qy, qz, qw]
        
        // Extract poses
        Eigen::Matrix<T, 3, 1> t_i(pose_i[0], pose_i[1], pose_i[2]);
        Eigen::Matrix<T, 3, 1> t_j(pose_j[0], pose_j[1], pose_j[2]);
        Eigen::Quaternion<T> q_i(pose_i[6], pose_i[3], pose_i[4], pose_i[5]);
        Eigen::Quaternion<T> q_j(pose_j[6], pose_j[3], pose_j[4], pose_j[5]);
        
        // Convert to rotation matrices
        Eigen::Matrix<T, 3, 3> R_i = q_i.toRotationMatrix();
        Eigen::Matrix<T, 3, 3> R_j = q_j.toRotationMatrix();
        
        // Compute relative transformation T_ji = T_j * T_i^{-1} (corresponding to ORB-SLAM3's Sji = Sjw * Swi)
        Eigen::Matrix<T, 3, 3> R_ji = R_j * R_i.transpose();
        Eigen::Matrix<T, 3, 1> t_ji = R_j * (R_i.transpose() * (-t_i)) + t_j;
        
        // Expected relative transformation
        Eigen::Matrix<T, 3, 3> R_expected = relative_rotation_.cast<T>();
        Eigen::Matrix<T, 3, 1> t_expected = relative_translation_.cast<T>();
        
        // Compute rotation error
        Eigen::Matrix<T, 3, 3> R_error_mat = R_expected.transpose() * R_ji;
        Eigen::Matrix<T, 3, 1> rotation_error = LogSO3(R_error_mat);
        
        // Compute translation error
        Eigen::Matrix<T, 3, 1> translation_error = t_ji - t_expected;
        
        // Combine residuals
        residuals[0] = rotation_error[0];
        residuals[1] = rotation_error[1];
        residuals[2] = rotation_error[2];
        residuals[3] = translation_error[0];
        residuals[4] = translation_error[1];
        residuals[5] = translation_error[2];
        
        // Apply information matrix
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



// KeyFrame structure
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
    
    // SE3 state [tx, ty, tz, qx, qy, qz, qw]
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
        // Update SE3 state
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
    
    // Update translation and quaternion from SE3 state
    void UpdateFromState() {
        translation = Eigen::Vector3d(se3_state[0], se3_state[1], se3_state[2]);
        quaternion = Eigen::Quaterniond(se3_state[6], se3_state[3], se3_state[4], se3_state[5]);
        quaternion.normalize();
    }
};

// Data structures
struct OptimizationData {
    std::map<int, std::shared_ptr<KeyFrame>> keyframes;
    std::map<int, std::vector<int>> spanning_tree;
    std::map<int, std::map<int, int>> covisibility;
    std::map<int, std::vector<int>> loop_connections;

    // Loop edge information
    std::map<int, std::set<int>> loop_edges;  // KF_ID -> {Loop_KF_IDs}
    
    int loop_kf_id;
    int current_kf_id;
    bool fixed_scale;
    int init_kf_id;
    int max_kf_id;
    
    // Loop match relative transformation
    Eigen::Matrix4d loop_transform_matrix;
    
    // SE3 corrected poses
    std::map<int, KeyFrame> corrected_poses;
    std::map<int, KeyFrame> non_corrected_poses;

    // Store vertex initial poses
    std::map<int, Eigen::Matrix4d> vertex_initial_poses_Tcw; // World to camera
    std::map<int, Eigen::Matrix4d> vertex_initial_poses_Twc; // Camera to world

};

class ORBSlamLoopOptimizer {
private:
    OptimizationData data_;
    std::unique_ptr<ceres::Problem> problem_;

    // Shared between functions
    std::set<std::pair<long unsigned int, long unsigned int>> sInsertedEdges;
    
public:
    ORBSlamLoopOptimizer() {
        problem_ = std::make_unique<ceres::Problem>();
    }
    
    // Parse keyframe pose file
    bool ParseKeyFramePoses(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Cannot open file: " << filename << std::endl;
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
        
        std::cout << "Parsed keyframe poses: " << data_.keyframes.size() << " keyframes" << std::endl;
        return true;
    }
    
    // Parse keyframe info file
    bool ParseKeyFrameInfo(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Cannot open file: " << filename << std::endl;
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
        
        std::cout << "Keyframe info parsing complete" << std::endl;
        return true;
    }
    
    // Parse map info
    bool ParseMapInfo(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Cannot open file: " << filename << std::endl;
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
        
        std::cout << "Map info: INIT_KF_ID=" << data_.init_kf_id 
                  << ", MAX_KF_ID=" << data_.max_kf_id << std::endl;
        return true;
    }
    
    // Parse keyframe IDs
    bool ParseKeyFrameIds(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Cannot open file: " << filename << std::endl;
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
        
        std::cout << "Keyframe IDs: LOOP_KF_ID=" << data_.loop_kf_id 
                  << ", CURRENT_KF_ID=" << data_.current_kf_id 
                  << ", FIXED_SCALE=" << data_.fixed_scale << std::endl;
        return true;
    }
    
    // Parse loop match
    bool ParseLoopMatch(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Cannot open file: " << filename << std::endl;
            return false;
        }
        
        std::string line;
        data_.loop_transform_matrix.setIdentity();
        
        int line_count = 0;
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;
            
            if (line_count == 0) {
                // First line contains keyframe IDs, already read from keyframe_ids.txt
                line_count++;
                continue;
            }
            
            std::istringstream iss(line);
            // Read 4x4 transformation matrix
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    if (!(iss >> data_.loop_transform_matrix(i, j))) {
                        std::cerr << "Error: Cannot parse transformation matrix" << std::endl;
                        return false;
                    }
                }
            }
            break;
        }
        
        std::cout << "Loop match transformation matrix:" << std::endl;
        std::cout << data_.loop_transform_matrix << std::endl;
        return true;
    }
    
    // Parse corrected Sim3 (convert to SE3, ignore scale)
    bool ParseCorrectedSim3(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Cannot open file: " << filename << std::endl;
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
        
        std::cout << "Corrected SE3: " << data_.corrected_poses.size() << " keyframes" << std::endl;
        return true;
    }
    
    // Parse non-corrected Sim3 (convert to SE3, ignore scale)
    bool ParseNonCorrectedSim3(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Cannot open file: " << filename << std::endl;
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
        
        std::cout << "Non-corrected SE3: " << data_.non_corrected_poses.size() << " keyframes" << std::endl;
        return true;
    }
    
    // Parse spanning tree
    bool ParseSpanningTree(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Cannot open file: " << filename << std::endl;
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
        
        std::cout << "Spanning tree: " << data_.spanning_tree.size() << " parent nodes" << std::endl;
        return true;
    }

    // Parse loop edges file
    bool ParseLoopEdges(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Cannot open file: " << filename << std::endl;
            return false;
        }
        
        std::string line;
        int loop_edge_count = 0;
        
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;
            
            std::istringstream iss(line);
            int kf_id, loop_kf_id;
            
            if (iss >> kf_id >> loop_kf_id) {
                // Add bidirectional loop edges
                data_.loop_edges[kf_id].insert(loop_kf_id);
                data_.loop_edges[loop_kf_id].insert(kf_id);
                loop_edge_count++;
            }
        }
        
        std::cout << "Loop edges: " << loop_edge_count << " edges, involving " << data_.loop_edges.size() << " keyframes" << std::endl;
        return true;
    }

    // Parse covisibility
    bool ParseCovisibility(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Cannot open file: " << filename << std::endl;
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
        
        std::cout << "Covisibility: " << data_.covisibility.size() << " keyframes" << std::endl;
        return true;
    }
    
    // Parse loop connections
    bool ParseLoopConnections(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Cannot open file: " << filename << std::endl;
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
        
        std::cout << "Loop connections: " << data_.loop_connections.size() << " keyframes" << std::endl;
        return true;
    }
    
    // Parse all data files
    bool ParseAllData(const std::string& data_dir) {
        std::string base_path = data_dir;
        if (base_path.back() != '/') base_path += "/";
        
        // Parse data files
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
        if (!ParseLoopEdges(base_path + "loop_edges.txt")) return false;
        
        std::cout << "\nAll data files parsed successfully" << std::endl;
        return true;
    }
    
    // Setup optimization problem and add keyframe vertices
    void SetupOptimizationProblem() {
        std::cout << "\nSetting up optimization problem..." << std::endl;
        
        // Clear previous poses
        data_.vertex_initial_poses_Tcw.clear();
        data_.vertex_initial_poses_Twc.clear();
        
        // Add vertex for each keyframe
        for (auto& kf_pair : data_.keyframes) {
            auto& kf = kf_pair.second;
            
            // Skip bad keyframes
            if (kf->is_bad) continue;
            
            // Define Tcw matrix (camera pose in world coordinates)
            Eigen::Matrix4d Tcw = Eigen::Matrix4d::Identity();
            bool useCorrected = false;
            
            // Check for corrected pose
            if (data_.corrected_poses.find(kf->id) != data_.corrected_poses.end()) {
                // Use corrected SE3
                const auto& corrected = data_.corrected_poses[kf->id];
                for (int i = 0; i < 7; ++i) {
                    kf->se3_state[i] = corrected.se3_state[i];
                }
                kf->UpdateFromState();
                
                // Save corrected pose as initial value
                Tcw.block<3, 3>(0, 0) = corrected.quaternion.toRotationMatrix();
                Tcw.block<3, 1>(0, 3) = corrected.translation;
                useCorrected = true;
            } else {
                // Use current keyframe state
                Tcw.block<3, 3>(0, 0) = kf->quaternion.toRotationMatrix();
                Tcw.block<3, 1>(0, 3) = kf->translation;
            }
            
            // Compute Twc (world pose in camera coordinates)
            Eigen::Matrix4d Twc = Tcw.inverse();
            
            // Save poses
            data_.vertex_initial_poses_Tcw[kf->id] = Tcw;
            data_.vertex_initial_poses_Twc[kf->id] = Twc;
            
            // Add parameter block to optimization problem
            problem_->AddParameterBlock(kf->se3_state.data(), 7);
            
            // Set SE3 parameterization (manifold) - similar to ORB-SLAM3's VertexSim3Expmap
            problem_->SetManifold(kf->se3_state.data(), new SE3Parameterization());
            
            // Fix initial keyframe
            if (kf->id == data_.init_kf_id || kf->is_fixed) {
                problem_->SetParameterBlockConstant(kf->se3_state.data());
                std::cout << "Fixed keyframe " << kf->id << std::endl;
            }
            
            if (kf->id % 100 == 0 || kf->id == data_.init_kf_id || kf->id == data_.current_kf_id || kf->id == data_.loop_kf_id) {
                std::cout << "  Added keyframe vertex KF" << kf->id << ": "
                         << (useCorrected ? "using CorrectedPose" : "using original pose") 
                         << " position: [" << Tcw.block<3, 1>(0, 3).transpose() << "]" 
                         << std::endl;
            }
        }
        
        std::cout << "Added " << data_.keyframes.size() << " keyframe vertices" << std::endl;
        std::cout << "Saved " << data_.vertex_initial_poses_Tcw.size() << " initial poses" << std::endl;
        std::cout << "Optimization setup complete, parameter blocks: " << problem_->NumParameterBlocks() << std::endl;
    }

// Add loop constraints
    void AddLoopConstraints() {
        std::cout << "\n=== Adding Loop Edges ===" << std::endl;
        
        const int minFeat = 100; // Minimum feature threshold
        Eigen::Matrix<double, 6, 6> matLambda = Eigen::Matrix<double, 6, 6>::Identity();
        
        int count_loop = 0;
        int attempted_loop = 0;
        
        std::cout << "Minimum features for edge connection: " << minFeat << std::endl;
        std::cout << "\n=== Information Matrix ===" << std::endl;
        std::cout << matLambda << std::endl;
        
        // Check for saved vertex poses
        if (data_.vertex_initial_poses_Tcw.empty()) {
            std::cerr << "Error: No saved vertex initial poses found!" << std::endl;
            return;
        }
        
        // Iterate through all loop connections
        for (const auto& mit : data_.loop_connections) {
            int nIDi = mit.first;  // Corresponds to ORB-SLAM3's pKF->mnId
            const auto& spConnections = mit.second;  // Corresponds to ORB-SLAM3's spConnections
            
            // Check if keyframe i is valid
            if (data_.keyframes.find(nIDi) == data_.keyframes.end() || 
                data_.keyframes[nIDi]->is_bad) {
                continue;
            }
            
            auto pKF = data_.keyframes[nIDi];
            
            std::cout << "KF" << nIDi << " connections: " << spConnections.size() << std::endl;
            
            // Get transformation matrix for keyframe i
            Eigen::Matrix4d Siw = Eigen::Matrix4d::Identity();
            bool useCorrectedSim3_i = false;
            
            // Prefer corrected pose
            if (data_.corrected_poses.find(nIDi) != data_.corrected_poses.end()) {
                const auto& corrected = data_.corrected_poses[nIDi];
                Siw.block<3, 3>(0, 0) = corrected.quaternion.toRotationMatrix();
                Siw.block<3, 1>(0, 3) = corrected.translation;
                useCorrectedSim3_i = true;
            } 
            else if (data_.vertex_initial_poses_Tcw.find(nIDi) != data_.vertex_initial_poses_Tcw.end()) {
                Siw = data_.vertex_initial_poses_Tcw[nIDi];
            }
            else {
                // Should not reach here
                Siw.block<3, 3>(0, 0) = pKF->quaternion.toRotationMatrix();
                Siw.block<3, 1>(0, 3) = pKF->translation;
                std::cerr << "Warning: KF" << nIDi << " has no saved initial pose" << std::endl;
            }
            
            // Compute inverse transformation
            Eigen::Matrix4d Swi = Siw.inverse();
            
            // Iterate through connected keyframes
            for (int nIDj : spConnections) {
                attempted_loop++;
                
                // Check if keyframe j is valid
                if (data_.keyframes.find(nIDj) == data_.keyframes.end() || 
                    data_.keyframes[nIDj]->is_bad) {
                    continue;
                }
                
                auto pKFj = data_.keyframes[nIDj];
                
                // Weight check
                bool skipEdge = false;
                int weight = 0;
                
                // Check if main loop edge
                bool isMainLoopEdge = (nIDi == data_.current_kf_id && nIDj == data_.loop_kf_id);
                
                if (!isMainLoopEdge) {
                    // Bidirectional weight lookup
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
                    // Main loop edge - get weight but no threshold check
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
                
                // Get transformation matrix for keyframe j
                Eigen::Matrix4d Sjw = Eigen::Matrix4d::Identity();
                bool useCorrectedSim3_j = false;
                
                if (data_.corrected_poses.find(nIDj) != data_.corrected_poses.end()) {
                    const auto& corrected = data_.corrected_poses[nIDj];
                    Sjw.block<3, 3>(0, 0) = corrected.quaternion.toRotationMatrix();
                    Sjw.block<3, 1>(0, 3) = corrected.translation;
                    useCorrectedSim3_j = true;
                } 
                else if (data_.vertex_initial_poses_Tcw.find(nIDj) != data_.vertex_initial_poses_Tcw.end()) {
                    Sjw = data_.vertex_initial_poses_Tcw[nIDj];
                }
                else {
                    // Should not reach here
                    Sjw.block<3, 3>(0, 0) = pKFj->quaternion.toRotationMatrix();
                    Sjw.block<3, 1>(0, 3) = pKFj->translation;
                    std::cerr << "Warning: KF" << nIDj << " has no saved initial pose" << std::endl;
                }
                
                // Compute relative transformation
                Eigen::Matrix4d Sji = Sjw * Swi;
                
                // Create optimization edge
                ceres::CostFunction* cost_function = SE3LoopConstraintCost::Create(Sji, matLambda);
                
                // Add residual block
                problem_->AddResidualBlock(cost_function,
                                         nullptr,
                                         pKF->se3_state.data(),   // Keyframe i
                                         pKFj->se3_state.data()); // Keyframe j
                
                count_loop++;
                
                // Record edge
                sInsertedEdges.insert(std::make_pair(std::min((long unsigned int)nIDi, (long unsigned int)nIDj), 
                                                   std::max((long unsigned int)nIDi, (long unsigned int)nIDj)));
                
                // Output details
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
        
        // Information matrix (same as loop constraints)
        Eigen::Matrix<double, 6, 6> information = Eigen::Matrix<double, 6, 6>::Identity();
        std::cout << "\n=== Information Matrix ===" << std::endl;
        std::cout << information << std::endl;
        
        int normal_edges_added = 0;
        int attempted_normal = 0;
        int validKFCount = 0;  // Valid keyframe count
        
        // Check for saved vertex poses
        if (data_.vertex_initial_poses_Tcw.empty()) {
            std::cerr << "Error: No saved vertex initial poses found!" << std::endl;
            return;
        }
        
        // Phase 1: Add spanning tree edges (parent-child edges)
        std::cout << "\n=== Adding Spanning Tree Edges ===" << std::endl;
        
        // Iterate through all keyframes
        for (const auto& kf_pair : data_.keyframes) {
            int kf_id = kf_pair.first;
            auto& kf = kf_pair.second;
            
            // Skip bad keyframes
            if (kf->is_bad) continue;
            
            validKFCount++;
            
            // Print detailed info every 10 keyframes
            bool printDetailedInfo = (validKFCount % 10 == 0);
            
            if(printDetailedInfo) {
                std::cout << "Processing KF " << validKFCount << "/" << data_.keyframes.size() 
                          << " (KF" << kf_id << ")";
                
                // Show parent keyframe info
                if(kf->parent_id >= 0) {
                    std::cout << " | Parent: KF" << kf->parent_id;
                } else {
                    std::cout << " | Parent: None";
                }
                
                std::cout << std::endl;
            }
            
            // Get parent keyframe ID
            int parent_id = kf->parent_id;
            
            // Skip if no parent, self-loop, or invalid parent
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
            
            // Get inverse transformation for current keyframe Swi
            Eigen::Matrix4d T_iw = Eigen::Matrix4d::Identity();
            bool usingNonCorrected_i = false;
            
            // Use NonCorrectedSim3 if exists
            if (data_.non_corrected_poses.find(kf_id) != data_.non_corrected_poses.end()) {
                const auto& non_corrected_i = data_.non_corrected_poses[kf_id];
                Eigen::Matrix4d T_i = Eigen::Matrix4d::Identity();
                T_i.block<3, 3>(0, 0) = non_corrected_i.quaternion.toRotationMatrix();
                T_i.block<3, 1>(0, 3) = non_corrected_i.translation;
                T_iw = T_i.inverse();
                usingNonCorrected_i = true;
            } 
            else if (data_.vertex_initial_poses_Twc.find(kf_id) != data_.vertex_initial_poses_Twc.end()) {
                T_iw = data_.vertex_initial_poses_Twc[kf_id];
            }
            else {
                // Should not reach here
                std::cerr << "Warning: KF" << kf_id << " has no saved initial pose, using current state" << std::endl;
                Eigen::Matrix4d T_i = Eigen::Matrix4d::Identity();
                T_i.block<3, 3>(0, 0) = kf->quaternion.toRotationMatrix();
                T_i.block<3, 1>(0, 3) = kf->translation;
                T_iw = T_i.inverse();
            }
            
            // Get parent keyframe transformation Sjw
            Eigen::Matrix4d T_j = Eigen::Matrix4d::Identity();
            bool usingNonCorrected_j = false;
            
            if (data_.non_corrected_poses.find(parent_id) != data_.non_corrected_poses.end()) {
                const auto& non_corrected_parent = data_.non_corrected_poses[parent_id];
                T_j.block<3, 3>(0, 0) = non_corrected_parent.quaternion.toRotationMatrix();
                T_j.block<3, 1>(0, 3) = non_corrected_parent.translation;
                usingNonCorrected_j = true;
            }
            else if (data_.vertex_initial_poses_Tcw.find(parent_id) != data_.vertex_initial_poses_Tcw.end()) {
                T_j = data_.vertex_initial_poses_Tcw[parent_id];
            }
            else {
                // Should not reach here
                std::cerr << "Warning: Parent KF" << parent_id << " has no saved initial pose, using current state" << std::endl;
                T_j.block<3, 3>(0, 0) = data_.keyframes[parent_id]->quaternion.toRotationMatrix();
                T_j.block<3, 1>(0, 3) = data_.keyframes[parent_id]->translation;
            }
            
            // Compute relative transformation Sji = Sjw * Swi
            Eigen::Matrix4d T_ji = T_j * T_iw;
            Eigen::Vector3d translation = T_ji.block<3, 1>(0, 3);
            
            attempted_normal++;
            
            // Add constraint
            ceres::CostFunction* cost_function = SE3LoopConstraintCost::Create(T_ji, information);
            problem_->AddResidualBlock(cost_function, nullptr, 
                                      data_.keyframes[kf_id]->se3_state.data(),      // Child node
                                      data_.keyframes[parent_id]->se3_state.data()); // Parent node
            
            normal_edges_added++;
            
            
            // Print detailed info
            if(printDetailedInfo) {
                std::cout << "  Added Spanning Tree Edge: KF" << kf_id << " -> KF" << parent_id 
                          << " | Translation: [" << translation.transpose() << "]" 
                          << " | Scale: 1" << std::endl;
                
                // Additional detailed info
                std::string source_i = usingNonCorrected_i ? "NonCorrectedSim3" : "VertexInitialPose";
                std::string source_j = usingNonCorrected_j ? "NonCorrectedSim3" : "VertexInitialPose";
                
                std::cout << "    KF" << kf_id << " using: " << source_i
                          << " | KF" << parent_id << " using: " << source_j 
                          << std::endl;
                
                // Print camera direction and rotation info
                Eigen::Vector3d z_vec(0, 0, 1);
                Eigen::Vector3d z_dir_i = T_iw.block<3, 3>(0, 0) * z_vec;
                Eigen::Vector3d z_dir_j = T_j.inverse().block<3, 3>(0, 0) * z_vec;
                
                // Compute relative rotation angle
                double angle = acos(z_dir_i.dot(z_dir_j) / (z_dir_i.norm() * z_dir_j.norm())) * 180.0 / M_PI;
                
                std::cout << "    Camera Dir KF" << kf_id << ": [" << z_dir_i.transpose() << "]" << std::endl;
                std::cout << "    Camera Dir KF" << parent_id << ": [" << z_dir_j.transpose() << "]" << std::endl;
                std::cout << "    Rotation Angle: " << angle << " degrees" << std::endl;
                std::cout << "    Edge Info Trace: " << information.trace() << std::endl;
            }
        }
        
        std::cout << "\nSpanning Tree Edges Added: " << normal_edges_added << "/" << attempted_normal << std::endl;
        
        // Phase 2: Add covisibility graph edges
         std::cout << "\n=== Adding Covisibility Graph Edges ===" << std::endl;
        
       
        
        const int minFeat = 100;  // Minimum feature threshold (consistent with ORB-SLAM3)
        int count_covis = 0;      // Covisibility edges added to graph
        int count_all_valid_covis = 0;  // All valid covisibility relationships (including not added)
        std::vector<int> covis_per_kf(data_.max_kf_id + 1, 0);  // Covisibility edges per keyframe
        std::map<int, std::vector<std::pair<int, int>>> covis_weights;  // Store covisibility weights
        
        std::cout << "Minimum features for covisibility edge: " << minFeat << std::endl;
        
        // Iterate through all keyframes
        for (const auto& kf_pair : data_.keyframes) {
            int nIDi = kf_pair.first;  // Current keyframe ID
            auto& pKF = kf_pair.second;
            
            // Skip bad keyframes
            if (pKF->is_bad) continue;
            
            // Get inverse transformation for current keyframe Swi
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
            
            // Get parent keyframe ID
            int pParentKF = pKF->parent_id;
            
            // Get keyframes with covisibility weight >= minFeat
            // This simulates ORB-SLAM3's GetCovisiblesByWeight function
            std::vector<std::pair<int, int>> orderedConnections;
            if (data_.covisibility.find(nIDi) != data_.covisibility.end()) {
                // Get all connected keyframes from covisibility graph
                for (const auto& covis_pair : data_.covisibility[nIDi]) {
                    int connected_id = covis_pair.first;
                    int weight = covis_pair.second;
                    
                    // Keep only keyframes with weight >= minFeat
                    if (weight >= minFeat) {
                        orderedConnections.push_back(std::make_pair(connected_id, weight));
                    }
                }
                
                // Sort by weight descending (important! consistent with ORB-SLAM3)
                std::sort(orderedConnections.begin(), orderedConnections.end(), 
                    [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
                        return a.second > b.second; // Sort by weight descending
                    });
            }
            
            // Extract sorted keyframe IDs
            std::vector<int> vpConnectedKFs;
            for (const auto& pair : orderedConnections) {
                vpConnectedKFs.push_back(pair.first);
            }
            
            // Iterate through covisible keyframes
            for (int nIDj : vpConnectedKFs) {
                // Check if connected keyframe is valid
                if (data_.keyframes.find(nIDj) == data_.keyframes.end() || 
                    data_.keyframes[nIDj]->is_bad)
                    continue;
                
                auto pKFn = data_.keyframes[nIDj];
                

                
                // Check conditions:
                // 1. Not parent keyframe
                // 2. Not child of current keyframe (check from spanning_tree)
                bool isChild = false;
                if (data_.spanning_tree.find(nIDi) != data_.spanning_tree.end()) {
                    const auto& children = data_.spanning_tree[nIDi];
                    isChild = std::find(children.begin(), children.end(), nIDj) != children.end();
                }
                

                // Add debug code for KF 0
                if (nIDi == 0 || nIDj == 0) {
                    // Get covisibility weight
                    int weight = 0;
                    if (data_.covisibility.find(nIDi) != data_.covisibility.end() &&
                        data_.covisibility[nIDi].find(nIDj) != data_.covisibility[nIDi].end()) {
                        weight = data_.covisibility[nIDi][nIDj];
                    }
                    
                    std::cout << "DEBUG KF0: ";
                    std::cout << "Processing KF" << nIDi << " - KF" << nIDj 
                            << ", weight=" << weight
                            << ", parent-child:" << (nIDj == pParentKF || isChild ? "yes" : "no")
                            << ", nIDj < nIDi:" << (nIDj < nIDi ? "yes" : "no")
                            << ", already in edge set:" << (sInsertedEdges.count(std::make_pair(std::min(nIDi, nIDj), std::max(nIDi, nIDj))) ? "yes" : "no")
                            << std::endl;
                }
                
                



                
                if (nIDj != pParentKF && !isChild) {
                    // Count all valid covisibility relationships
                    count_all_valid_covis++;
                    
                    // Get covisibility weight
                    int weight = 0;
                    if (data_.covisibility.find(nIDi) != data_.covisibility.end() &&
                        data_.covisibility[nIDi].find(nIDj) != data_.covisibility[nIDi].end()) {
                        weight = data_.covisibility[nIDi][nIDj];
                    }
                    
                    // Store covisibility weight info
                    covis_weights[nIDi].push_back(std::make_pair(nIDj, weight));
                    
                    // Only process keyframes with ID less than current (avoid duplicates)
                    if (!pKFn->is_bad && nIDj < nIDi) {
                        // Check if edge already added
                        if (sInsertedEdges.count(std::make_pair(
                            std::min(nIDi, nIDj),
                            std::max(nIDi, nIDj))))
                            continue;
                        
                        // Get transformation for connected keyframe Snw
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
                        
                        // Compute relative transformation Sni = Snw * Swi
                        Eigen::Matrix4d Sni = Snw * Swi;
                        
                        // Add constraint
                        ceres::CostFunction* en = SE3LoopConstraintCost::Create(Sni, information);
                        problem_->AddResidualBlock(en, nullptr, 
                                                 data_.keyframes[nIDi]->se3_state.data(),
                                                 data_.keyframes[nIDj]->se3_state.data());
                        
                        // Add debug for KF 0 edge addition
                        if (nIDi == 0 || nIDj == 0) {
                            std::cout << "DEBUG KF0: Added edge KF" << nIDi << " - KF" << nIDj 
                                    << ", KF" << nIDi << " current edges:" << covis_per_kf[nIDi] + 1
                                    << ", KF" << nIDj << " current edges:" << covis_per_kf[nIDj] + 1
                                    << std::endl;
                        }
                        
                        // Increment covisibility edge counts
                        count_covis++;
                        covis_per_kf[nIDi]++;
                        covis_per_kf[nIDj]++;
                        normal_edges_added++;  // Include in total normal edge count
                        
                        // Record added edge
                        sInsertedEdges.insert(std::make_pair(
                            std::min(nIDi, nIDj),
                            std::max(nIDi, nIDj)
                        ));
                    }
                }
            }
        }

        // Statistics for keyframe covisibility relationships by ID intervals
        std::cout << "\nKeyframe covisibility statistics by ID groups:" << std::endl;
        for (int id = 0; id <= data_.max_kf_id; id += 20) {
            // Check if keyframe with this ID exists
            bool found = false;
            for (const auto& kf_pair : data_.keyframes) {
                if (kf_pair.first == id && !kf_pair.second->is_bad) {
                    found = true;
                    break;
                }
            }
            
            if (found) {
                std::cout << "KF ID " << id << ": Covisibility edges added to graph = " << covis_per_kf[id] << std::endl;
                
                // Output all valid covisibility relationships and weights
                if (covis_weights.find(id) != covis_weights.end()) {
                    std::cout << "  All valid covisibility relationships (ID, weight):" << std::endl;
                    for (const auto& pair : covis_weights[id]) {
                        std::cout << "  -> KF " << pair.first << ", weight: " << pair.second << std::endl;
                    }
                    std::cout << "  Total valid covisibility relationships: " << covis_weights[id].size() << std::endl;
                } else {
                    std::cout << "  No valid covisibility relationships" << std::endl;
                }
            }
        }
        
        std::cout << "\nTotal covisibility edges added to graph: " << count_covis << std::endl;
        std::cout << "All valid covisibility relationships (including not added to graph): " << count_all_valid_covis << std::endl;

        std::cout << "\nSuccessful Normal Edges: " << normal_edges_added << "/" << (attempted_normal + count_all_valid_covis) << std::endl;
        std::cout << "Total KeyFrames Processed: " << validKFCount << "/" << data_.keyframes.size() << std::endl;
    }
    
    // Output optimized poses
    void OutputOptimizedPoses(const std::string& output_file) {
        std::ofstream file(output_file);
        if (!file.is_open()) {
            std::cerr << "Cannot create output file: " << output_file << std::endl;
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
        
        std::cout << "Optimized poses saved to: " << output_file << std::endl;
    }



    void VerifyVertexPosesConsistency() {
        std::cout << "\n=== Verify Vertex Poses Consistency ===" << std::endl;
        
        if (data_.vertex_initial_poses_Tcw.empty()) {
            std::cout << "No saved vertex initial poses, skipping verification" << std::endl;
            return;
        }
        
        int mismatch_count = 0;
        double max_diff = 0.0;
        int max_diff_kf = -1;
        
        for (const auto& kf_pair : data_.keyframes) {
            int kf_id = kf_pair.first;
            const auto& kf = kf_pair.second;
            
            if (kf->is_bad) continue;
            
            // Check if saved initial pose exists
            if (data_.vertex_initial_poses_Tcw.find(kf_id) == data_.vertex_initial_poses_Tcw.end()) 
                continue;
            
            // Get saved initial pose
            const Eigen::Matrix4d& T_saved = data_.vertex_initial_poses_Tcw[kf_id];
            
            // Get current keyframe pose
            Eigen::Matrix4d T_current = Eigen::Matrix4d::Identity();
            T_current.block<3, 3>(0, 0) = kf->quaternion.toRotationMatrix();
            T_current.block<3, 1>(0, 3) = kf->translation;
            
            // Calculate position difference
            Eigen::Vector3d pos_saved = T_saved.block<3, 1>(0, 3);
            Eigen::Vector3d pos_current = T_current.block<3, 1>(0, 3);
            double pos_diff = (pos_saved - pos_current).norm();
            
            // Check for significant differences
            if (pos_diff > 1e-6) {
                mismatch_count++;
                
                if (pos_diff > max_diff) {
                    max_diff = pos_diff;
                    max_diff_kf = kf_id;
                }
                
                // Print differences for some keyframes
                if (kf_id % 100 == 0 || kf_id == data_.init_kf_id || kf_id == data_.loop_kf_id || kf_id == data_.current_kf_id) {
                    std::cout << "KF" << kf_id << " pose difference: " << pos_diff 
                             << " meters | Saved position: [" << pos_saved.transpose() 
                             << "] | Current position: [" << pos_current.transpose() << "]" << std::endl;
                }
            }
        }
        
        std::cout << "Total " << mismatch_count << " keyframes differ from saved initial poses" << std::endl;
        if (max_diff_kf >= 0) {
            std::cout << "Maximum difference: " << max_diff << " meters, at KF" << max_diff_kf << std::endl;
        }
    }




    // Get parameter block information
    void PrintProblemInfo() {
        std::cout << "\nOptimization problem info:" << std::endl;
        std::cout << "Parameter blocks: " << problem_->NumParameterBlocks() << std::endl;
        std::cout << "Residual blocks: " << problem_->NumResidualBlocks() << std::endl;
        std::cout << "Parameters: " << problem_->NumParameters() << std::endl;
        std::cout << "Residuals: " << problem_->NumResiduals() << std::endl;
    }

    // Output optimized Twc format poses (camera position in world coordinates)
    void OutputOptimizedPosesTwc(const std::string& output_file) {
        std::ofstream file(output_file);
        if (!file.is_open()) {
            std::cerr << "Cannot create output file: " << output_file << std::endl;
            return;
        }
        
        file << "# KF_ID tx ty tz qx qy qz qw (Twc format - camera position in world coordinates)" << std::endl;
        
        for (const auto& kf_pair : data_.keyframes) {
            const auto& kf = kf_pair.second;
            if (kf->is_bad) continue;
            
            // Get Tcw from SE3 state
            Eigen::Vector3d t_cw(kf->se3_state[0], kf->se3_state[1], kf->se3_state[2]);
            Eigen::Quaterniond q_cw(kf->se3_state[6], kf->se3_state[3], kf->se3_state[4], kf->se3_state[5]);
            
            // Build Tcw transformation matrix
            Eigen::Matrix4d T_cw = Eigen::Matrix4d::Identity();
            T_cw.block<3, 3>(0, 0) = q_cw.toRotationMatrix();
            T_cw.block<3, 1>(0, 3) = t_cw;
            
            // Calculate Twc = Tcw^(-1)
            Eigen::Matrix4d T_wc = T_cw.inverse();
            
            // Extract Twc translation and rotation
            Eigen::Vector3d t_wc = T_wc.block<3, 1>(0, 3);
            Eigen::Matrix3d R_wc = T_wc.block<3, 3>(0, 0);
            Eigen::Quaterniond q_wc(R_wc);
            
            // Output Twc format
            file << kf->id << " "
                 << t_wc.x() << " " << t_wc.y() << " " << t_wc.z() << " "
                 << q_wc.x() << " " << q_wc.y() << " " << q_wc.z() << " " << q_wc.w() << std::endl;
        }
        
        std::cout << "Optimized Twc poses saved to: " << output_file << std::endl;
    }
    
    // Output both Tcw and Twc formats
    void OutputBothFormats(const std::string& output_dir, const std::string& suffix = "") {
        std::string tcw_file = output_dir + "/poses_tcw" + suffix + ".txt";
        std::string twc_file = output_dir + "/poses_twc" + suffix + ".txt";
        
        // Output Tcw format
        OutputOptimizedPoses(tcw_file);
        
        // Output Twc format  
        OutputOptimizedPosesTwc(twc_file);
    }

    // Output optimized Twc format poses (TUM format, sorted by timestamp)
    void OutputOptimizedPosesTwcTUM(const std::string& output_file) {
        std::ofstream file(output_file);
        if (!file.is_open()) {
            std::cerr << "Cannot create output file: " << output_file << std::endl;
            return;
        }
        
        // Collect all keyframe timestamps and poses
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
            
            // Get Tcw from SE3 state
            Eigen::Vector3d t_cw(kf->se3_state[0], kf->se3_state[1], kf->se3_state[2]);
            Eigen::Quaterniond q_cw(kf->se3_state[6], kf->se3_state[3], kf->se3_state[4], kf->se3_state[5]);
            
            // Build Tcw transformation matrix
            Eigen::Matrix4d T_cw = Eigen::Matrix4d::Identity();
            T_cw.block<3, 3>(0, 0) = q_cw.toRotationMatrix();
            T_cw.block<3, 1>(0, 3) = t_cw;
            
            // Calculate Twc = Tcw^(-1)
            Eigen::Matrix4d T_wc = T_cw.inverse();
            
            // Extract Twc translation and rotation
            Eigen::Vector3d t_wc = T_wc.block<3, 1>(0, 3);
            Eigen::Matrix3d R_wc = T_wc.block<3, 3>(0, 0);
            Eigen::Quaterniond q_wc(R_wc);
            
            // Create KeyFramePose object
            KeyFramePose kf_pose;
            kf_pose.timestamp = kf->timestamp;
            kf_pose.kf_id = kf->id;
            kf_pose.position_wc = t_wc;
            kf_pose.quaternion_wc = q_wc;
            
            kf_poses.push_back(kf_pose);
        }
        
        // Sort by timestamp
        std::sort(kf_poses.begin(), kf_poses.end(), 
                  [](const KeyFramePose& a, const KeyFramePose& b) {
                      return a.timestamp < b.timestamp;
                  });
        
        // Output TUM format
        for (const auto& kf_pose : kf_poses) {
            // TUM format: timestamp tx ty tz qx qy qz qw
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
        
        std::cout << "Optimized TUM format Twc trajectory saved to: " << output_file << std::endl;
        std::cout << "Contains " << kf_poses.size() << " keyframes, sorted by timestamp" << std::endl;
    }
    
    // Output initial Twc format poses (TUM format, sorted by timestamp)
    void OutputInitialPosesTwcTUM(const std::string& output_file) {
        std::ofstream file(output_file);
        if (!file.is_open()) {
            std::cerr << "Cannot create output file: " << output_file << std::endl;
            return;
        }
        
        // Collect all keyframe timestamps and poses
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
            
            // Check for corrected pose, use if available, otherwise use current pose
            Eigen::Vector3d t_cw;
            Eigen::Quaterniond q_cw;
            
            if (data_.corrected_poses.find(kf->id) != data_.corrected_poses.end()) {
                const auto& corrected = data_.corrected_poses[kf->id];
                t_cw = corrected.translation;
                q_cw = corrected.quaternion;
            } else {
                // Use current pose as initial pose
                t_cw = kf->translation;
                q_cw = kf->quaternion;
            }
            
            // Build Tcw transformation matrix
            Eigen::Matrix4d T_cw = Eigen::Matrix4d::Identity();
            T_cw.block<3, 3>(0, 0) = q_cw.toRotationMatrix();
            T_cw.block<3, 1>(0, 3) = t_cw;
            
            // Calculate Twc = Tcw^(-1)
            Eigen::Matrix4d T_wc = T_cw.inverse();
            
            // Extract Twc translation and rotation
            Eigen::Vector3d t_wc = T_wc.block<3, 1>(0, 3);
            Eigen::Matrix3d R_wc = T_wc.block<3, 3>(0, 0);
            Eigen::Quaterniond q_wc(R_wc);
            
            // Create KeyFramePose object
            KeyFramePose kf_pose;
            kf_pose.timestamp = kf->timestamp;
            kf_pose.kf_id = kf->id;
            kf_pose.position_wc = t_wc;
            kf_pose.quaternion_wc = q_wc;
            
            kf_poses.push_back(kf_pose);
        }
        
        // Sort by timestamp
        std::sort(kf_poses.begin(), kf_poses.end(), 
                  [](const KeyFramePose& a, const KeyFramePose& b) {
                      return a.timestamp < b.timestamp;
                  });
        
        // Output TUM format
        for (const auto& kf_pose : kf_poses) {
            // TUM format: timestamp tx ty tz qx qy qz qw
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
        
        std::cout << "Initial TUM format Twc trajectory saved to: " << output_file << std::endl;
        std::cout << "Contains " << kf_poses.size() << " keyframes, sorted by timestamp" << std::endl;
    }
    
    // Output TUM format trajectory files
    void OutputTUMTrajectory(const std::string& output_dir) {
        std::string tum_before_file = output_dir + "/trajectory_before_optimization.txt";
        std::string tum_after_file = output_dir + "/trajectory_after_optimization.txt";
        
        // Output TUM format trajectory before optimization
        OutputInitialPosesTwcTUM(tum_before_file);
        
        // Output TUM format trajectory after optimization
        OutputOptimizedPosesTwcTUM(tum_after_file);
    }

    // Get keyframe info for debugging
    void PrintKeyFrameInfo(int id) {
        if (data_.keyframes.find(id) != data_.keyframes.end()) {
            const auto& kf = data_.keyframes[id];
            std::cout << "Keyframe " << id << ":" << std::endl;
            std::cout << "  Position: [" << kf->translation.transpose() << "]" << std::endl;
            std::cout << "  Quaternion: [" << kf->quaternion.x() << ", " << kf->quaternion.y() 
                      << ", " << kf->quaternion.z() << ", " << kf->quaternion.w() << "]" << std::endl;
            std::cout << "  Fixed: " << (kf->is_fixed ? "yes" : "no") << std::endl;
            std::cout << "  Bad: " << (kf->is_bad ? "yes" : "no") << std::endl;
        }
    }


    // Print optimization results
    void PrintOptimizationResults() {
        std::cout << "\n=== Optimization Results Analysis ===" << std::endl;
        
        // Print before/after comparison for key frames
        std::vector<int> key_frames = {0, data_.loop_kf_id, data_.current_kf_id};
        
        for (int kf_id : key_frames) {
            if (data_.keyframes.find(kf_id) == data_.keyframes.end()) continue;
            
            auto kf = data_.keyframes[kf_id];
            
            // Pose before optimization (from corrected_poses or original)
            Eigen::Vector3d pos_before;
            Eigen::Quaterniond quat_before;
            
            if (data_.corrected_poses.find(kf_id) != data_.corrected_poses.end()) {
                pos_before = data_.corrected_poses[kf_id].translation;
                quat_before = data_.corrected_poses[kf_id].quaternion;
            } else {
                // Use initial pose
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
            
            // Pose after optimization
            Eigen::Vector3d pos_after(kf->se3_state[0], kf->se3_state[1], kf->se3_state[2]);
            Eigen::Quaterniond quat_after(kf->se3_state[6], kf->se3_state[3], kf->se3_state[4], kf->se3_state[5]);
            
            // Calculate position change
            double pos_change = (pos_after - pos_before).norm();
            
            // Calculate rotation change (angle)
            Eigen::Quaterniond quat_diff = quat_before.inverse() * quat_after;
            double angle_change = 2.0 * acos(std::abs(quat_diff.w())) * 180.0 / M_PI;
            
            std::cout << "Keyframe " << kf_id << ":" << std::endl;
            std::cout << "  Position change: " << pos_change << " meters" << std::endl;
            std::cout << "  Rotation change: " << angle_change << " degrees" << std::endl;
            std::cout << "  Position before: [" << pos_before.transpose() << "]" << std::endl;
            std::cout << "  Position after: [" << pos_after.transpose() << "]" << std::endl;
            std::cout << std::endl;
        }
        
        // Calculate average position change for all keyframes
        double total_pos_change = 0.0;
        int count = 0;
        
        for (const auto& kf_pair : data_.keyframes) {
            if (kf_pair.second->is_bad) continue;
            
            // Simplified: assume frames without corrected poses have 0 position change
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
            std::cout << "Average position change: " << total_pos_change / count << " meters" << std::endl;
        }
        
        std::cout << "Keyframes optimized: " << count << std::endl;
    }


    bool OptimizeEssentialGraph() {
        std::cout << "\nStarting essential graph optimization..." << std::endl;
        
        // Configure solver options
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY; // Can try different solvers
        options.minimizer_progress_to_stdout = true;
        options.max_num_iterations = 100;  // Increased iterations
        options.num_threads = 4;
        options.function_tolerance = 1e-6;
        options.gradient_tolerance = 1e-8;
        options.parameter_tolerance = 1e-8;
        
        // Solve
        ceres::Solver::Summary summary;
        ceres::Solve(options, problem_.get(), &summary);
        
        std::cout << "\n=== Optimization Detailed Report ===" << std::endl;
        std::cout << summary.FullReport() << std::endl;
        
        // Update all keyframe poses
        for (auto& kf_pair : data_.keyframes) {
            kf_pair.second->UpdateFromState();
        }
        
        return summary.IsSolutionUsable();
    }
};



int main() {
    // Data file paths
    std::string data_dir = "/Datasets/CERES_Work/input/optimization_data";
    std::string output_dir = "/Datasets/CERES_Work/output";
    
    // Create output directory
    system(("mkdir -p " + output_dir).c_str());
    
    // Create optimizer
    ORBSlamLoopOptimizer optimizer;
    
    // Parse all data files
    if (!optimizer.ParseAllData(data_dir)) {
        std::cerr << "Data parsing failed" << std::endl;
        return -1;
    }
    
    // Setup optimization problem
    optimizer.SetupOptimizationProblem();
    
    // Verify pose consistency
    optimizer.VerifyVertexPosesConsistency();
    
    // Add loop constraints
    optimizer.AddLoopConstraints();
    

    // Add normal edge constraints (spanning tree)
    optimizer.AddNormalEdgeConstraints();

    
    // Verify pose consistency again after adding all edges
    optimizer.VerifyVertexPosesConsistency();
    
    // Print problem info
    optimizer.PrintProblemInfo();
    
    // Print some keyframe info for debugging
    optimizer.PrintKeyFrameInfo(0);  // Initial keyframe
    
    // Save poses before optimization
    std::cout << "\nSaving poses before optimization..." << std::endl;
    optimizer.OutputBothFormats(output_dir, "_before_optimization");
    
    // Execute optimization!
    std::cout << "\n=== Starting Loop Optimization ===" << std::endl;
    
    bool success = optimizer.OptimizeEssentialGraph();
    
    if (success) {
        std::cout << "\n=== Optimization Successfully Completed ===" << std::endl;
        
        // Output TUM format trajectory files (sorted by timestamp)
        std::cout << "\nSaving TUM format trajectory files..." << std::endl;
        optimizer.OutputTUMTrajectory(output_dir);
        
        // Output optimized poses (both formats)
        std::cout << "\nSaving optimized poses..." << std::endl;
        optimizer.OutputBothFormats(output_dir, "_after_optimization");
        
        // Print before/after comparison for some keyframes
        optimizer.PrintOptimizationResults();
        
        std::cout << "\nOutput files description:" << std::endl;
        std::cout << "- trajectory_before_optimization.txt: TUM format trajectory before optimization (sorted by timestamp)" << std::endl;
        std::cout << "- trajectory_after_optimization.txt: TUM format trajectory after optimization (sorted by timestamp)" << std::endl;
        std::cout << "- poses_tcw_*.txt: Tcw format (world to camera transformation)" << std::endl;
        std::cout << "- poses_twc_*.txt: Twc format (camera position in world coordinates)" << std::endl;
        
    } else {
        std::cout << "\n=== Optimization Failed ===" << std::endl;
        return -1;
    }
    
    
    return 0;
}
