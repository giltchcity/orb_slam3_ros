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
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <algorithm> // For std::sort and std::min
#include <sophus/se3.hpp>

// Function to convert pose from Tcw to Twc format
// Convert from parameter array to Sophus::SE3d (Tcw format)
Sophus::SE3d GetSE3FromArray(const double* pose) {
    // Extract quaternion and translation
    Eigen::Quaterniond q(pose[3], pose[0], pose[1], pose[2]); // w, x, y, z
    Eigen::Vector3d t(pose[4], pose[5], pose[6]);
    
    // Create and return the SE3 transformation
    return Sophus::SE3d(q, t);
}

// Convert from Sophus::SE3d to parameter array
void StoreArrayFromSE3(const Sophus::SE3d& se3, double* pose) {
    // Extract quaternion and translation
    Eigen::Quaterniond q = se3.unit_quaternion();
    Eigen::Vector3d t = se3.translation();
    
    // Store in array format [qx, qy, qz, qw, tx, ty, tz]
    pose[0] = q.x();
    pose[1] = q.y();
    pose[2] = q.z();
    pose[3] = q.w();
    pose[4] = t.x();
    pose[5] = t.y();
    pose[6] = t.z();
}

// Convert from Tcw to Twc format using Sophus
void convertTcwToTwc(const double* tcw_pose, double* twc_pose) {
    // Convert array to Sophus::SE3d
    Sophus::SE3d Tcw = GetSE3FromArray(tcw_pose);
    
    // Compute the inverse (Twc = Tcw^(-1))
    Sophus::SE3d Twc = Tcw.inverse();
    
    // Convert back to array format
    StoreArrayFromSE3(Twc, twc_pose);
}



// Structure to store pose data for comparison
struct PoseData {
    double timestamp;
    Eigen::Vector3d position;
    Eigen::Quaterniond rotation;
};

// Function to read a trajectory file without timestamp matching
std::vector<PoseData> readTrajectory(const std::string& filename) {
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
        
        // Read timestamp but don't use it for matching
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
}


// Function to find the closest pose by timestamp
int findClosestPose(const std::vector<PoseData>& trajectory, double timestamp, double max_diff = 0.02) {
    int closest_idx = -1;
    double min_diff = max_diff;
    
    for (size_t i = 0; i < trajectory.size(); ++i) {
        double diff = std::abs(trajectory[i].timestamp - timestamp);
        if (diff < min_diff) {
            min_diff = diff;
            closest_idx = i;
        }
    }
    
    return closest_idx;
}

// Function to calculate translation error between two poses
double calculateTranslationError(const Eigen::Vector3d& p1, const Eigen::Vector3d& p2) {
    return (p1 - p2).norm();
}

// Function to calculate rotation error between two quaternions (in degrees)
double calculateRotationError(const Eigen::Quaterniond& q1, const Eigen::Quaterniond& q2) {
    // Make sure the quaternions are normalized
    Eigen::Quaterniond q1_norm = q1.normalized();
    Eigen::Quaterniond q2_norm = q2.normalized();
    
    // Calculate the angular distance
    double dot_product = std::abs(q1_norm.dot(q2_norm));
    if (dot_product > 1.0) dot_product = 1.0; // Clamp to prevent numerical issues
    
    double angle_rad = 2.0 * std::acos(dot_product);
    double angle_deg = angle_rad * 180.0 / M_PI;
    
    return angle_deg;
}

// Implementation of the SE3 Manifold for Ceres 2.2
class SE3Manifold : public ceres::Manifold {
public:
    virtual ~SE3Manifold() {}

    // Ambient space: 7D for quaternion + translation (qx, qy, qz, qw, tx, ty, tz)
    virtual int AmbientSize() const override { return 7; }
    
    // Tangent space: 6D for rotation + translation
    virtual int TangentSize() const override { return 6; }

    // Plus operation: x_plus_delta = [q * exp(delta_rot), t + delta_trans]
    virtual bool Plus(const double* x, const double* delta, double* x_plus_delta) const override {
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

    // Compute the Jacobian of the Plus operation
    virtual bool PlusJacobian(const double* x, double* jacobian) const override {
        // Initialize Jacobian to zero
        Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> J(jacobian);
        J.setZero();

        // For small rotations, the Jacobian of the rotation part
        // has a 3x3 block for the quaternion part (x,y,z)
        J.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * 0.5;
        
        // The translation part has a 3x3 identity block
        J.block<3, 3>(4, 3) = Eigen::Matrix3d::Identity();

        return true;
    }

    // Minus operation: y_minus_x computes the delta that takes x to y
    virtual bool Minus(const double* y, const double* x, double* y_minus_x) const override {
        // Extract the quaternions and translations
        Eigen::Quaterniond q_x(x[3], x[0], x[1], x[2]); // w, x, y, z
        Eigen::Vector3d t_x(x[4], x[5], x[6]);
        
        Eigen::Quaterniond q_y(y[3], y[0], y[1], y[2]); // w, x, y, z
        Eigen::Vector3d t_y(y[4], y[5], y[6]);

        // Compute the rotation difference: q_diff = q_y * q_x^(-1)
        Eigen::Quaterniond q_diff = q_y * q_x.conjugate();
        
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

    // Compute the Jacobian of the Minus operation with respect to x
    virtual bool MinusJacobian(const double* x, double* jacobian) const override {
        // Initialize Jacobian to zero
        Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> J(jacobian);
        J.setZero();

        // For small rotations, the Jacobian of the rotation part
        // has a 3x3 block for the quaternion part (x,y,z)
        J.block<3, 3>(0, 0) = -Eigen::Matrix3d::Identity() * 2.0;
        
        // The translation part has a -I block
        J.block<3, 3>(3, 4) = -Eigen::Matrix3d::Identity();

        return true;
    }
};

// Cost function for SE3 relative pose constraints
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

    virtual ~SE3RelativePoseCostFunction() {}

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
                
                // The Jacobian blocks are complex for SE3 relative pose errors
                // For a production system, use automatic differentiation instead
                
                // For rotation error w.r.t. pose_i (simplified approximation)
                J_i.block<3, 3>(0, 0) = -Eigen::Matrix3d::Identity();
                
                // For translation error w.r.t. pose_i
                J_i.block<3, 3>(3, 4) = -R_i.transpose();
                
                // Apply weight
                J_i *= m_weight;
            }
            
            if (jacobians[1]) {
                // Jacobian with respect to pose_j
                Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> J_j(jacobians[1]);
                J_j.setZero();
                
                // For rotation error w.r.t. pose_j (simplified approximation)
                J_j.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
                
                // For translation error w.r.t. pose_j
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

// Helper function to convert quaternion and translation to 4x4 transformation matrix
Eigen::Matrix4d QuaternionTranslationToMatrix4d(const Eigen::Quaterniond& q, const Eigen::Vector3d& t) {
    Eigen::Matrix4d mat = Eigen::Matrix4d::Identity();
    mat.block<3, 3>(0, 0) = q.toRotationMatrix();
    mat.block<3, 1>(0, 3) = t;
    return mat;
}

// Helper function to convert a vector of values to a 4x4 matrix
Eigen::Matrix4d convertToMatrix4d(const std::vector<double>& values) {
    Eigen::Matrix4d mat = Eigen::Matrix4d::Identity();
    
    // Check if we have 16 values for a 4x4 matrix
    if (values.size() == 16) {
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                mat(i, j) = values[i * 4 + j];
            }
        }
    } else if (values.size() == 12) {
        // If we have 12 values (just 3x4 part)
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 4; j++) {
                mat(i, j) = values[i * 4 + j];
            }
        }
    }
    
    return mat;
}

// Helper function to convert a 4x4 matrix stored in row-major order as a string to an Eigen::Matrix4d
Eigen::Matrix4d parseMatrix4d(const std::string& matrix_str) {
    std::istringstream iss(matrix_str);
    std::vector<double> values;
    double value;
    
    while (iss >> value) {
        values.push_back(value);
    }
    
    return convertToMatrix4d(values);
}

// Helper function to print the 3D position of a keyframe
void print3DPosition(const std::string& label, const Eigen::Vector3d& position) {
    std::cout << "    " << label << ": [" 
              << position.x() << ", " 
              << position.y() << ", " 
              << position.z() << "]" << std::endl;
}

// Helper function to print a rotation in angle-axis format
void printRotation(const std::string& label, const Eigen::Matrix3d& rotation) {
    Eigen::AngleAxisd aa(rotation);
    double angle_deg = aa.angle() * 180.0 / M_PI;
    
    std::cout << "    " << label << ": " << angle_deg << " deg around [" 
              << aa.axis().x() << ", " << aa.axis().y() << ", " << aa.axis().z() << "]" << std::endl;
}

// Helper function to print a translation
void printTranslation(const std::string& label, const Eigen::Vector3d& translation) {
    std::cout << "    " << label << ": [" << translation.x() << ", " 
              << translation.y() << ", " << translation.z() 
              << "], norm: " << translation.norm() << std::endl;
}

// Helper function to create quaternion from pose data
Eigen::Quaterniond getQuaternionFromPose(const std::vector<double>& pose) {
    return Eigen::Quaterniond(pose[3], pose[0], pose[1], pose[2]); // w, x, y, z
}

// Helper function to create translation from pose data
Eigen::Vector3d getTranslationFromPose(const std::vector<double>& pose) {
    return Eigen::Vector3d(pose[4], pose[5], pose[6]);
}

// Function to compute and print relative pose with detailed information
void printDetailedRelativePose(
    int parent_id, int child_id,
    const double* parent_opt_pose, const double* child_opt_pose,
    const std::map<int, std::vector<double>>& keyframe_poses,
    const std::map<int, std::vector<double>>& corrected_poses,
    const std::map<int, std::vector<double>>& non_corrected_poses,
    bool is_boundary,
    int counter,
    bool is_before_optimization) {
    
    // Only print every 10 keyframes unless it's a boundary
    if (counter % 10 != 0 && !is_boundary) {
        return;
    }
    
    std::cout << "\n======================================================" << std::endl;
    std::cout << "  Relative pose KF " << parent_id << " -> KF " << child_id;
    if (is_before_optimization) {
        std::cout << " (BEFORE optimization)" << std::endl;
    } else {
        std::cout << " (AFTER optimization)" << std::endl;
    }
    std::cout << "======================================================" << std::endl;
    
    // Determine whether this is a boundary between corrected and uncorrected regions
    bool parent_corrected = (corrected_poses.find(parent_id) != corrected_poses.end());
    bool child_corrected = (corrected_poses.find(child_id) != corrected_poses.end());
    
    // CASE 1: Non-boundary keyframes
    if (!is_boundary) {
        // Extract original poses from keyframe_poses.txt
        Eigen::Quaterniond q_parent_orig = getQuaternionFromPose(keyframe_poses.at(parent_id));
        Eigen::Vector3d t_parent_orig = getTranslationFromPose(keyframe_poses.at(parent_id));
        
        Eigen::Quaterniond q_child_orig = getQuaternionFromPose(keyframe_poses.at(child_id));
        Eigen::Vector3d t_child_orig = getTranslationFromPose(keyframe_poses.at(child_id));
        
        // Compute original relative transform
        Eigen::Matrix3d R_parent_orig = q_parent_orig.toRotationMatrix();
        Eigen::Matrix3d R_child_orig = q_child_orig.toRotationMatrix();
        Eigen::Matrix3d R_rel_orig = R_parent_orig.transpose() * R_child_orig;
        Eigen::Vector3d t_rel_orig = R_parent_orig.transpose() * (t_child_orig - t_parent_orig);
        
        // Print original keyframe poses
        std::cout << "  ORIGINAL POSES (from keyframe_poses.txt):" << std::endl;
        std::cout << "  KF " << parent_id << ":" << std::endl;
        print3DPosition("Position", t_parent_orig);
        std::cout << "  KF " << child_id << ":" << std::endl;
        print3DPosition("Position", t_child_orig);
        
        // Print original relative pose
        std::cout << "  ORIGINAL RELATIVE POSE:" << std::endl;
        printRotation("Rotation", R_rel_orig);
        printTranslation("Translation", t_rel_orig);
        
        // Extract optimized poses
        Eigen::Quaterniond q_parent_opt(parent_opt_pose[3], parent_opt_pose[0], parent_opt_pose[1], parent_opt_pose[2]);
        Eigen::Vector3d t_parent_opt(parent_opt_pose[4], parent_opt_pose[5], parent_opt_pose[6]);
        
        Eigen::Quaterniond q_child_opt(child_opt_pose[3], child_opt_pose[0], child_opt_pose[1], child_opt_pose[2]);
        Eigen::Vector3d t_child_opt(child_opt_pose[4], child_opt_pose[5], child_opt_pose[6]);
        
        // Compute optimized relative transform
        Eigen::Matrix3d R_parent_opt = q_parent_opt.toRotationMatrix();
        Eigen::Matrix3d R_child_opt = q_child_opt.toRotationMatrix();
        Eigen::Matrix3d R_rel_opt = R_parent_opt.transpose() * R_child_opt;
        Eigen::Vector3d t_rel_opt = R_parent_opt.transpose() * (t_child_opt - t_parent_opt);
        
        // Print optimized keyframe poses
        if (is_before_optimization) {
            std::cout << "  INITIAL POSES (for optimization):" << std::endl;
        } else {
            std::cout << "  OPTIMIZED POSES:" << std::endl;
        }
        std::cout << "  KF " << parent_id << ":" << std::endl;
        print3DPosition("Position", t_parent_opt);
        std::cout << "  KF " << child_id << ":" << std::endl;
        print3DPosition("Position", t_child_opt);
        
        // Print optimized relative pose
        if (is_before_optimization) {
            std::cout << "  INITIAL RELATIVE POSE:" << std::endl;
        } else {
            std::cout << "  OPTIMIZED RELATIVE POSE:" << std::endl;
        }
        printRotation("Rotation", R_rel_opt);
        printTranslation("Translation", t_rel_opt);
    }
    // CASE 2: Boundary keyframes
    else {
        std::cout << "  *** BOUNDARY NODE BETWEEN CORRECTED AND UNCORRECTED REGIONS ***" << std::endl;
        
        // Determine which node is corrected/uncorrected
        int corrected_id = parent_corrected ? parent_id : child_id;
        int uncorrected_id = parent_corrected ? child_id : parent_id;
        
        std::cout << "  KF " << corrected_id << " (corrected) and KF " << uncorrected_id << " (uncorrected)" << std::endl;
        
        // Extract poses for corrected keyframe
        if (non_corrected_poses.find(corrected_id) != non_corrected_poses.end()) {
            // If we have pre-correction data for this keyframe
            Eigen::Quaterniond q_pre_correction = getQuaternionFromPose(non_corrected_poses.at(corrected_id));
            Eigen::Vector3d t_pre_correction = getTranslationFromPose(non_corrected_poses.at(corrected_id));
            
            std::cout << "  CORRECTED KF " << corrected_id << " (PRE-CORRECTION from non_corrected_sim3.txt):" << std::endl;
            print3DPosition("Position", t_pre_correction);
            
            Eigen::Quaterniond q_post_correction = getQuaternionFromPose(corrected_poses.at(corrected_id));
            Eigen::Vector3d t_post_correction = getTranslationFromPose(corrected_poses.at(corrected_id));
            
            std::cout << "  CORRECTED KF " << corrected_id << " (POST-CORRECTION from corrected_sim3.txt):" << std::endl;
            print3DPosition("Position", t_post_correction);
        } else {
            std::cout << "  WARNING: No pre-correction data found for KF " << corrected_id << std::endl;
        }
        
        // Extract pose for uncorrected keyframe
        Eigen::Quaterniond q_uncorrected = getQuaternionFromPose(keyframe_poses.at(uncorrected_id));
        Eigen::Vector3d t_uncorrected = getTranslationFromPose(keyframe_poses.at(uncorrected_id));
        
        std::cout << "  UNCORRECTED KF " << uncorrected_id << " (from keyframe_poses.txt):" << std::endl;
        print3DPosition("Position", t_uncorrected);
        
        // Compute and print the "true" relative pose at boundary
        // (using non_corrected pose for corrected node and original pose for uncorrected node)
        if (non_corrected_poses.find(corrected_id) != non_corrected_poses.end()) {
            Eigen::Quaterniond q_corrected_pre = getQuaternionFromPose(non_corrected_poses.at(corrected_id));
            Eigen::Vector3d t_corrected_pre = getTranslationFromPose(non_corrected_poses.at(corrected_id));
            
            Eigen::Matrix3d R_corrected_pre = q_corrected_pre.toRotationMatrix();
            Eigen::Matrix3d R_uncorrected = q_uncorrected.toRotationMatrix();
            
            Eigen::Matrix3d R_rel_true;
            Eigen::Vector3d t_rel_true;
            
            if (corrected_id == parent_id) {
                // Parent is corrected, child is uncorrected
                R_rel_true = R_corrected_pre.transpose() * R_uncorrected;
                t_rel_true = R_corrected_pre.transpose() * (t_uncorrected - t_corrected_pre);
            } else {
                // Child is corrected, parent is uncorrected
                R_rel_true = R_uncorrected.transpose() * R_corrected_pre;
                t_rel_true = R_uncorrected.transpose() * (t_corrected_pre - t_uncorrected);
            }
            
            std::cout << "  TRUE BOUNDARY RELATIVE POSE (using pre-correction and original poses):" << std::endl;
            printRotation("Rotation", R_rel_true);
            printTranslation("Translation", t_rel_true);
        }
        
        // Extract optimized poses
        Eigen::Quaterniond q_parent_opt(parent_opt_pose[3], parent_opt_pose[0], parent_opt_pose[1], parent_opt_pose[2]);
        Eigen::Vector3d t_parent_opt(parent_opt_pose[4], parent_opt_pose[5], parent_opt_pose[6]);
        
        Eigen::Quaterniond q_child_opt(child_opt_pose[3], child_opt_pose[0], child_opt_pose[1], child_opt_pose[2]);
        Eigen::Vector3d t_child_opt(child_opt_pose[4], child_opt_pose[5], child_opt_pose[6]);
        
        // Compute optimized relative transform
        Eigen::Matrix3d R_parent_opt = q_parent_opt.toRotationMatrix();
        Eigen::Matrix3d R_child_opt = q_child_opt.toRotationMatrix();
        Eigen::Matrix3d R_rel_opt = R_parent_opt.transpose() * R_child_opt;
        Eigen::Vector3d t_rel_opt = R_parent_opt.transpose() * (t_child_opt - t_parent_opt);
        
        // Print optimized keyframe poses
        if (is_before_optimization) {
            std::cout << "  INITIAL POSES (for optimization):" << std::endl;
        } else {
            std::cout << "  OPTIMIZED POSES:" << std::endl;
        }
        std::cout << "  KF " << parent_id << ":" << std::endl;
        print3DPosition("Position", t_parent_opt);
        std::cout << "  KF " << child_id << ":" << std::endl;
        print3DPosition("Position", t_child_opt);
        
        // Print optimized relative pose
        if (is_before_optimization) {
            std::cout << "  INITIAL RELATIVE POSE:" << std::endl;
        } else {
            std::cout << "  OPTIMIZED RELATIVE POSE:" << std::endl;
        }
        printRotation("Rotation", R_rel_opt);
        printTranslation("Translation", t_rel_opt);
    }
    
    std::cout << "======================================================" << std::endl;
}

int main(int argc, char** argv) {
    std::string data_dir = "/Datasets/CERES_Work/input/optimization_data/";
    
    // 1. Read loop match information
    std::ifstream loop_match_file(data_dir + "loop_match.txt");
    if (!loop_match_file.is_open()) {
        std::cerr << "Error: Could not open loop_match.txt" << std::endl;
        return -1;
    }
    
    int current_kf_id, loop_kf_id;
    std::string line;
    
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
    std::cout << "Loop constraint matrix:\n" << loop_constraint << std::endl;
    
    // 2. Read keyframe poses
    std::map<int, std::vector<double>> keyframe_poses;
    std::ifstream kf_poses_file(data_dir + "keyframe_poses.txt");
    if (!kf_poses_file.is_open()) {
        std::cerr << "Error: Could not open keyframe_poses.txt" << std::endl;
        return -1;
    }
    
    std::getline(kf_poses_file, line); // Skip the header line
    int kf_id;
    double timestamp, tx, ty, tz, qx, qy, qz, qw;
    
    while (kf_poses_file >> kf_id >> timestamp >> tx >> ty >> tz >> qx >> qy >> qz >> qw) {
        std::vector<double> pose = {qx, qy, qz, qw, tx, ty, tz}; // qx, qy, qz, qw, tx, ty, tz format
        keyframe_poses[kf_id] = pose;
    }
    
    std::cout << "Read poses for " << keyframe_poses.size() << " keyframes" << std::endl;
    
    // 3. Read corrected Sim3 poses (SE3 for RGBD/stereo)
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
    
    // 4. Read non-corrected Sim3 poses (original poses for corrected keyframes)
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
    
    // Identify the corrected range to find potential boundary
    int min_corrected_id = std::numeric_limits<int>::max();
    int max_corrected_id = 0;
    
    for (const auto& entry : corrected_poses) {
        min_corrected_id = std::min(min_corrected_id, entry.first);
        max_corrected_id = std::max(max_corrected_id, entry.first);
    }
    
    std::cout << "Corrected keyframe range: " << min_corrected_id << " to " << max_corrected_id << std::endl;
    
    // 5. Read spanning tree information
    std::map<int, int> spanning_tree; // child -> parent
    std::ifstream spanning_tree_file(data_dir + "spanning_tree.txt");
    if (!spanning_tree_file.is_open()) {
        std::cerr << "Error: Could not open spanning_tree.txt" << std::endl;
        return -1;
    }
    
    std::getline(spanning_tree_file, line); // Skip the header line
    int child_id, parent_id;
    
    while (spanning_tree_file >> child_id >> parent_id) {
        spanning_tree[child_id] = parent_id;
    }
    
    
    std::cout << "Read " << spanning_tree.size() << " spanning tree edges" << std::endl;
    
    // Identify and collect boundary keyframes
    std::set<std::pair<int, int>> boundary_keyframes;
    for (const auto& edge : spanning_tree) {
        int child_id = edge.first;
        int parent_id = edge.second;
        
        bool child_corrected = (corrected_poses.find(child_id) != corrected_poses.end());
        bool parent_corrected = (corrected_poses.find(parent_id) != corrected_poses.end());
        bool is_boundary = (child_corrected != parent_corrected);
        
        if (is_boundary) {
            boundary_keyframes.insert(std::make_pair(parent_id, child_id));
        }
    }
    
    std::cout << "\nFound " << boundary_keyframes.size() << " boundary keyframe pairs:" << std::endl;
    for (const auto& pair : boundary_keyframes) {
        std::cout << "  - KF " << pair.first << " -> KF " << pair.second;
        bool parent_corrected = (corrected_poses.find(pair.first) != corrected_poses.end());
        if (parent_corrected) {
            std::cout << " (Parent corrected, Child uncorrected)" << std::endl;
        } else {
            std::cout << " (Parent uncorrected, Child corrected)" << std::endl;
        }
    }
    std::cout << std::endl;
    
    // 6. Read map information
    std::ifstream map_info_file(data_dir + "map_info.txt");
    if (!map_info_file.is_open()) {
        std::cerr << "Error: Could not open map_info.txt" << std::endl;
        return -1;
    }
    
    int map_id, init_kf_id, max_kf_id, num_kfs, num_mps;
    std::string key;
    
    map_info_file >> key >> map_id;
    map_info_file >> key >> init_kf_id;
    map_info_file >> key >> max_kf_id;
    map_info_file >> key >> num_kfs;
    map_info_file >> key >> num_mps;
    
    std::cout << "Map info: " << num_kfs << " keyframes, " << num_mps << " map points" << std::endl;
    std::cout << "Init KF: " << init_kf_id << ", Max KF: " << max_kf_id << std::endl;
    
    // Create the optimization problem
    ceres::Problem problem;
    
    // Create the SE3 manifold
    ceres::Manifold* se3_manifold = new SE3Manifold();
    
    // Storage for the optimized poses
    std::map<int, double*> optimized_poses;
    
    // Add the keyframe poses as variables to optimize
    for (auto& kf_pose : keyframe_poses) {
        int kf_id = kf_pose.first;
        
        // Check if this keyframe has a corrected pose
        if (corrected_poses.find(kf_id) != corrected_poses.end()) {
            // Use the corrected pose for initial values
            double* pose = new double[7];
            for (int i = 0; i < 7; i++) {
                pose[i] = corrected_poses[kf_id][i];
            }
            optimized_poses[kf_id] = pose;
            std::cout << "Using corrected pose for KF " << kf_id << std::endl;
        } else {
            // Use the original pose
            double* pose = new double[7];
            for (int i = 0; i < 7; i++) {
                pose[i] = kf_pose.second[i];
            }
            optimized_poses[kf_id] = pose;
        }
        
        // Add the variable to the optimization problem
        problem.AddParameterBlock(optimized_poses[kf_id], 7, se3_manifold);
        
        // Fix the initial keyframe
        if (kf_id == init_kf_id) {
            problem.SetParameterBlockConstant(optimized_poses[kf_id]);
            std::cout << "Fixed KF " << kf_id << " (initial keyframe)" << std::endl;
        }
    }
    
    // Add the loop closure constraint
    if (optimized_poses.find(current_kf_id) != optimized_poses.end() && 
        optimized_poses.find(loop_kf_id) != optimized_poses.end()) {
        
        // Add the residual with high weight (1000.0) for loop closure
        ceres::CostFunction* loop_cost_function = 
            new SE3RelativePoseCostFunction(loop_constraint, 1000.0);
        
        problem.AddResidualBlock(
            loop_cost_function,
            nullptr, // No robust loss function for now
            optimized_poses[current_kf_id],
            optimized_poses[loop_kf_id]
        );
        
        std::cout << "Added loop closure constraint between KF " << current_kf_id 
                  << " and KF " << loop_kf_id << " with weight 1000.0" << std::endl;
    } else {
        std::cerr << "Warning: Could not add loop constraint, keyframes not found" << std::endl;
    }
    
    // Add spanning tree constraints
    std::cout << "\nBefore optimization - Relative poses:" << std::endl;
    
    int spanning_tree_count = 0;
    int print_counter = 0;
    
    for (auto& edge : spanning_tree) {
        int child_id = edge.first;
        int parent_id = edge.second;
        
        if (optimized_poses.find(child_id) != optimized_poses.end() && 
            optimized_poses.find(parent_id) != optimized_poses.end()) {
            
            // Check if we're at a boundary between corrected and uncorrected poses
            bool child_corrected = (corrected_poses.find(child_id) != corrected_poses.end());
            bool parent_corrected = (corrected_poses.find(parent_id) != corrected_poses.end());
            bool is_boundary = (child_corrected != parent_corrected);
            
            // Always print boundary information
            if (is_boundary) {
                std::cout << "FOUND BOUNDARY: KF " << parent_id << " -> KF " << child_id;
                if (parent_corrected) {
                    std::cout << " (Parent corrected, Child uncorrected)" << std::endl;
                } else {
                    std::cout << " (Parent uncorrected, Child corrected)" << std::endl;
                }
            }
            
                  
            // Setup the correct constraint depending on the region
            Eigen::Matrix4d T_parent_child = Eigen::Matrix4d::Identity();
            
            // CORRECTED CODE: Handle boundary constraints properly
            if (is_boundary) {
                // For boundary constraints, use pre-correction and original poses to determine the true constraint
                int corrected_id = parent_corrected ? parent_id : child_id;
                int uncorrected_id = parent_corrected ? child_id : parent_id;
                
                // Check if we have pre-correction data for the corrected keyframe
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
                nullptr, // No robust loss function for now
                optimized_poses[parent_id],
                optimized_poses[child_id]
            );
            
            // If this is a boundary constraint, we might want to add a positional constraint too
            if (is_boundary) {
                std::cout << "Added boundary constraint between KF " << parent_id 
                          << " and KF " << child_id << " with weight " << constraint_weight << std::endl;
            }
            
            spanning_tree_count++;
        }
    }
    
    std::cout << "\nAdded " << spanning_tree_count << " spanning tree constraints" << std::endl;
    
    // Configure the solver
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 20;
    options.function_tolerance = 1e-6;
    options.gradient_tolerance = 1e-10;
    options.parameter_tolerance = 1e-8;
    
    // Run the solver
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    
    std::cout << summary.BriefReport() << std::endl;
    
    // Print relative poses after optimization
    std::cout << "\nAfter optimization - Relative poses:" << std::endl;
    
    print_counter = 0; // Reset counter for after optimization
    
    for (auto& edge : spanning_tree) {
        int child_id = edge.first;
        int parent_id = edge.second;
        
        if (optimized_poses.find(child_id) != optimized_poses.end() && 
            optimized_poses.find(parent_id) != optimized_poses.end()) {
            
            bool child_corrected = (corrected_poses.find(child_id) != corrected_poses.end());
            bool parent_corrected = (corrected_poses.find(parent_id) != corrected_poses.end());
            bool is_boundary = (child_corrected != parent_corrected);
            
            // Print detailed pose information after optimization
            printDetailedRelativePose(
                parent_id, child_id, 
                optimized_poses[parent_id], optimized_poses[child_id],
                keyframe_poses, corrected_poses, non_corrected_poses,
                is_boundary,
                print_counter,
                false // after optimization
            );
            
            print_counter++;
        }
    }
    

    // Write the optimized results to a TUM format file (in Twc format for visualization)
    std::string output_file = "/Datasets/CERES_Work/output/optimized_poses.txt";
    std::ofstream out_file(output_file);
    
    if (!out_file.is_open()) {
        std::cerr << "Error: Could not open output file" << std::endl;
        return -1;
    }
    
    // First, collect all poses with timestamps
    std::vector<std::pair<double, int>> timestamps_kfids;
    for (const auto& kf_pair : keyframe_poses) {
        int kf_id = kf_pair.first;
        
        // Get the timestamp from keyframe_poses.txt
        std::ifstream ts_file(data_dir + "keyframe_poses.txt");
        std::string ts_line;
        std::getline(ts_file, ts_line); // Skip header
        
        int id;
        double ts;
        while (ts_file >> id >> ts) {
            if (id == kf_id) {
                timestamps_kfids.push_back(std::make_pair(ts, kf_id));
                break;
            }
            // Skip rest of line
            std::getline(ts_file, ts_line);
        }
    }
    
    // Sort by timestamp
    std::sort(timestamps_kfids.begin(), timestamps_kfids.end());
    
    // Write in TUM format: timestamp tx ty tz qx qy qz qw
    for (const auto& ts_kf : timestamps_kfids) {
        double timestamp = ts_kf.first;
        int kf_id = ts_kf.second;
        
        if (optimized_poses.find(kf_id) != optimized_poses.end()) {
            const double* tcw_pose = optimized_poses[kf_id];
            double twc_pose[7]; // Temporary storage for Twc pose
            
            // Convert from Tcw to Twc format using Sophus
            convertTcwToTwc(tcw_pose, twc_pose);
            
            // TUM format: timestamp tx ty tz qx qy qz qw
            out_file << std::fixed << std::setprecision(9) << timestamp << " "
                    << twc_pose[4] << " " << twc_pose[5] << " " << twc_pose[6] << " "
                    << twc_pose[0] << " " << twc_pose[1] << " " << twc_pose[2] << " " << twc_pose[3] << std::endl;
        }
    }
    
    out_file.close();
    
    std::cout << "Optimized poses saved to: " << output_file << " (in Twc format for visualization)" << std::endl;
    


        std::cout << "\n=== Comparing trajectories ===" << std::endl;
    // File paths
    std::string groundtruth_file = "/Datasets/CERES_Work/Vis_Result/standard_trajectory_with_loop.txt";
    
    // Read trajectories
    std::vector<PoseData> optimized_traj = readTrajectory(output_file);
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
