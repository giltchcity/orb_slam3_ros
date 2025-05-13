#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <set>
#include <string>
#include <cmath>
#include <mutex>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <ceres/ceres.h>

// Forward declarations
class KeyFrame;
class MapPoint;
class Map;

// Define SE3Pose type for convenience
typedef std::pair<Eigen::Quaterniond, Eigen::Vector3d> SE3Pose;

// Define KeyFrameAndPose type
typedef std::map<KeyFrame*, SE3Pose> KeyFrameAndPose;

// Simplified KeyFrame class
class KeyFrame {
public:
    unsigned long mnId;
    double mTimeStamp;
    bool mbFixedLinearizationPoint;
    bool mbBad;
    KeyFrame* mpParent;
    std::set<KeyFrame*> mspLoopEdges;
    std::vector<KeyFrame*> mvpOrderedConnectedKeyFrames;
    std::vector<int> mvOrderedWeights;
    bool bImu;
    KeyFrame* mPrevKF;
    KeyFrame* mNextKF;
    
    // Pose as rotation and translation separately
    Eigen::Matrix3f mRcw;  // Rotation
    Eigen::Vector3f mtcw;  // Translation

    // Methods
    bool isBad() const { return mbBad; }
    
    // Get/Set pose methods using Eigen directly
    void GetPose(Eigen::Matrix3f& R, Eigen::Vector3f& t) const {
        R = mRcw;
        t = mtcw;
    }
    
    void SetPose(const Eigen::Matrix3f& R, const Eigen::Vector3f& t) {
        mRcw = R;
        mtcw = t;
    }
    
    // Convert to rotation and translation
    void GetPoseInverse(Eigen::Matrix3f& Rwc, Eigen::Vector3f& twc) const {
        Rwc = mRcw.transpose();
        twc = -Rwc * mtcw;
    }
    
    KeyFrame* GetParent() { return mpParent; }
    bool hasChild(KeyFrame* /* pKF */) const { return false; }  // Simplified
    std::set<KeyFrame*> GetLoopEdges() const { return mspLoopEdges; }
    std::vector<KeyFrame*> GetCovisiblesByWeight(int /* minWeight */) const {
        return mvpOrderedConnectedKeyFrames;  // Simplified
    }
    
    int GetWeight(KeyFrame* pKF) const { 
        // Find pKF in mvpOrderedConnectedKeyFrames and return weight
        for(size_t i=0; i<mvpOrderedConnectedKeyFrames.size(); i++) {
            if(mvpOrderedConnectedKeyFrames[i] == pKF)
                return mvOrderedWeights[i];
        }
        return 0;
    }
};

// Simplified MapPoint class
class MapPoint {
public:
    unsigned long mnId;
    bool mbBad;
    Eigen::Vector3f mWorldPos;
    KeyFrame* mpRefKF;
    unsigned long mnCorrectedByKF;
    unsigned long mnCorrectedReference;

    // Methods
    bool isBad() const { return mbBad; }
    Eigen::Vector3f GetWorldPos() const { return mWorldPos; }
    void SetWorldPos(const Eigen::Vector3f& pos) { mWorldPos = pos; }
    KeyFrame* GetReferenceKeyFrame() { return mpRefKF; }
    void UpdateNormalAndDepth() { /* Implement if needed */ }
};

// Simplified Map class
class Map {
public:
    unsigned long mnId;
    unsigned long mnInitKFid;
    unsigned long mnMaxKFid;
    bool mbImuInitialized;
    std::mutex mMutexMapUpdate;

    // Methods
    int GetId() const { return mnId; }
    unsigned long GetInitKFid() const { return mnInitKFid; }
    unsigned long GetMaxKFid() const { return mnMaxKFid; }
    bool IsInertial() const { return false; }  // Simplified
    bool isImuInitialized() const { return mbImuInitialized; }
    void IncreaseChangeIndex() { /* Implement as needed */ }
    
    // Data access
    std::vector<KeyFrame*> GetAllKeyFrames() const { return mvpKeyFrames; }
    std::vector<MapPoint*> GetAllMapPoints() const { return mvpMapPoints; }
    int KeyFramesInMap() const { return mvpKeyFrames.size(); }
    int MapPointsInMap() const { return mvpMapPoints.size(); }

    // Set data
    void SetKeyFrames(const std::vector<KeyFrame*>& vpKFs) { mvpKeyFrames = vpKFs; }
    void SetMapPoints(const std::vector<MapPoint*>& vpMPs) { mvpMapPoints = vpMPs; }

private:
    std::vector<KeyFrame*> mvpKeyFrames;
    std::vector<MapPoint*> mvpMapPoints;
};

// Stabilized angle-axis conversion for better numerical stability
template <typename T>
void StabilizedRotationToMatrix(const T* angle_axis, T* R) {
    T angle = sqrt(angle_axis[0]*angle_axis[0] + 
                  angle_axis[1]*angle_axis[1] + 
                  angle_axis[2]*angle_axis[2]);
    
    // Handle small angles specially to avoid division by zero
    if (angle < T(1e-10)) {
        // For small angles, use a first-order approximation
        R[0] = T(1.0);
        R[1] = -angle_axis[2];
        R[2] = angle_axis[1];
        R[3] = angle_axis[2];
        R[4] = T(1.0);
        R[5] = -angle_axis[0];
        R[6] = -angle_axis[1];
        R[7] = angle_axis[0];
        R[8] = T(1.0);
        return;
    }
    
    T axis[3];
    axis[0] = angle_axis[0] / angle;
    axis[1] = angle_axis[1] / angle;
    axis[2] = angle_axis[2] / angle;
    
    T cos_angle = cos(angle);
    T sin_angle = sin(angle);
    T one_minus_cos = T(1.0) - cos_angle;
    
    // Standard Rodrigues formula
    R[0] = axis[0] * axis[0] * one_minus_cos + cos_angle;
    R[1] = axis[0] * axis[1] * one_minus_cos - axis[2] * sin_angle;
    R[2] = axis[0] * axis[2] * one_minus_cos + axis[1] * sin_angle;
    R[3] = axis[1] * axis[0] * one_minus_cos + axis[2] * sin_angle;
    R[4] = axis[1] * axis[1] * one_minus_cos + cos_angle;
    R[5] = axis[1] * axis[2] * one_minus_cos - axis[0] * sin_angle;
    R[6] = axis[2] * axis[0] * one_minus_cos - axis[1] * sin_angle;
    R[7] = axis[2] * axis[1] * one_minus_cos + axis[0] * sin_angle;
    R[8] = axis[2] * axis[2] * one_minus_cos + cos_angle;
}

// SE3 edge cost function for CERES
struct SE3EdgeCostFunction {
    SE3EdgeCostFunction(const Eigen::Matrix3d& R_meas, const Eigen::Vector3d& t_meas, double weight = 1.0) 
        : R_measurement(R_meas), t_measurement(t_meas), weight_(weight) {}
    
    template <typename T>
    bool operator()(const T* const pose_i, const T* const pose_j, T* residuals) const {
        // Extract parameters (angle-axis + translation)
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> rot_i(pose_i);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> trans_i(pose_i + 3);
        
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> rot_j(pose_j);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> trans_j(pose_j + 3);
        
        // Convert angle-axis to rotation matrices using the stabilized function
        T R_i[9], R_j[9];
        StabilizedRotationToMatrix(rot_i.data(), R_i);
        StabilizedRotationToMatrix(rot_j.data(), R_j);
        
        // Convert measurement to template type
        Eigen::Matrix<T, 3, 3> R_meas = R_measurement.cast<T>();
        Eigen::Matrix<T, 3, 1> t_meas = t_measurement.cast<T>();
        
        // Create rotation matrices from the array data
        Eigen::Matrix<T, 3, 3> R_i_mat, R_j_mat;
        R_i_mat << R_i[0], R_i[1], R_i[2], 
                   R_i[3], R_i[4], R_i[5], 
                   R_i[6], R_i[7], R_i[8];
        
        R_j_mat << R_j[0], R_j[1], R_j[2], 
                   R_j[3], R_j[4], R_j[5], 
                   R_j[6], R_j[7], R_j[8];
        
        // Calculate rotation error: log(R_j.transpose() * R_meas * R_i)
        Eigen::Matrix<T, 3, 3> R_error = R_j_mat.transpose() * R_meas * R_i_mat;
        
        // Convert rotation error to angle-axis
        Eigen::AngleAxis<T> aa_error(R_error);
        Eigen::Matrix<T, 3, 1> rot_error = aa_error.angle() * aa_error.axis();
        
        // Calculate translation error: R_j.transpose() * (t_meas + R_meas * t_i - t_j)
        Eigen::Matrix<T, 3, 1> t_error = R_j_mat.transpose() * (t_meas + R_meas * trans_i - trans_j);
        
        // Apply weight to residuals
        T sqrt_weight = sqrt(T(weight_));
        
        // Fill residuals
        residuals[0] = sqrt_weight * rot_error[0];
        residuals[1] = sqrt_weight * rot_error[1];
        residuals[2] = sqrt_weight * rot_error[2];
        residuals[3] = sqrt_weight * t_error[0];
        residuals[4] = sqrt_weight * t_error[1];
        residuals[5] = sqrt_weight * t_error[2];
        
        return true;
    }
    
    static ceres::CostFunction* Create(const Eigen::Matrix3d& R_meas, const Eigen::Vector3d& t_meas, double weight = 1.0) {
        return new ceres::AutoDiffCostFunction<SE3EdgeCostFunction, 6, 6, 6>(
            new SE3EdgeCostFunction(R_meas, t_meas, weight));
    }
    
    Eigen::Matrix3d R_measurement;
    Eigen::Vector3d t_measurement;
    double weight_;
};

// Custom manifold for SE3 in Ceres 2.2
class SE3Manifold : public ceres::Manifold {
public:
    virtual ~SE3Manifold() {}

    // Plus operation for SE3: x_plus_delta = Plus(x, delta)
    virtual bool Plus(const double* x, const double* delta, double* x_plus_delta) const {
        // x is [angle-axis (3), translation (3)]
        // delta is [angle-axis delta (3), translation delta (3)]
        // x_plus_delta is the result
        
        // Handle rotation (angle-axis)
        Eigen::Map<const Eigen::Vector3d> angleAxis(x);
        Eigen::Map<const Eigen::Vector3d> delta_angleAxis(delta);
        Eigen::Map<Eigen::Vector3d> result_angleAxis(x_plus_delta);
        
        // Convert to quaternions, multiply, and convert back to angle-axis
        Eigen::AngleAxisd aa1(angleAxis.norm(), angleAxis.normalized());
        Eigen::AngleAxisd aa2(delta_angleAxis.norm(), delta_angleAxis.normalized());
        
        Eigen::Quaterniond q1(aa1);
        Eigen::Quaterniond q2(aa2);
        Eigen::Quaterniond q_res = q2 * q1;
        
        Eigen::AngleAxisd aa_res(q_res);
        result_angleAxis = aa_res.angle() * aa_res.axis();
        
        // Handle translation
        Eigen::Map<const Eigen::Vector3d> translation(x + 3);
        Eigen::Map<const Eigen::Vector3d> delta_translation(delta + 3);
        Eigen::Map<Eigen::Vector3d> result_translation(x_plus_delta + 3);
        
        // Compute updated translation
        Eigen::Matrix3d R_delta = aa2.toRotationMatrix();
        result_translation = R_delta * translation + delta_translation;
        
        return true;
    }

    // PlusJacobian replaces ComputeJacobian in Ceres 2.2
    virtual bool PlusJacobian(const double* x, double* jacobian) const {
        // For SE3, Jacobian is 6x6
        Eigen::Map<Eigen::Matrix<double, 6, 6, Eigen::RowMajor>> J(jacobian);
        J.setZero();
        
        // Rotation block: identity for small changes
        J.block<3, 3>(0, 0).setIdentity();
        
        // Translation block: identity
        J.block<3, 3>(3, 3).setIdentity();
        
        // Cross-term: rotation affects translation
        Eigen::Map<const Eigen::Vector3d> angleAxis(x);
        Eigen::AngleAxisd aa(angleAxis.norm(), angleAxis.normalized());
        Eigen::Matrix3d R = aa.toRotationMatrix();
        
        Eigen::Map<const Eigen::Vector3d> t(x + 3);
        Eigen::Matrix3d skew;
        skew << 0, -t(2), t(1),
                t(2), 0, -t(0),
                -t(1), t(0), 0;
                
        J.block<3, 3>(3, 0) = -R * skew;
        
        return true;
    }
    
    // Minus operation for SE3: y_minus_x = Minus(y, x)
    virtual bool Minus(const double* y, const double* x, double* y_minus_x) const {
        // y and x are [angle-axis (3), translation (3)]
        // y_minus_x is the tangent space difference [rotation_diff (3), translation_diff (3)]
        
        // Convert to SE3 matrices
        Eigen::Map<const Eigen::Vector3d> x_aa(x);
        Eigen::Map<const Eigen::Vector3d> x_t(x + 3);
        Eigen::Map<const Eigen::Vector3d> y_aa(y);
        Eigen::Map<const Eigen::Vector3d> y_t(y + 3);
        
        // Convert angle-axis to rotation matrices
        Eigen::AngleAxisd x_rotation(x_aa.norm(), x_aa.normalized());
        Eigen::AngleAxisd y_rotation(y_aa.norm(), y_aa.normalized());
        
        Eigen::Matrix3d R_x = x_rotation.toRotationMatrix();
        Eigen::Matrix3d R_y = y_rotation.toRotationMatrix();
        
        // Compute the relative rotation R_rel = R_y * R_x^T
        Eigen::Matrix3d R_rel = R_y * R_x.transpose();
        
        // Convert back to angle-axis representation
        Eigen::AngleAxisd aa_rel(R_rel);
        Eigen::Map<Eigen::Vector3d> result_aa(y_minus_x);
        result_aa = aa_rel.angle() * aa_rel.axis();
        
        // Compute the relative translation: R_x^T * (y_t - x_t)
        Eigen::Map<Eigen::Vector3d> result_t(y_minus_x + 3);
        result_t = R_x.transpose() * (y_t - x_t);
        
        return true;
    }
    
    // MinusJacobian computes the derivative of Minus(y, x) with respect to x
    virtual bool MinusJacobian(const double* x, double* jacobian) const {
        // Compute the Jacobian of y_minus_x with respect to x
        Eigen::Map<Eigen::Matrix<double, 6, 6, Eigen::RowMajor>> J(jacobian);
        J.setZero();
        
        // For small perturbations, we can approximate with negative identity for rotations
        J.block<3, 3>(0, 0) = -Eigen::Matrix3d::Identity();
        
        // Convert angle-axis to rotation matrix
        Eigen::Map<const Eigen::Vector3d> x_aa(x);
        Eigen::AngleAxisd aa(x_aa.norm(), x_aa.normalized());
        Eigen::Matrix3d R_x = aa.toRotationMatrix();
        
        // For translation, we need -R_x^T
        J.block<3, 3>(3, 3) = -R_x.transpose();
        
        // Cross-term: Translation is affected by rotation
        Eigen::Map<const Eigen::Vector3d> t(x + 3);
        Eigen::Matrix3d skew;
        skew << 0, -t(2), t(1),
                t(2), 0, -t(0),
                -t(1), t(0), 0;
                
        // Note: This is an approximation for small angle changes
        J.block<3, 3>(3, 0) = R_x.transpose() * skew;
        
        return true;
    }

    virtual bool RightMultiplyByPlusJacobian(const double* x, 
                                           const int num_rows,
                                           const double* ambient_matrix,
                                           double* tangent_matrix) const {
        // This method computes tangent_matrix = ambient_matrix * plus_jacobian
        // where plus_jacobian is the Jacobian of Plus(x, delta) with respect to delta
        
        // We'll use the default implementation provided by ceres::Manifold
        double* plus_jacobian = new double[6 * 6];
        PlusJacobian(x, plus_jacobian);
        
        // Perform the matrix multiplication
        Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 6, Eigen::RowMajor>> 
            ambient(ambient_matrix, num_rows, 6);
        Eigen::Map<const Eigen::Matrix<double, 6, 6, Eigen::RowMajor>> 
            jacobian(plus_jacobian, 6, 6);
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 6, Eigen::RowMajor>> 
            tangent(tangent_matrix, num_rows, 6);
        
        tangent = ambient * jacobian;
        
        delete[] plus_jacobian;
        return true;
    }

    virtual int AmbientSize() const { return 6; }  // Renamed from GlobalSize in Ceres 2.2
    virtual int TangentSize() const { return 6; }  // Renamed from LocalSize in Ceres 2.2
};

// Function to save trajectory to a file
void SaveTrajectory(const std::string& filename, const std::vector<KeyFrame*>& vpKFs, 
                    const std::vector<double*>& vPoseParams = std::vector<double*>()) {
    std::ofstream f;
    f.open(filename.c_str());
    f << std::fixed;
    
    for(size_t i=0; i<vpKFs.size(); i++) {
        KeyFrame* pKF = vpKFs[i];
        if(pKF->isBad()) continue;
        
        if(!vPoseParams.empty() && vPoseParams[pKF->mnId]) {
            // Save from parameter blocks
            double* params = vPoseParams[pKF->mnId];
            
            // Convert angle-axis to rotation matrix
            Eigen::Vector3d rot(params[0], params[1], params[2]);
            Eigen::Vector3d trans(params[3], params[4], params[5]);
            
            double R[9];
            double angle = rot.norm();
            if(angle > 1e-10) {
                Eigen::Vector3d axis = rot / angle;
                Eigen::AngleAxisd aa(angle, axis);
                Eigen::Matrix3d Rcw = aa.toRotationMatrix();
                
                // Save as Twc (inverse of Tcw)
                Eigen::Matrix3d Rwc = Rcw.transpose();
                Eigen::Vector3d twc = -Rwc * trans;
                
                Eigen::Quaterniond q(Rwc);
                
                f << pKF->mnId << " " << pKF->mTimeStamp << " " 
                  << twc.x() << " " << twc.y() << " " << twc.z() << " "
                  << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
            } else {
                // Degenerate case - identity rotation
                Eigen::Vector3d twc = -trans;
                f << pKF->mnId << " " << pKF->mTimeStamp << " " 
                  << twc.x() << " " << twc.y() << " " << twc.z() << " "
                  << 0.0 << " " << 0.0 << " " << 0.0 << " " << 1.0 << std::endl;
            }
        } else {
            // Save directly from KeyFrame pose
            Eigen::Matrix3f Rwc;
            Eigen::Vector3f twc;
            pKF->GetPoseInverse(Rwc, twc);
            
            Eigen::Quaternionf q(Rwc);
            
            f << pKF->mnId << " " << pKF->mTimeStamp << " " 
              << twc.x() << " " << twc.y() << " " << twc.z() << " "
              << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
        }
    }
    
    f.close();
    std::cout << "Saved trajectory to " << filename << std::endl;
}

// OptimizeEssentialGraphCeres - Ceres 2.2 implementation of OptimizeEssentialGraph
void OptimizeEssentialGraphCeres(Map* pMap, KeyFrame* pLoopKF, KeyFrame* pCurKF,
                                 const KeyFrameAndPose& NonCorrectedSE3,
                                 const KeyFrameAndPose& CorrectedSE3,
                                 const std::map<KeyFrame*, std::set<KeyFrame*>>& LoopConnections,
                                 const bool& bFixScale) {
    
    std::cout << "Starting OptimizeEssentialGraphCeres with Ceres 2.2..." << std::endl;
    
    // Output paths for verification
    std::string outputDir = "/Datasets/CERES_Work/output/";
    std::string initialTrajectoryFile = outputDir + "step1_initial_trajectory.txt";
    std::string optimizedTrajectoryFile = outputDir + "step2_optimized_trajectory.txt";
    
    // Setup CERES optimizer - similar to g2o setup in original code
    ceres::Problem problem;
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 20; // Same as original code
    options.function_tolerance = 1e-6;
    options.initial_trust_region_radius = 1e-16; // Similar to g2o's lambda init
    
    const std::vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();
    const std::vector<MapPoint*> vpMPs = pMap->GetAllMapPoints();
    
    const unsigned int nMaxKFid = pMap->GetMaxKFid();
    
    // Create vectors to store rotations and translations - equivalent to vScw in original code
    std::vector<Eigen::Quaterniond> vRotations(nMaxKFid+1);
    std::vector<Eigen::Vector3d> vTranslations(nMaxKFid+1);
    std::vector<double*> vPoseParams(nMaxKFid+1, nullptr);  // Parameters for Ceres
    
    // Vector to store corrected poses after optimization (equivalent to vCorrectedSwc)
    std::vector<SE3Pose> vCorrectedPoses(nMaxKFid+1);
    
    const int minFeat = 100; // Minimum features for connections, same as original
    
    std::cout << "Creating parameter blocks for " << vpKFs.size() << " keyframes" << std::endl;
    
    // Create custom SE3 manifold for Ceres 2.2
    ceres::Manifold* se3_manifold = new SE3Manifold();
    
    // For each keyframe, add parameter block - similar to original code vertex creation
    for(KeyFrame* pKF : vpKFs) {
        if(pKF->isBad())
            continue;
        
        const int nIDi = pKF->mnId;
        double* pose_params = new double[6];  // 3 for rotation as angle-axis, 3 for translation
        
        // Try to get the pose from CorrectedSE3 first - exactly like original code
        KeyFrameAndPose::const_iterator it = CorrectedSE3.find(pKF);
        if(it != CorrectedSE3.end()) {
            // If found in CorrectedSE3, use that pose
            const SE3Pose& se3pose = it->second;
            Eigen::Quaterniond q = se3pose.first;
            Eigen::Vector3d t = se3pose.second;
            
            // Store in our equivalent of vScw
            vRotations[nIDi] = q;
            vTranslations[nIDi] = t;
            
            // Convert to angle-axis for Ceres
            Eigen::AngleAxisd aa(q);
            pose_params[0] = aa.angle() * aa.axis()[0];
            pose_params[1] = aa.angle() * aa.axis()[1];
            pose_params[2] = aa.angle() * aa.axis()[2];
            pose_params[3] = t[0];
            pose_params[4] = t[1];
            pose_params[5] = t[2];
            
            if(nIDi == pLoopKF->mnId || nIDi == pCurKF->mnId) {
                std::cout << "Important KF " << nIDi << " found in CorrectedSE3" << std::endl;
            }
        } else {
            // If not found, use current keyframe pose (exactly like original code)
            Eigen::Matrix3f Rcw;
            Eigen::Vector3f tcw;
            pKF->GetPose(Rcw, tcw);
            
            // Convert to double
            Eigen::Matrix3d Rcw_d = Rcw.cast<double>();
            Eigen::Vector3d tcw_d = tcw.cast<double>();
            
            // Create quaternion from rotation matrix
            Eigen::Quaterniond q(Rcw_d);
            q.normalize(); // Ensure normalized quaternion
            
            // Store in our vectors (equivalent to vScw)
            vRotations[nIDi] = q;
            vTranslations[nIDi] = tcw_d;
            
            // Convert to angle-axis for Ceres
            Eigen::AngleAxisd aa(q);
            pose_params[0] = aa.angle() * aa.axis()[0];
            pose_params[1] = aa.angle() * aa.axis()[1];
            pose_params[2] = aa.angle() * aa.axis()[2];
            pose_params[3] = tcw_d[0];
            pose_params[4] = tcw_d[1];
            pose_params[5] = tcw_d[2];
            
            if(pKF == pLoopKF || pKF == pCurKF) {
                std::cout << "Important KF " << nIDi << " (";
                if(pKF == pLoopKF) std::cout << "LoopKF";
                if(pKF == pCurKF) std::cout << "CurKF";
                std::cout << ") not in CorrectedSE3, using current pose" << std::endl;
            }
        }
        
        vPoseParams[nIDi] = pose_params;
        
        // Add parameter block with SE3 manifold
        problem.AddParameterBlock(pose_params, 6, se3_manifold);
        
        // Fix the initial keyframe (same as original code)
        if(pKF->mnId == pMap->GetInitKFid()) {
            problem.SetParameterBlockConstant(pose_params);
            std::cout << "Fixed initial keyframe: " << pKF->mnId << std::endl;
        }
    }
    
    // Save initial trajectory for verification
    SaveTrajectory(initialTrajectoryFile, vpKFs, vPoseParams);
    
    std::cout << "All parameter blocks created successfully" << std::endl;
    
    // The rest of the implementation will be completed later
    // This includes:
    // - Adding loop edges
    // - Adding spanning tree edges
    // - Adding existing loop edges
    // - Adding covisibility graph edges
    // - Running the optimization
    // - Updating poses and map points

    // As requested, we've completed approximately the first 100 lines of the function
    // The rest will be implemented based on your further directions
}

// Main function for testing
int main(int /* argc */, char** /* argv */) {
    std::string inputDir = "/Datasets/CERES_Work/input/optimization_data";
    
    // Create output directory if it doesn't exist
    std::string outputDir = "/Datasets/CERES_Work/output";
    system(("mkdir -p " + outputDir).c_str());
    
    std::cout << "Loading data from: " << inputDir << std::endl;
    
    // Load map info
    unsigned long nMaxKFid = 0;
    unsigned long nInitKFid = 0;
    
    std::ifstream fMapInfo(inputDir + "/map_info.txt");
    if(fMapInfo.is_open()) {
        std::string line;
        while(std::getline(fMapInfo, line)) {
            std::istringstream iss(line);
            std::string tag;
            iss >> tag;
            
            if(tag == "MAX_KF_ID") {
                iss >> nMaxKFid;
            } else if(tag == "INIT_KF_ID") {
                iss >> nInitKFid;
            }
        }
        fMapInfo.close();
    }
    
    std::cout << "Map info: MaxKFId = " << nMaxKFid << ", InitKFId = " << nInitKFid << std::endl;
    
    // Load keyframes
    std::vector<KeyFrame*> vpKFs;
    std::map<unsigned long, KeyFrame*> mapKFs;
    
    // Load keyframe basic info
    std::string kfsFile = inputDir + "/keyframes.txt";
    std::ifstream f(kfsFile);
    
    if(f.is_open()) {
        std::string line;
        // Skip header
        std::getline(f, line);
        
        int kfCount = 0;
        
        while(std::getline(f, line)) {
            std::istringstream iss(line);
            unsigned long id, parentId;
            int hasVelocity, isFixed, isBad, isInertial, isVirtual;
            
            iss >> id >> parentId >> hasVelocity >> isFixed >> isBad >> isInertial >> isVirtual;
            
            KeyFrame* pKF = new KeyFrame();
            pKF->mnId = id;
            pKF->bImu = (isInertial == 1);
            pKF->mbBad = (isBad == 1);
            
            mapKFs[id] = pKF;
            vpKFs.push_back(pKF);
            kfCount++;
        }
        
        f.close();
        std::cout << "Loaded " << kfCount << " keyframes" << std::endl;
    } else {
        std::cerr << "Cannot open file: " << kfsFile << std::endl;
        return -1;
    }
    
    // Load keyframe poses
    std::string posesFile = inputDir + "/keyframe_poses.txt";
    std::ifstream fPoses(posesFile);
    
    if(fPoses.is_open()) {
        std::string line;
        // Skip header
        std::getline(fPoses, line);
        
        int poseCount = 0;
        
        while(std::getline(fPoses, line)) {
            std::istringstream iss(line);
            unsigned long id;
            float tx, ty, tz, qx, qy, qz, qw;
            
            iss >> id >> tx >> ty >> tz >> qx >> qy >> qz >> qw;
            
            if(mapKFs.find(id) != mapKFs.end()) {
                KeyFrame* pKF = mapKFs[id];
                
                Eigen::Quaternionf q(qw, qx, qy, qz);
                Eigen::Matrix3f R = q.toRotationMatrix();
                Eigen::Vector3f t(tx, ty, tz);
                
                pKF->SetPose(R, t);
                poseCount++;
            }
        }
        
        fPoses.close();
        std::cout << "Loaded " << poseCount << " keyframe poses" << std::endl;
    } else {
        std::cerr << "Cannot open file: " << posesFile << std::endl;
        return -1;
    }
    
    // Load parent-child relationships
    std::string spanningFile = inputDir + "/spanning_tree.txt";
    std::ifstream fSpanning(spanningFile);
    
    if(fSpanning.is_open()) {
        std::string line;
        // Skip header
        std::getline(fSpanning, line);
        
        int spanningCount = 0;
        
        while(std::getline(fSpanning, line)) {
            std::istringstream iss(line);
            unsigned long childId, parentId;
            
            iss >> childId >> parentId;
            
            if(mapKFs.find(childId) != mapKFs.end() && mapKFs.find(parentId) != mapKFs.end()) {
                mapKFs[childId]->mpParent = mapKFs[parentId];
                spanningCount++;
            }
        }
        
        fSpanning.close();
        std::cout << "Loaded " << spanningCount << " spanning tree edges" << std::endl;
    }
    
    // Load loop edges
    std::string loopEdgesFile = inputDir + "/loop_edges.txt";
    std::ifstream fLoopEdges(loopEdgesFile);
    
    if(fLoopEdges.is_open()) {
        std::string line;
        // Skip header
        std::getline(fLoopEdges, line);
        
        int loopEdgeCount = 0;
        
        while(std::getline(fLoopEdges, line)) {
            std::istringstream iss(line);
            unsigned long id1, id2;
            
            iss >> id1 >> id2;
            
            if(mapKFs.find(id1) != mapKFs.end() && mapKFs.find(id2) != mapKFs.end()) {
                mapKFs[id1]->mspLoopEdges.insert(mapKFs[id2]);
                mapKFs[id2]->mspLoopEdges.insert(mapKFs[id1]);
                loopEdgeCount++;
            }
        }
        
        fLoopEdges.close();
        std::cout << "Loaded " << loopEdgeCount << " loop edges" << std::endl;
    }
    
    // Load covisibility graph
    std::string covisFile = inputDir + "/covisibility.txt";
    std::ifstream fCovis(covisFile);
    
    if(fCovis.is_open()) {
        std::string line;
        // Skip header
        std::getline(fCovis, line);
        
        int covisCount = 0;
        
        std::map<KeyFrame*, std::vector<KeyFrame*>> mapConnected;
        std::map<KeyFrame*, std::vector<int>> mapWeights;
        
        while(std::getline(fCovis, line)) {
            std::istringstream iss(line);
            unsigned long id1, id2;
            int weight;
            
            iss >> id1 >> id2 >> weight;
            
            if(mapKFs.find(id1) != mapKFs.end() && mapKFs.find(id2) != mapKFs.end()) {
                KeyFrame* pKF1 = mapKFs[id1];
                KeyFrame* pKF2 = mapKFs[id2];
                
                mapConnected[pKF1].push_back(pKF2);
                mapWeights[pKF1].push_back(weight);
                
                mapConnected[pKF2].push_back(pKF1);
                mapWeights[pKF2].push_back(weight);
                
                covisCount++;
            }
        }
        
        // Assign to KeyFrames
        for(auto& pair : mapConnected) {
            KeyFrame* pKF = pair.first;
            pKF->mvpOrderedConnectedKeyFrames = pair.second;
            pKF->mvOrderedWeights = mapWeights[pKF];
        }
        
        fCovis.close();
        std::cout << "Loaded " << covisCount << " covisibility edges" << std::endl;
    }
    
    // Load map points
    std::vector<MapPoint*> vpMPs;
    std::string mpsFile = inputDir + "/mappoints.txt";
    std::ifstream fMPs(mpsFile);
    
    if(fMPs.is_open()) {
        std::string line;
        // Skip header
        std::getline(fMPs, line);
        
        int mpCount = 0;
        
        while(std::getline(fMPs, line)) {
            std::istringstream iss(line);
            unsigned long id, refKFId, correctedByKF, correctedRef;
            float x, y, z;
            
            iss >> id >> x >> y >> z >> refKFId >> correctedByKF >> correctedRef;
            
            MapPoint* pMP = new MapPoint();
            pMP->mnId = id;
            pMP->mWorldPos = Eigen::Vector3f(x, y, z);
            pMP->mnCorrectedByKF = correctedByKF;
            pMP->mnCorrectedReference = correctedRef;
            
            if(mapKFs.find(refKFId) != mapKFs.end()) {
                pMP->mpRefKF = mapKFs.at(refKFId);
            }
            
            vpMPs.push_back(pMP);
            mpCount++;
        }
        
        fMPs.close();
        std::cout << "Loaded " << mpCount << " map points" << std::endl;
    }
    
    // Load SE3 data from Sim3 files
    KeyFrameAndPose NonCorrectedSE3;
    std::string ncseFile = inputDir + "/non_corrected_sim3.txt";
    std::ifstream fNCSE(ncseFile);
    
    if(fNCSE.is_open()) {
        std::string line;
        // Skip header
        std::getline(fNCSE, line);
        
        int ncseCount = 0;
        
        while(std::getline(fNCSE, line)) {
            std::istringstream iss(line);
            unsigned long id;
            double s, tx, ty, tz, qx, qy, qz, qw;
            
            iss >> id >> s >> tx >> ty >> tz >> qx >> qy >> qz >> qw;
            
            if(mapKFs.find(id) != mapKFs.end()) {
                KeyFrame* pKF = mapKFs.at(id);
                
                Eigen::Quaterniond q(qw, qx, qy, qz);
                q.normalize(); // Ensure normalized quaternion
                
                Eigen::Vector3d t(tx, ty, tz);
                // For fixed scale, we divide by scale
                if(s != 1.0) t = t / s;
                
                NonCorrectedSE3[pKF] = std::make_pair(q, t);
                ncseCount++;
            }
        }
        
        fNCSE.close();
        std::cout << "Loaded " << ncseCount << " NonCorrectedSE3 poses" << std::endl;
    }
    
    KeyFrameAndPose CorrectedSE3;
    std::string cseFile = inputDir + "/corrected_sim3.txt";
    std::ifstream fCSE(cseFile);
    
    if(fCSE.is_open()) {
        std::string line;
        // Skip header
        std::getline(fCSE, line);
        
        int cseCount = 0;
        
        while(std::getline(fCSE, line)) {
            std::istringstream iss(line);
            unsigned long id;
            double s, tx, ty, tz, qx, qy, qz, qw;
            
            iss >> id >> s >> tx >> ty >> tz >> qx >> qy >> qz >> qw;
            
            if(mapKFs.find(id) != mapKFs.end()) {
                KeyFrame* pKF = mapKFs.at(id);
                
                Eigen::Quaterniond q(qw, qx, qy, qz);
                q.normalize(); // Ensure normalized quaternion
                
                Eigen::Vector3d t(tx, ty, tz);
                // For fixed scale, we divide by scale
                if(s != 1.0) t = t / s;
                
                CorrectedSE3[pKF] = std::make_pair(q, t);
                cseCount++;
            }
        }
        
        fCSE.close();
        std::cout << "Loaded " << cseCount << " CorrectedSE3 poses" << std::endl;
    }
    
    // Load loop connections
    std::map<KeyFrame*, std::set<KeyFrame*>> LoopConnections;
    std::string loopConnFile = inputDir + "/loop_connections.txt";
    std::ifstream fLoopConn(loopConnFile);
    
    if(fLoopConn.is_open()) {
        std::string line;
        // Skip header
        std::getline(fLoopConn, line);
        
        int connCount = 0;
        
        while(std::getline(fLoopConn, line)) {
            std::istringstream iss(line);
            unsigned long id;
            iss >> id;
            
            if(mapKFs.find(id) == mapKFs.end())
                continue;
                
            KeyFrame* pKF = mapKFs.at(id);
            std::set<KeyFrame*> sConnected;
            
            unsigned long connId;
            while(iss >> connId) {
                if(mapKFs.find(connId) != mapKFs.end()) {
                    sConnected.insert(mapKFs.at(connId));
                    connCount++;
                }
            }
            
            LoopConnections[pKF] = sConnected;
        }
        
        fLoopConn.close();
        std::cout << "Loaded " << connCount << " loop connections" << std::endl;
    }
    
    // Load keyframe IDs for pLoopKF and pCurKF
    unsigned long loopKFId = 0, curKFId = 0;
    bool bFixScale = true;
    
    std::ifstream fKFIds(inputDir + "/keyframe_ids.txt");
    if(fKFIds.is_open()) {
        std::string line;
        while(std::getline(fKFIds, line)) {
            std::istringstream iss(line);
            std::string tag;
            iss >> tag;
            
            if(tag == "LOOP_KF_ID") {
                iss >> loopKFId;
            } else if(tag == "CURRENT_KF_ID") {
                iss >> curKFId;
            } else if(tag == "FIXED_SCALE") {
                int val;
                iss >> val;
                bFixScale = (val == 1);
            }
        }
        fKFIds.close();
    }
    
    std::cout << "Loop KF ID: " << loopKFId << ", Current KF ID: " << curKFId << ", Fixed Scale: " << bFixScale << std::endl;
    
    // Find pLoopKF and pCurKF
    KeyFrame* pLoopKF = nullptr;
    KeyFrame* pCurKF = nullptr;
    
    if(mapKFs.find(loopKFId) != mapKFs.end()) {
        pLoopKF = mapKFs[loopKFId];
    }
    
    if(mapKFs.find(curKFId) != mapKFs.end()) {
        pCurKF = mapKFs[curKFId];
    }
    
    if(!pLoopKF || !pCurKF) {
        std::cerr << "Cannot find pLoopKF or pCurKF" << std::endl;
        return -1;
    }
    
    // Create Map
    Map* pMap = new Map();
    pMap->mnId = 0;
    pMap->mnInitKFid = nInitKFid;
    pMap->mnMaxKFid = nMaxKFid;
    pMap->SetKeyFrames(vpKFs);
    pMap->SetMapPoints(vpMPs);
    
    // Run OptimizeEssentialGraph with CERES - but only the first 100 lines
    std::cout << "Starting OptimizeEssentialGraphCeres - First 100 lines only" << std::endl;
    OptimizeEssentialGraphCeres(pMap, pLoopKF, pCurKF, NonCorrectedSE3, CorrectedSE3, LoopConnections, bFixScale);
    
    // Clean up
    for(KeyFrame* pKF : vpKFs)
        delete pKF;
    
    for(MapPoint* pMP : vpMPs)
        delete pMP;
    
    delete pMap;
    
    return 0;
}
