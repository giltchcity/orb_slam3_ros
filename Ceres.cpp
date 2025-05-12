#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <set>
#include <string>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <sys/stat.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <ceres/ceres.h>

// Sim3 class that replicates g2o's behavior exactly
class Sim3 {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // Default constructor - identity transform
    Sim3() : 
        q_(Eigen::Quaterniond::Identity()),
        t_(Eigen::Vector3d::Zero()),
        s_(1.0) {}
    
    // Constructor from rotation, translation, and scale
    Sim3(const Eigen::Quaterniond& q, const Eigen::Vector3d& t, double s) : 
        q_(q), t_(t), s_(s) {
        q_.normalize();
    }
    
    // Get rotation as quaternion
    inline const Eigen::Quaterniond& quaternion() const { return q_; }
    
    // Get rotation as matrix
    inline Eigen::Matrix3d rotation() const { return q_.toRotationMatrix(); }
    
    // Get translation vector
    inline const Eigen::Vector3d& translation() const { return t_; }
    
    // Get scale factor
    inline double scale() const { return s_; }
    
    // Apply transformation to a 3D point (s * R * p + t)
    Eigen::Vector3d map(const Eigen::Vector3d& p) const {
        return s_ * (q_ * p) + t_;
    }
    
    // Inverse transformation
    Sim3 inverse() const {
        Eigen::Quaterniond q_inv = q_.conjugate();
        Eigen::Vector3d t_inv = -(q_inv * t_) / s_;
        return Sim3(q_inv, t_inv, 1.0/s_);
    }
    
    // Right multiplication operator - CRITICAL for matching g2o's behavior
    Sim3 operator*(const Sim3& other) const {
        Eigen::Quaterniond q_res = q_ * other.quaternion();
        Eigen::Vector3d t_res = s_ * (q_ * other.translation()) + t_;
        double s_res = s_ * other.scale();
        return Sim3(q_res, t_res, s_res);
    }

private:
    Eigen::Quaterniond q_;  // Rotation as quaternion
    Eigen::Vector3d t_;     // Translation vector
    double s_;              // Scale factor
};

// Data structures to store loaded data

// KeyFrame structure
struct KeyFrame {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    int id;                        // KeyFrame ID
    double timestamp;              // Timestamp
    Eigen::Quaterniond rotation;   // Rotation as quaternion
    Eigen::Vector3d position;      // Position
    
    // For the optimization
    bool is_bad = false;           // Is this a bad KF
    KeyFrame* parent = nullptr;    // Parent in spanning tree
    int parent_id = -1;            // Parent ID
    std::set<int> loop_edges;      // Loop edge connections
    std::map<int, int> covisible_keyframes; // KF_ID -> weight
    
    // For inertial support (if available)
    bool is_inertial = false;      // Is this an inertial KF
    KeyFrame* prev_kf = nullptr;   // Previous KF in inertial sequence
};

// MapPoint structure
struct MapPoint {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    int id;                        // MapPoint ID
    Eigen::Vector3d position;      // 3D position
    bool is_bad = false;           // Is this a bad point
    int reference_kf_id = -1;      // Reference KeyFrame ID
    int corrected_by_kf = -1;      // Corrected by KeyFrame ID
    int corrected_reference = -1;  // Corrected reference
};

// Custom Sim3 manifold for Ceres 2.2 - This is critical for matching g2o
class Sim3Manifold : public ceres::Manifold {
public:
    Sim3Manifold(bool fix_scale = false) : fix_scale_(fix_scale) {}
    
    virtual ~Sim3Manifold() {}
    
    // Parameter block format: [tx, ty, tz, qx, qy, qz, qw, s]
    virtual int AmbientSize() const { return 8; }
    
    // Dimension of the local tangent space
    virtual int TangentSize() const { return fix_scale_ ? 6 : 7; }
    
    // This is the critical function that must exactly match g2o's behavior
    virtual bool Plus(const double* x, const double* delta, double* x_plus_delta) const {
        // Extract the current parameters
        Eigen::Vector3d t(x[0], x[1], x[2]);
        Eigen::Quaterniond q(x[6], x[3], x[4], x[5]); // w, x, y, z order
        q.normalize();
        double s = x[7];
        
        // Create delta components
        Eigen::Map<const Eigen::Matrix<double, 7, 1>> delta_vec(delta);
        
        // Extract delta rotation (first 3 elements) and create quaternion
        Eigen::Vector3d omega = delta_vec.head<3>();
        double theta = omega.norm();
        Eigen::Quaterniond dq;
        
        if (theta > 1e-10) {
            omega = omega / theta; // normalize
            dq = Eigen::Quaterniond(cos(theta/2.0), 
                                    sin(theta/2.0) * omega.x(),
                                    sin(theta/2.0) * omega.y(), 
                                    sin(theta/2.0) * omega.z());
        } else {
            dq = Eigen::Quaterniond(1.0, 0.0, 0.0, 0.0);
        }
        
        // Extract delta translation (next 3 elements)
        Eigen::Vector3d dt = delta_vec.segment<3>(3);
        
        // Extract delta scale (last element) or use 0 if scale is fixed
        double ds = fix_scale_ ? 0.0 : delta_vec(6);
        
        // Apply right multiplication update - this is CRITICAL for matching g2o
        // Rotation update: q' = q * dq
        Eigen::Quaterniond q_plus = q * dq;
        q_plus.normalize();
        
        // Scale update: s' = s * exp(ds)
        double s_plus = s * exp(ds);
        
        // Translation update: t' = t + dt
        Eigen::Vector3d t_plus = t + dt;
        
        // Store the updated values
        x_plus_delta[0] = t_plus.x();
        x_plus_delta[1] = t_plus.y();
        x_plus_delta[2] = t_plus.z();
        x_plus_delta[3] = q_plus.x();
        x_plus_delta[4] = q_plus.y();
        x_plus_delta[5] = q_plus.z();
        x_plus_delta[6] = q_plus.w();
        x_plus_delta[7] = s_plus;
        
        return true;
    }
    
    // In Ceres 2.2, ComputeJacobian is renamed to PlusJacobian
    virtual bool PlusJacobian(const double* x, double* jacobian) const {
        // Set to zero
        std::memset(jacobian, 0, sizeof(double) * 8 * TangentSize());
        
        // Fill the Jacobian with identity blocks for the rotation and translation components
        Eigen::Map<Eigen::Matrix<double, 8, Eigen::Dynamic, Eigen::RowMajor>> J(
            jacobian, 8, TangentSize());
        
        // Rotation Jacobian (first 3 rows)
        J.block<3, 3>(3, 0) = Eigen::Matrix3d::Identity(); // dq/domega
        
        // Translation Jacobian (next 3 rows)
        J.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity(); // dt/ddt
        
        // Scale Jacobian (last row)
        if (!fix_scale_) {
            J(7, 6) = x[7]; // ds/ds = s
        }
        
        return true;
    }
    
    // Required by Ceres 2.2
    virtual bool Minus(const double* y, const double* x, double* delta) const {
        // This is not actually used in our optimization, but must be implemented
        // for the Manifold interface
        
        // Extract parameters
        Eigen::Vector3d t_x(x[0], x[1], x[2]);
        Eigen::Quaterniond q_x(x[6], x[3], x[4], x[5]); // w, x, y, z
        q_x.normalize();
        double s_x = x[7];
        
        Eigen::Vector3d t_y(y[0], y[1], y[2]);
        Eigen::Quaterniond q_y(y[6], y[3], y[4], y[5]); // w, x, y, z
        q_y.normalize();
        double s_y = y[7];
        
        // Compute delta rotation
        Eigen::Quaterniond q_delta = q_x.conjugate() * q_y;
        q_delta.normalize();
        
        // Convert to axis-angle
        Eigen::Vector3d omega;
        double angle = 2.0 * atan2(q_delta.vec().norm(), q_delta.w());
        
        if (q_delta.vec().norm() > 1e-10) {
            omega = angle * q_delta.vec() / q_delta.vec().norm();
        } else {
            omega.setZero();
        }
        
        // Compute delta translation
        Eigen::Vector3d dt = t_y - t_x;
        
        // Compute delta scale
        double ds = log(s_y / s_x);
        
        // Fill delta vector
        delta[0] = omega.x();
        delta[1] = omega.y();
        delta[2] = omega.z();
        delta[3] = dt.x();
        delta[4] = dt.y();
        delta[5] = dt.z();
        
        if (!fix_scale_) {
            delta[6] = ds;
        }
        
        return true;
    }
    
    // Required by Ceres 2.2
    virtual bool MinusJacobian(const double* x, double* jacobian) const {
        // This is not actually used in our optimization, but must be implemented
        // for the Manifold interface
        
        // For simplicity, we'll use the identity Jacobian
        std::memset(jacobian, 0, sizeof(double) * TangentSize() * 8);
        
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 8, Eigen::RowMajor>> J(
            jacobian, TangentSize(), 8);
        
        // First 3 rows: rotation
        J.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity();
        
        // Next 3 rows: translation
        J.block<3, 3>(3, 0) = Eigen::Matrix3d::Identity();
        
        // Last row: scale (if unfixed)
        if (!fix_scale_) {
            J(6, 7) = 1.0 / x[7]; // dds/ds = 1/s
        }
        
        return true;
    }

private:
    bool fix_scale_;  // Whether to keep scale fixed
};

// Sim3 error term that exactly matches g2o's EdgeSim3
struct Sim3Error {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Sim3Error(const Sim3& measurement, double information_factor = 1.0)
        : measurement_(measurement), information_factor_(information_factor) {}

    template <typename T>
    bool operator()(const T* const param_i, const T* const param_j, T* residuals) const {
        // Extract parameters
        // Sim3 Parameter block: [tx, ty, tz, qx, qy, qz, qw, s]
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> t_i(param_i);
        Eigen::Quaternion<T> q_i(param_i[6], param_i[3], param_i[4], param_i[5]); // w, x, y, z
        q_i.normalize();
        T s_i = param_i[7];
        
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> t_j(param_j);
        Eigen::Quaternion<T> q_j(param_j[6], param_j[3], param_j[4], param_j[5]); // w, x, y, z
        q_j.normalize();
        T s_j = param_j[7];
        
        // Convert measurement to T
        Eigen::Quaternion<T> q_meas(
            T(measurement_.quaternion().w()),
            T(measurement_.quaternion().x()),
            T(measurement_.quaternion().y()),
            T(measurement_.quaternion().z()));
        q_meas.normalize();
        
        Eigen::Matrix<T, 3, 1> t_meas(
            T(measurement_.translation().x()),
            T(measurement_.translation().y()),
            T(measurement_.translation().z()));
        
        T s_meas = T(measurement_.scale());
        
        // Compute the relative Sim3 transformation: Sji = Sj * Si^-1
        // First compute Si^-1
        Eigen::Quaternion<T> q_i_inv = q_i.conjugate();
        Eigen::Matrix<T, 3, 1> t_i_inv = -(q_i_inv * t_i) / s_i;
        T s_i_inv = T(1.0) / s_i;
        
        // Now compute Sji = Sj * Si^-1
        Eigen::Quaternion<T> q_ji = q_j * q_i_inv;
        q_ji.normalize();
        
        Eigen::Matrix<T, 3, 1> t_ji = s_j * (q_j * t_i_inv) + t_j;
        T s_ji = s_j * s_i_inv;
        
        // Compute the error between measurement and computed transformation
        // Rotation error: log(q_meas^-1 * q_ji)
        Eigen::Quaternion<T> q_error = q_meas.conjugate() * q_ji;
        q_error.normalize();
        
        // Convert quaternion error to axis-angle representation
        Eigen::Matrix<T, 3, 1> omega;
        T angle = T(2.0) * atan2(q_error.vec().norm(), q_error.w());
        
        if (q_error.vec().norm() > T(1e-10)) {
            omega = angle * q_error.vec() / q_error.vec().norm();
        } else {
            omega.setZero();
        }
        
        // Translation error: t_ji - t_meas
        Eigen::Matrix<T, 3, 1> t_error = t_ji - t_meas;
        
        // Scale error: log(s_ji / s_meas)
        T s_error = log(s_ji / s_meas);
        
        // Construct the 7D error vector [omega, t_error, s_error]
        residuals[0] = omega.x() * T(information_factor_);
        residuals[1] = omega.y() * T(information_factor_);
        residuals[2] = omega.z() * T(information_factor_);
        residuals[3] = t_error.x() * T(information_factor_);
        residuals[4] = t_error.y() * T(information_factor_);
        residuals[5] = t_error.z() * T(information_factor_);
        residuals[6] = s_error * T(information_factor_);
        
        return true;
    }

    static ceres::CostFunction* Create(const Sim3& measurement, double information_factor = 1.0) {
        return new ceres::AutoDiffCostFunction<Sim3Error, 7, 8, 8>(
            new Sim3Error(measurement, information_factor));
    }

private:
    Sim3 measurement_;
    double information_factor_;
};

class EssentialGraphOptimizer {
public:
    // Constructor
    EssentialGraphOptimizer() 
        : current_kf_id(0), loop_kf_id(1), init_kf_id(0) {}
    
    // Main loading functions
    
    // Load keyframe trajectory
    bool LoadTrajectory(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open trajectory file: " << filename << std::endl;
            return false;
        }
        
        keyframes_.clear();
        
        std::string line;
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;
            
            std::istringstream iss(line);
            double timestamp, tx, ty, tz, qx, qy, qz, qw;
            int id = -1;
            
            if (iss >> id >> timestamp >> tx >> ty >> tz >> qx >> qy >> qz >> qw) {
                // This format has ID as the first column
                KeyFrame kf;
                kf.id = id;
                kf.timestamp = timestamp;
                kf.position = Eigen::Vector3d(tx, ty, tz);
                kf.rotation = Eigen::Quaterniond(qw, qx, qy, qz).normalized();
                kf.is_bad = false;
                
                keyframes_[kf.id] = kf;
                if (kf.id > current_kf_id) {
                    current_kf_id = kf.id;
                }
            } else {
                // Try the other format with just timestamp
                iss.clear();
                iss.str(line);
                
                if (iss >> timestamp >> tx >> ty >> tz >> qx >> qy >> qz >> qw) {
                    KeyFrame kf;
                    kf.id = keyframes_.size();
                    kf.timestamp = timestamp;
                    kf.position = Eigen::Vector3d(tx, ty, tz);
                    kf.rotation = Eigen::Quaterniond(qw, qx, qy, qz).normalized();
                    kf.is_bad = false;
                    
                    keyframes_[kf.id] = kf;
                    if (kf.id > current_kf_id) {
                        current_kf_id = kf.id;
                    }
                }
            }
        }
        
        file.close();
        std::cout << "Loaded " << keyframes_.size() << " keyframes from " << filename << std::endl;
        return !keyframes_.empty();
    }
    
    // Load loop constraints
    bool LoadLoopConstraints(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open loop constraints file: " << filename << std::endl;
            return false;
        }
        
        loop_constraints_.clear();
        
        std::string line;
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;
            
            std::istringstream iss(line);
            int source_id, target_id;
            std::string constraint_type;
            double r00, r01, r02, r10, r11, r12, r20, r21, r22, t0, t1, t2, scale;
            
            if (iss >> source_id >> target_id >> constraint_type 
                   >> r00 >> r01 >> r02 >> r10 >> r11 >> r12 >> r20 >> r21 >> r22 
                   >> t0 >> t1 >> t2 >> scale) {
                
                Eigen::Matrix3d rot;
                rot << r00, r01, r02,
                       r10, r11, r12,
                       r20, r21, r22;
                       
                Eigen::Quaterniond q(rot);
                Eigen::Vector3d t(t0, t1, t2);
                
                Sim3 constraint(q, t, scale);
                
                // Store the constraint
                loop_constraints_.push_back(std::make_tuple(source_id, target_id, constraint));
                
                // Add to KeyFrame loop edges
                if (keyframes_.find(source_id) != keyframes_.end()) {
                    keyframes_[source_id].loop_edges.insert(target_id);
                }
                if (keyframes_.find(target_id) != keyframes_.end()) {
                    keyframes_[target_id].loop_edges.insert(source_id);
                }
                
                // Add to loop connections
                loop_connections_[source_id].insert(target_id);
            }
        }
        
        file.close();
        std::cout << "Loaded " << loop_constraints_.size() << " loop constraints from " << filename << std::endl;
        return !loop_constraints_.empty();
    }
    
    // Load essential graph
    bool LoadEssentialGraph(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open essential graph file: " << filename << std::endl;
            return false;
        }
        
        std::string line;
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;
            
            std::istringstream iss(line);
            std::string edge_type;
            int source_id, target_id;
            double weight, info_scale;
            
            if (iss >> edge_type >> source_id >> target_id >> weight >> info_scale) {
                if (edge_type == "SPANNING_TREE") {
                    // Set parent relationship
                    if (keyframes_.find(source_id) != keyframes_.end() && 
                        keyframes_.find(target_id) != keyframes_.end()) {
                        keyframes_[source_id].parent_id = target_id;
                        keyframes_[source_id].parent = &keyframes_[target_id];
                        
                        spanning_tree_edges_.push_back(std::make_pair(source_id, target_id));
                    }
                } else if (edge_type == "COVISIBILITY") {
                    // Add covisibility relationship
                    if (keyframes_.find(source_id) != keyframes_.end() && 
                        keyframes_.find(target_id) != keyframes_.end()) {
                        keyframes_[source_id].covisible_keyframes[target_id] = weight;
                        
                        covisibility_edges_.push_back(std::make_tuple(source_id, target_id, weight));
                    }
                } else if (edge_type == "LOOP") {
                    // Add loop relationship (already handled in LoadLoopConstraints)
                    if (keyframes_.find(source_id) != keyframes_.end() && 
                        keyframes_.find(target_id) != keyframes_.end()) {
                        keyframes_[source_id].loop_edges.insert(target_id);
                        keyframes_[target_id].loop_edges.insert(source_id);
                    }
                }
            }
        }
        
        file.close();
        std::cout << "Loaded essential graph with " << spanning_tree_edges_.size() 
                  << " spanning tree edges and " << covisibility_edges_.size() 
                  << " covisibility edges from " << filename << std::endl;
        return true;
    }
    
    // Load connection changes
    bool LoadConnectionChanges(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open connection changes file: " << filename << std::endl;
            return false;
        }
        
        std::string line;
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;
            
            std::istringstream iss(line);
            int source_id, target_id, is_new;
            
            if (iss >> source_id >> target_id >> is_new) {
                if (is_new == 1) {
                    connection_changes_.push_back(std::make_pair(source_id, target_id));
                    
                    // Add to loop connections
                    loop_connections_[source_id].insert(target_id);
                    loop_connections_[target_id].insert(source_id);
                }
            }
        }
        
        file.close();
        std::cout << "Loaded " << connection_changes_.size() << " connection changes from " << filename << std::endl;
        return true;
    }
    
    // Load manual connections file
    bool LoadConnections(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open connections file: " << filename << std::endl;
            return false;
        }
        
        std::string line;
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;
            
            std::istringstream iss(line);
            int source_id, target_id;
            double weight;
            int is_loop;
            
            if (iss >> source_id >> target_id >> weight >> is_loop) {
                if (is_loop == 1) {
                    // This is a loop edge
                    if (keyframes_.find(source_id) != keyframes_.end() && 
                        keyframes_.find(target_id) != keyframes_.end()) {
                        keyframes_[source_id].loop_edges.insert(target_id);
                        keyframes_[target_id].loop_edges.insert(source_id);
                        
                        // Add to loop connections
                        loop_connections_[source_id].insert(target_id);
                        loop_connections_[target_id].insert(source_id);
                    }
                } else {
                    // Regular covisibility
                    if (keyframes_.find(source_id) != keyframes_.end() && 
                        keyframes_.find(target_id) != keyframes_.end()) {
                        keyframes_[source_id].covisible_keyframes[target_id] = weight;
                        
                        covisibility_edges_.push_back(std::make_tuple(source_id, target_id, weight));
                    }
                }
            }
        }
        
        file.close();
        std::cout << "Loaded connections from " << filename << std::endl;
        return true;
    }
    
    // Load Sim3 transformations
    bool LoadSim3Transformations(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open Sim3 transforms file: " << filename << std::endl;
            return false;
        }
        
        std::string line;
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;
            
            std::istringstream iss(line);
            int kf_id;
            double orig_scale, orig_tx, orig_ty, orig_tz, orig_qx, orig_qy, orig_qz, orig_qw;
            double corr_scale, corr_tx, corr_ty, corr_tz, corr_qx, corr_qy, corr_qz, corr_qw;
            int propagation_source;
            
            if (iss >> kf_id 
                   >> orig_scale >> orig_tx >> orig_ty >> orig_tz >> orig_qx >> orig_qy >> orig_qz >> orig_qw 
                   >> corr_scale >> corr_tx >> corr_ty >> corr_tz >> corr_qx >> corr_qy >> corr_qz >> corr_qw 
                   >> propagation_source) {
                
                // Original Sim3
                Eigen::Quaterniond orig_q(orig_qw, orig_qx, orig_qy, orig_qz);
                Eigen::Vector3d orig_t(orig_tx, orig_ty, orig_tz);
                Sim3 orig_sim3(orig_q, orig_t, orig_scale);
                
                // Corrected Sim3
                Eigen::Quaterniond corr_q(corr_qw, corr_qx, corr_qy, corr_qz);
                Eigen::Vector3d corr_t(corr_tx, corr_ty, corr_tz);
                Sim3 corr_sim3(corr_q, corr_t, corr_scale);
                
                // Store
                non_corrected_sim3_[kf_id] = orig_sim3;
                corrected_sim3_[kf_id] = corr_sim3;
            }
        }
        
        file.close();
        std::cout << "Loaded " << non_corrected_sim3_.size() << " Sim3 transformations from " << filename << std::endl;
        return !non_corrected_sim3_.empty();
    }
    
    // Load map points
    bool LoadMapPoints(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open map points file: " << filename << std::endl;
            return false;
        }
        
        map_points_.clear();
        
        std::string line;
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;
            
            std::istringstream iss(line);
            int id, ref_kf_id;
            double x, y, z;
            int is_bad = 0;
            
            if (iss >> id >> x >> y >> z >> ref_kf_id >> is_bad) {
                MapPoint mp;
                mp.id = id;
                mp.position = Eigen::Vector3d(x, y, z);
                mp.reference_kf_id = ref_kf_id;
                mp.is_bad = (is_bad == 1);
                
                map_points_[id] = mp;
            }
        }
        
        file.close();
        std::cout << "Loaded " << map_points_.size() << " map points from " << filename << std::endl;
        return !map_points_.empty();
    }
    
    // Load loop info
    bool LoadLoopInfo(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open loop info file: " << filename << std::endl;
            return false;
        }
        
        std::string line;
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;
            
            std::istringstream iss(line);
            std::string key;
            std::string value;
            
            if (std::getline(iss, key, ':') && std::getline(iss, value)) {
                // Trim whitespace
                key.erase(0, key.find_first_not_of(" \t"));
                key.erase(key.find_last_not_of(" \t") + 1);
                value.erase(0, value.find_first_not_of(" \t"));
                value.erase(value.find_last_not_of(" \t") + 1);
                
                if (key == "CURRENT_KF_ID") {
                    current_kf_id = std::stoi(value);
                } else if (key == "MATCHED_KF_ID") {
                    loop_kf_id = std::stoi(value);
                } else if (key == "MAP_ID") {
                    // Map ID, usually 0
                } else if (key == "FIXED_SCALE") {
                    // Store in optimization parameters
                    optimization_params["FIXED_SCALE"] = value;
                }
            }
        }
        
        file.close();
        std::cout << "Loaded loop info: current_kf=" << current_kf_id << ", loop_kf=" << loop_kf_id << std::endl;
        return true;
    }
    
    // Load optimization parameters
    bool LoadOptimizationParams(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open optimization params file: " << filename << std::endl;
            return false;
        }
        
        std::string line;
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;
            
            std::istringstream iss(line);
            std::string key, value;
            
            if (std::getline(iss, key, ':') && std::getline(iss, value)) {
                // Trim whitespace
                key.erase(0, key.find_first_not_of(" \t"));
                key.erase(key.find_last_not_of(" \t") + 1);
                value.erase(0, value.find_first_not_of(" \t"));
                value.erase(value.find_last_not_of(" \t") + 1);
                
                optimization_params[key] = value;
            }
        }
        
        file.close();
        std::cout << "Loaded optimization parameters" << std::endl;
        return !optimization_params.empty();
    }
    
    // Optimize the essential graph - main function
    bool OptimizeEssentialGraph(const std::string& output_file) {
        // Determine if we should fix scale
        bool fix_scale = true;  // Default to fixing scale
        if (optimization_params.find("FIXED_SCALE") != optimization_params.end()) {
            fix_scale = (optimization_params["FIXED_SCALE"] == "true");
        }
        
        // Set up the Ceres problem
        ceres::Problem problem;
        
        // Create parameter blocks for all keyframes
        std::map<int, double*> sim3_blocks;
        std::map<int, Sim3> vScw;  // Original Sim3 (similar to ORB-SLAM3)
        std::map<int, Sim3> vCorrectedSwc;  // Corrected inverse Sim3 (for map point correction)
        
        // The minimum number of common features to create an edge
        const int minFeat = 100;  // Same as in ORB-SLAM3
        
        // Add keyframe vertices
        std::cout << "Setting up keyframe vertices..." << std::endl;
        for (const auto& kf_pair : keyframes_) {
            int id = kf_pair.first;
            const KeyFrame& kf = kf_pair.second;
            
            if (kf.is_bad) continue;
            
            // Parameter block format: [tx, ty, tz, qx, qy, qz, qw, s]
            double* sim3_block = new double[8];
            
            // Check if this keyframe has corrected Sim3
            auto it_corrected = corrected_sim3_.find(id);
            if (it_corrected != corrected_sim3_.end()) {
                // Use the corrected Sim3
                const Sim3& corrected_sim3 = it_corrected->second;
                
                sim3_block[0] = corrected_sim3.translation().x();
                sim3_block[1] = corrected_sim3.translation().y();
                sim3_block[2] = corrected_sim3.translation().z();
                sim3_block[3] = corrected_sim3.quaternion().x();
                sim3_block[4] = corrected_sim3.quaternion().y();
                sim3_block[5] = corrected_sim3.quaternion().z();
                sim3_block[6] = corrected_sim3.quaternion().w();
                sim3_block[7] = corrected_sim3.scale();
                
                vScw[id] = corrected_sim3;
            } else {
                // Use the current pose with scale = 1.0
                sim3_block[0] = kf.position.x();
                sim3_block[1] = kf.position.y();
                sim3_block[2] = kf.position.z();
                sim3_block[3] = kf.rotation.x();
                sim3_block[4] = kf.rotation.y();
                sim3_block[5] = kf.rotation.z();
                sim3_block[6] = kf.rotation.w();
                sim3_block[7] = 1.0;  // Scale = 1.0
                
                Sim3 Siw(kf.rotation, kf.position, 1.0);
                vScw[id] = Siw;
            }
            
            // Store the parameter block
            sim3_blocks[id] = sim3_block;
            
            // Add the parameter block to the problem with appropriate manifold
            ceres::Manifold* manifold = new Sim3Manifold(fix_scale);
            problem.AddParameterBlock(sim3_block, 8, manifold);
            
            // Fix the initial keyframe
            if (id == init_kf_id || id == loop_kf_id) {
                problem.SetParameterBlockConstant(sim3_block);
            }
        }
        
        // Set to track inserted edges and avoid duplicates
        std::set<std::pair<int, int>> inserted_edges;
        
        // Add loop closure edges (from LoopConnections)
        std::cout << "Adding loop closure edges..." << std::endl;
        int count_loop = 0;
        for (const auto& entry : loop_connections_) {
            int source_id = entry.first;
            const std::set<int>& connected_kfs = entry.second;
            
            if (sim3_blocks.find(source_id) == sim3_blocks.end()) continue;
            
            const Sim3& Siw = vScw[source_id];
            const Sim3 Swi = Siw.inverse();
            
            for (int target_id : connected_kfs) {
                if (sim3_blocks.find(target_id) == sim3_blocks.end()) continue;
                
                // Skip if already processed (both directions)
                std::pair<int, int> edge_pair(std::min(source_id, target_id), std::max(source_id, target_id));
                if (inserted_edges.count(edge_pair)) continue;
                
                // Get the weight from covisibility graph if available
                int weight = 0;
                if (keyframes_.find(source_id) != keyframes_.end() &&
                    keyframes_[source_id].covisible_keyframes.find(target_id) != 
                    keyframes_[source_id].covisible_keyframes.end()) {
                    weight = keyframes_[source_id].covisible_keyframes[target_id];
                }
                
                // Skip low weight edges unless it's the current loop closure
                if ((source_id != current_kf_id || target_id != loop_kf_id) && weight < minFeat) continue;
                
                const Sim3& Sjw = vScw[target_id];
                const Sim3 Sji = Sjw * Swi;
                
                // Create cost function
                ceres::CostFunction* cost_function = Sim3Error::Create(Sji, 1.0);
                
                // Add robust loss function
                ceres::LossFunction* loss_function = new ceres::HuberLoss(1.345);
                
                // Add to problem
                problem.AddResidualBlock(
                    cost_function,
                    loss_function,
                    sim3_blocks[source_id],
                    sim3_blocks[target_id]);
                
                // Mark as processed
                inserted_edges.insert(edge_pair);
                count_loop++;
            }
        }
        std::cout << "Added " << count_loop << " loop edges" << std::endl;
        
        // Add spanning tree edges
        std::cout << "Adding spanning tree edges..." << std::endl;
        int count_spanning = 0;
        for (const auto& kf_pair : keyframes_) {
            int source_id = kf_pair.first;
            const KeyFrame& kf = kf_pair.second;
            
            if (kf.is_bad || kf.parent_id < 0) continue;
            
            int target_id = kf.parent_id;
            
            if (sim3_blocks.find(source_id) == sim3_blocks.end() || 
                sim3_blocks.find(target_id) == sim3_blocks.end()) continue;
            
            // Skip if already processed
            std::pair<int, int> edge_pair(std::min(source_id, target_id), std::max(source_id, target_id));
            if (inserted_edges.count(edge_pair)) continue;
            
            // Compute the relative Sim3
            const Sim3& Siw = vScw[source_id];
            const Sim3& Sjw = vScw[target_id];
            const Sim3 Sji = Sjw * Siw.inverse();
            
            // Create cost function
            ceres::CostFunction* cost_function = Sim3Error::Create(Sji, 1.0);
            
            // Add robust loss function
            ceres::LossFunction* loss_function = new ceres::HuberLoss(1.345);
            
            // Add to problem
            problem.AddResidualBlock(
                cost_function,
                loss_function,
                sim3_blocks[source_id],
                sim3_blocks[target_id]);
            
            // Mark as processed
            inserted_edges.insert(edge_pair);
            count_spanning++;
        }
        std::cout << "Added " << count_spanning << " spanning tree edges" << std::endl;
        
        // Add existing loop edges (previously detected loops)
        std::cout << "Adding existing loop edges..." << std::endl;
        int count_existing_loop = 0;
        for (const auto& kf_pair : keyframes_) {
            int source_id = kf_pair.first;
            const KeyFrame& kf = kf_pair.second;
            
            if (kf.is_bad) continue;
            
            for (int target_id : kf.loop_edges) {
                // Only process once
                if (target_id >= source_id) continue;
                
                if (sim3_blocks.find(source_id) == sim3_blocks.end() || 
                    sim3_blocks.find(target_id) == sim3_blocks.end()) continue;
                
                // Skip if already processed
                std::pair<int, int> edge_pair(std::min(source_id, target_id), std::max(source_id, target_id));
                if (inserted_edges.count(edge_pair)) continue;
                
                // Compute the relative Sim3
                const Sim3& Siw = vScw[source_id];
                const Sim3& Sjw = vScw[target_id];
                const Sim3 Sji = Sjw * Siw.inverse();
                
                // Create cost function
                ceres::CostFunction* cost_function = Sim3Error::Create(Sji, 1.0);
                
                // Add robust loss function
                ceres::LossFunction* loss_function = new ceres::HuberLoss(1.345);
                
                // Add to problem
                problem.AddResidualBlock(
                    cost_function,
                    loss_function,
                    sim3_blocks[source_id],
                    sim3_blocks[target_id]);
                
                // Mark as processed
                inserted_edges.insert(edge_pair);
                count_existing_loop++;
            }
        }
        std::cout << "Added " << count_existing_loop << " existing loop edges" << std::endl;
        
        // Add covisibility edges
        std::cout << "Adding covisibility edges..." << std::endl;
        int count_covisibility = 0;
        for (const auto& kf_pair : keyframes_) {
            int source_id = kf_pair.first;
            const KeyFrame& kf = kf_pair.second;
            
            if (kf.is_bad) continue;
            
            // Process covisible keyframes
            for (const auto& covis_pair : kf.covisible_keyframes) {
                int target_id = covis_pair.first;
                int weight = covis_pair.second;
                
                // Only process in one direction and skip parent
                if (target_id <= source_id || target_id == kf.parent_id) continue;
                
                // Skip if already in loop edges
                if (kf.loop_edges.count(target_id)) continue;
                
                if (sim3_blocks.find(source_id) == sim3_blocks.end() || 
                    sim3_blocks.find(target_id) == sim3_blocks.end()) continue;
                
                // Skip if already processed
                std::pair<int, int> edge_pair(source_id, target_id);
                if (inserted_edges.count(edge_pair)) continue;
                
                // Skip if weight too low
                if (weight < minFeat) continue;
                
                // Compute the relative Sim3
                const Sim3& Siw = vScw[source_id];
                const Sim3& Sjw = vScw[target_id];
                const Sim3 Sji = Sjw * Siw.inverse();
                
                // Create cost function
                ceres::CostFunction* cost_function = Sim3Error::Create(Sji, 1.0);
                
                // Add robust loss function
                ceres::LossFunction* loss_function = new ceres::HuberLoss(1.345);
                
                // Add to problem
                problem.AddResidualBlock(
                    cost_function,
                    loss_function,
                    sim3_blocks[source_id],
                    sim3_blocks[target_id]);
                
                // Mark as processed
                inserted_edges.insert(edge_pair);
                count_covisibility++;
            }
        }
        std::cout << "Added " << count_covisibility << " covisibility edges" << std::endl;
        
        // Set solver options
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        options.minimizer_progress_to_stdout = true;
        options.num_threads = 8;
        
        // Get max iterations from optimization parameters, default to 20
        options.max_num_iterations = 20;
        if (optimization_params.find("NUM_ITERATIONS") != optimization_params.end()) {
            options.max_num_iterations = std::stoi(optimization_params["NUM_ITERATIONS"]);
        }
        
        // Critical: Set initial trust region radius to match g2o's lambda
        options.initial_trust_region_radius = 1e-16;
        
        // Solve the optimization problem
        std::cout << "Optimizing..." << std::endl;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        
        std::cout << summary.BriefReport() << std::endl;
        
        // Update keyframes with the optimized values
        std::cout << "Updating keyframes and map points..." << std::endl;
        for (auto& kf_pair : keyframes_) {
            int id = kf_pair.first;
            KeyFrame& kf = kf_pair.second;
            
            if (sim3_blocks.find(id) == sim3_blocks.end()) continue;
            
            // Get the optimized Sim3
            const double* sim3_block = sim3_blocks[id];
            
            // Extract components
            Eigen::Vector3d t(sim3_block[0], sim3_block[1], sim3_block[2]);
            Eigen::Quaterniond q(sim3_block[6], sim3_block[3], sim3_block[4], sim3_block[5]); // w, x, y, z
            q.normalize();
            double s = sim3_block[7];
            
            // Store the corrected inverse Sim3 for map point correction
            Sim3 CorrectedSiw(q, t, s);
            vCorrectedSwc[id] = CorrectedSiw.inverse();
            
            // Update keyframe pose - convert to SE3 by dividing translation by scale
            kf.rotation = q;
            kf.position = t / s;
        }
        
        // Update map points
        for (auto& mp_pair : map_points_) {
            MapPoint& mp = mp_pair.second;
            
            if (mp.is_bad) continue;
            
            // Get reference keyframe
            int ref_kf_id = mp.reference_kf_id;
            
            // Check if this point was corrected by the current KF
            int nIDr;
            if (mp.corrected_by_kf == current_kf_id) {
                nIDr = mp.corrected_reference;
            } else {
                nIDr = ref_kf_id;
            }
            
            // Skip if ref KF not found in sim3 maps
            if (vScw.find(nIDr) == vScw.end() || vCorrectedSwc.find(nIDr) == vCorrectedSwc.end()) continue;
            
            // Get original and corrected Sim3 transforms
            Sim3 Srw = vScw[nIDr];
            Sim3 correctedSwr = vCorrectedSwc[nIDr];
            
            // Transform the point: world -> reference KF -> corrected world
            Eigen::Vector3d eigP3Dw = mp.position;
            Eigen::Vector3d eigCorrectedP3Dw = correctedSwr.map(Srw.map(eigP3Dw));
            
            // Update map point position
            mp.position = eigCorrectedP3Dw;
        }
        
        // Save the optimized trajectory
        SaveOptimizedData(output_file);
        
        // Clean up
        for (auto& block_pair : sim3_blocks) {
            delete[] block_pair.second;
        }
        
        std::cout << "Essential Graph optimization completed successfully!" << std::endl;
        return true;
    }
    
    // Save optimized data
    void SaveOptimizedData(const std::string& output_file) {
        // Save trajectory file
        std::ofstream file(output_file);
        if (!file.is_open()) {
            std::cerr << "Failed to open output file: " << output_file << std::endl;
            return;
        }
        
        file << "# id timestamp tx ty tz qx qy qz qw" << std::endl;
        
        // Sort keyframes by ID for consistency
        std::vector<int> sorted_ids;
        for (const auto& kf_pair : keyframes_) {
            if (!kf_pair.second.is_bad) {
                sorted_ids.push_back(kf_pair.first);
            }
        }
        
        std::sort(sorted_ids.begin(), sorted_ids.end());
        
        for (int id : sorted_ids) {
            const KeyFrame& kf = keyframes_[id];
            
            file << id << " " 
                 << std::fixed << std::setprecision(9) << kf.timestamp << " "
                 << std::fixed << std::setprecision(6)
                 << kf.position.x() << " " 
                 << kf.position.y() << " " 
                 << kf.position.z() << " "
                 << kf.rotation.x() << " " 
                 << kf.rotation.y() << " " 
                 << kf.rotation.z() << " " 
                 << kf.rotation.w() << std::endl;
        }
        
        file.close();
        std::cout << "Saved optimized trajectory to: " << output_file << std::endl;
        
        // Save map points if available
        if (!map_points_.empty()) {
            std::string mp_file = output_file.substr(0, output_file.find_last_of('.')) + "_mappoints.txt";
            std::ofstream mp_out(mp_file);
            
            if (mp_out.is_open()) {
                mp_out << "# id x y z ref_kf_id is_bad" << std::endl;
                
                for (const auto& mp_pair : map_points_) {
                    const MapPoint& mp = mp_pair.second;
                    
                    if (!mp.is_bad) {
                        mp_out << mp.id << " " 
                               << std::fixed << std::setprecision(6)
                               << mp.position.x() << " " 
                               << mp.position.y() << " " 
                               << mp.position.z() << " "
                               << mp.reference_kf_id << " " 
                               << "0" << std::endl;
                    }
                }
                
                mp_out.close();
                std::cout << "Saved optimized map points to: " << mp_file << std::endl;
            }
        }
    }
    
    // Data storage
    std::map<int, KeyFrame> keyframes_;
    std::map<int, MapPoint> map_points_;
    std::map<int, Sim3> non_corrected_sim3_;
    std::map<int, Sim3> corrected_sim3_;
    std::map<int, std::set<int>> loop_connections_;
    std::map<std::string, std::string> optimization_params;
    
    std::vector<std::tuple<int, int, Sim3>> loop_constraints_;
    std::vector<std::pair<int, int>> spanning_tree_edges_;
    std::vector<std::tuple<int, int, double>> covisibility_edges_;
    std::vector<std::pair<int, int>> connection_changes_;
    
    // Current state
    int current_kf_id;
    int loop_kf_id;
    int init_kf_id;
};

int main(int argc, char** argv) {
    // Set default paths
    std::string input_dir = "/Datasets/CERES_Work/input";
    std::string output_file = "/Datasets/CERES_Work/output/optimized_trajectory.txt";
    
    // Override if provided
    if (argc > 1) input_dir = argv[1];
    if (argc > 2) output_file = argv[2];
    
    // Create output directory if it doesn't exist
    std::string output_dir = output_file.substr(0, output_file.find_last_of('/'));
    struct stat st = {0};
    if (stat(output_dir.c_str(), &st) == -1) {
        mkdir(output_dir.c_str(), 0755);
    }
    
    // Create the optimizer
    EssentialGraphOptimizer optimizer;
    
    // Load input data
    std::cout << "Loading input data..." << std::endl;
    
    // Load Sim3 transformed trajectory
    if (!optimizer.LoadTrajectory(input_dir + "/sim3_transformed_trajectory.txt")) {
        std::cerr << "Failed to load trajectory!" << std::endl;
        return 1;
    }
    
    // Load Essential Graph
    optimizer.LoadEssentialGraph(input_dir + "/pre/essential_graph.txt");
    
    // Load Loop Constraints
    optimizer.LoadLoopConstraints(input_dir + "/metadata/loop_constraints.txt");
    
    // Load Connection Changes
    optimizer.LoadConnectionChanges(input_dir + "/metadata/connection_changes.txt");
    
    // Load Sim3 Transformations
    optimizer.LoadSim3Transformations(input_dir + "/metadata/sim3_transforms.txt");
    
    // Load Loop Info
    optimizer.LoadLoopInfo(input_dir + "/metadata/loop_info.txt");
    
    // Load Optimization Parameters
    optimizer.LoadOptimizationParams(input_dir + "/metadata/optimization_params.txt");
    
    // Load Connections (if available)
    try {
        optimizer.LoadConnections(input_dir + "/connections.txt");
    } catch (const std::exception& e) {
        std::cout << "No connections.txt file available (this is optional)" << std::endl;
    }
    
    // Try to load MapPoints (if available)
    try {
        optimizer.LoadMapPoints(input_dir + "/pre/mappoints_no_loop.txt");
    } catch (const std::exception& e) {
        std::cout << "No map points file available (this is optional)" << std::endl;
    }
    
    // Run the optimization
    std::cout << "Starting Essential Graph optimization..." << std::endl;
    if (!optimizer.OptimizeEssentialGraph(output_file)) {
        std::cerr << "Optimization failed!" << std::endl;
        return 1;
    }
    
    std::cout << "Essential Graph optimization completed successfully!" << std::endl;
    std::cout << "Optimized trajectory saved to: " << output_file << std::endl;
    
    return 0;
}
