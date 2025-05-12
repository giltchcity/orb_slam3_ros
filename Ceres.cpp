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
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <ceres/ceres.h>

// g2o::Sim3 equivalent for Ceres
class Sim3 {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    Sim3() : r_(1.0, 0.0, 0.0, 0.0), t_(0.0, 0.0, 0.0), s_(1.0) {}
    
    Sim3(const Eigen::Quaterniond& r, const Eigen::Vector3d& t, double s) : r_(r), t_(t), s_(s) {
        r_.normalize();
    }
    
    // Inverse of the Sim3 transformation
    Sim3 inverse() const {
        Eigen::Quaterniond r_inv = r_.conjugate();
        Eigen::Vector3d t_inv = -(r_inv * t_) / s_;
        return Sim3(r_inv, t_inv, 1.0/s_);
    }
    
    // Compose two Sim3 transformations
    Sim3 operator*(const Sim3& other) const {
        Eigen::Quaterniond r_result = r_ * other.r_;
        Eigen::Vector3d t_result = s_ * (r_ * other.t_) + t_;
        double s_result = s_ * other.s_;
        return Sim3(r_result, t_result, s_result);
    }
    
    // Transform a 3D point
    Eigen::Vector3d map(const Eigen::Vector3d& p) const {
        return s_ * (r_ * p) + t_;
    }
    
    // Accessors
    const Eigen::Quaterniond& rotation() const { return r_; }
    const Eigen::Vector3d& translation() const { return t_; }
    double scale() const { return s_; }
    
private:
    Eigen::Quaterniond r_;
    Eigen::Vector3d t_;
    double s_;
};

// Define KeyFrame structure
struct KeyFrame {
    int id;
    double timestamp;
    Eigen::Quaterniond rotation;
    Eigen::Vector3d position;
    int parent_id;
    bool is_bad;
    std::set<int> loop_edges;
    std::map<int, int> covisible_keyframes; // KF_ID -> weight
};

// MapPoint structure
struct MapPoint {
    int id;
    Eigen::Vector3d position;
    bool is_bad;
    int reference_kf_id;
    int corrected_by_kf;
    int corrected_reference;
};

// Edge structure for optimization
struct Edge {
    enum Type {
        SPANNING_TREE,
        LOOP,
        COVISIBILITY
    };
    
    Type type;
    int source_id;
    int target_id;
    double weight;
    
    // Relative pose constraint
    Eigen::Quaterniond rel_rotation;
    Eigen::Vector3d rel_translation;
    double rel_scale;
};

// 修改Sim3Parameterization类，使用现代Ceres 2.2 API
class Sim3Parameterization : public ceres::Manifold {
public:
    Sim3Parameterization(bool fix_scale = false) : fix_scale_(fix_scale) {}

    // Sim3有8个环境参数：[s, qw, qx, qy, qz, tx, ty, tz]
    virtual int AmbientSize() const override { return 8; }
    
    // 如果scale固定，则有6个自由度，否则有7个
    virtual int TangentSize() const override { return fix_scale_ ? 6 : 7; }

    // Plus操作：将更新量delta应用到当前参数x
    virtual bool Plus(const double* x, const double* delta, double* x_plus_delta) const override {
        // 提取当前参数
        double scale = x[0];
        Eigen::Quaterniond rotation(x[1], x[2], x[3], x[4]);  // w, x, y, z
        Eigen::Vector3d translation(x[5], x[6], x[7]);

        // 首先更新缩放因子（如果未固定）
        double new_scale = scale;
        int delta_idx = 0;
        
        if (!fix_scale_) {
            // 应用乘法缩放更新：s_new = s * exp(delta[0])
            new_scale = scale * exp(delta[delta_idx++]);
        }

        // 使用指数映射更新旋转
        Eigen::Vector3d omega(delta[delta_idx], delta[delta_idx+1], delta[delta_idx+2]);
        delta_idx += 3;
        
        Eigen::Quaterniond dq;
        if (omega.norm() < 1e-10) {
            dq = Eigen::Quaterniond(1.0, 0.0, 0.0, 0.0);
        } else {
            double theta = omega.norm();
            omega.normalize();
            dq = Eigen::Quaterniond(cos(theta/2.0), sin(theta/2.0)*omega.x(),
                                   sin(theta/2.0)*omega.y(), sin(theta/2.0)*omega.z());
        }
        
        Eigen::Quaterniond new_rotation = dq * rotation;
        new_rotation.normalize();

        // 更新平移
        Eigen::Vector3d new_translation = translation + Eigen::Vector3d(delta[delta_idx], 
                                                                      delta[delta_idx+1], 
                                                                      delta[delta_idx+2]);

        // 打包更新后的参数
        x_plus_delta[0] = new_scale;
        x_plus_delta[1] = new_rotation.w();
        x_plus_delta[2] = new_rotation.x();
        x_plus_delta[3] = new_rotation.y();
        x_plus_delta[4] = new_rotation.z();
        x_plus_delta[5] = new_translation.x();
        x_plus_delta[6] = new_translation.y();
        x_plus_delta[7] = new_translation.z();

        return true;
    }

    // Manifold接口需要实现VerifyJacobian，但我们可以不实现
    // Ceres将使用数值微分来计算雅可比矩阵

    // Ceres 2.x不再需要ComputeJacobian方法

private:
    bool fix_scale_;  // 如果为true，则缩放固定（6DoF），否则为7DoF
};

// Sim3 error class for Ceres that matches the g2o::EdgeSim3 approach
struct Sim3Error {
    Sim3Error(const Sim3& measurement, double information_scale = 1.0)
        : m_measurement(measurement), m_information_scale(information_scale) {}

    template <typename T>
    bool operator()(const T* const sim3_i_data, const T* const sim3_j_data, T* residuals) const {
        // Each sim3 parameter block is organized as:
        // [scale, rotation(quaternion), translation]
        // [s, qw, qx, qy, qz, tx, ty, tz]

        // Extract parameters for vertex i
        const T scale_i = sim3_i_data[0];
        Eigen::Quaternion<T> rotation_i(sim3_i_data[1], sim3_i_data[2], sim3_i_data[3], sim3_i_data[4]);
        Eigen::Matrix<T, 3, 1> translation_i(sim3_i_data[5], sim3_i_data[6], sim3_i_data[7]);
        
        // Extract parameters for vertex j
        const T scale_j = sim3_j_data[0];
        Eigen::Quaternion<T> rotation_j(sim3_j_data[1], sim3_j_data[2], sim3_j_data[3], sim3_j_data[4]);
        Eigen::Matrix<T, 3, 1> translation_j(sim3_j_data[5], sim3_j_data[6], sim3_j_data[7]);

        // Normalize quaternions
        rotation_i.normalize();
        rotation_j.normalize();

        // Convert measurement to template type
        Eigen::Quaternion<T> meas_rotation(
            T(m_measurement.rotation().w()),
            T(m_measurement.rotation().x()),
            T(m_measurement.rotation().y()),
            T(m_measurement.rotation().z()));
        meas_rotation.normalize();
        
        Eigen::Matrix<T, 3, 1> meas_translation(
            T(m_measurement.translation().x()),
            T(m_measurement.translation().y()),
            T(m_measurement.translation().z()));
        
        T meas_scale = T(m_measurement.scale());

        // Compute the relative Sim3 transformation: Sji_est = Sjw * Swi
        // This is the estimated transformation from i to j
        const T s_ji_est = scale_j / scale_i;
        const Eigen::Quaternion<T> q_ji_est = rotation_j * rotation_i.inverse();
        const Eigen::Matrix<T, 3, 1> t_ji_est = (T(1.0)/scale_i) * (rotation_j * rotation_i.inverse() * (-translation_i)) + translation_j;

        // Compute error between estimated and measured transformation
        // For rotation: error = log(meas_rotation^-1 * q_ji_est)
        const Eigen::Quaternion<T> q_error = meas_rotation.inverse() * q_ji_est;
        
        // Convert quaternion error to logarithm (axis-angle representation)
        Eigen::Matrix<T, 3, 1> omega;
        T angle = T(2.0) * atan2(q_error.vec().norm(), q_error.w());
        if (angle > T(M_PI)) {
            angle -= T(2.0 * M_PI);
        } else if (angle < -T(M_PI)) {
            angle += T(2.0 * M_PI);
        }
        
        if (q_error.vec().norm() > T(1e-10)) {
            omega = (angle / q_error.vec().norm()) * q_error.vec();
        } else {
            omega.setZero();
        }

        // For translation: error = t_ji_est - meas_translation
        const Eigen::Matrix<T, 3, 1> t_error = t_ji_est - meas_translation;
        
        // For scale: error = log(s_ji_est / meas_scale)
        const T s_error = log(s_ji_est / meas_scale);

        // Pack all errors into the residual vector
        residuals[0] = T(m_information_scale) * omega.x();
        residuals[1] = T(m_information_scale) * omega.y();
        residuals[2] = T(m_information_scale) * omega.z();
        residuals[3] = T(m_information_scale) * t_error.x();
        residuals[4] = T(m_information_scale) * t_error.y();
        residuals[5] = T(m_information_scale) * t_error.z();
        residuals[6] = T(m_information_scale) * s_error;

        return true;
    }

private:
    const Sim3 m_measurement;
    const double m_information_scale;
};

class EssentialGraphOptimizer {
public:
    // Default constructor
    EssentialGraphOptimizer() : current_kf_id(0), loop_kf_id(1) {}
    
    // Load keyframe trajectory
    bool LoadTrajectory(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open trajectory file: " << filename << std::endl;
            return false;
        }
        
        keyframes_.clear();
        
        std::string line;
        int id = 0;
        
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;
            
            std::istringstream iss(line);
            double timestamp_ns, tx, ty, tz, qx, qy, qz, qw;
            
            if (iss >> timestamp_ns >> tx >> ty >> tz >> qx >> qy >> qz >> qw) {
                KeyFrame kf;
                kf.id = id++;
                kf.timestamp = timestamp_ns / 1e9;
                kf.position = Eigen::Vector3d(tx, ty, tz);
                kf.rotation = Eigen::Quaterniond(qw, qx, qy, qz).normalized();
                kf.parent_id = -1;  // Will be set later
                kf.is_bad = false;
                
                keyframes_[kf.id] = kf;
                timestamp_to_id_[kf.timestamp] = kf.id;
                
                if (kf.id > current_kf_id) {
                    current_kf_id = kf.id;
                }
            }
        }
        
        file.close();
        std::cout << "Loaded " << keyframes_.size() << " keyframes" << std::endl;
        return !keyframes_.empty();
    }
    
    // Load loop constraints
    bool LoadLoopConstraints(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open loop constraints file: " << filename << std::endl;
            return false;
        }
        
        loop_edges_.clear();
        
        std::string line;
        std::getline(file, line); // Skip header
        
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;
            
            std::istringstream iss(line);
            int source_id, target_id;
            std::string constraint_type;
            double r00, r01, r02, r10, r11, r12, r20, r21, r22, t0, t1, t2, scale;
            
            if (iss >> source_id >> target_id >> constraint_type 
                   >> r00 >> r01 >> r02 >> r10 >> r11 >> r12 >> r20 >> r21 >> r22 
                   >> t0 >> t1 >> t2 >> scale) {
                
                Edge edge;
                edge.type = Edge::LOOP;
                edge.source_id = source_id;
                edge.target_id = target_id;
                edge.weight = 5.0; // Higher weight for loop edges
                
                Eigen::Matrix3d R;
                R << r00, r01, r02,
                     r10, r11, r12,
                     r20, r21, r22;
                
                edge.rel_rotation = Eigen::Quaterniond(R).normalized();
                edge.rel_translation = Eigen::Vector3d(t0, t1, t2);
                edge.rel_scale = scale;
                
                loop_edges_.push_back(edge);
                
                // Add to the loop edges for the keyframes
                if (keyframes_.find(source_id) != keyframes_.end()) {
                    keyframes_[source_id].loop_edges.insert(target_id);
                }
                if (keyframes_.find(target_id) != keyframes_.end()) {
                    keyframes_[target_id].loop_edges.insert(source_id);
                }
            }
        }
        
        file.close();
        std::cout << "Loaded " << loop_edges_.size() << " loop constraints" << std::endl;
        return !loop_edges_.empty();
    }
    
    // Load essential graph
    bool LoadEssentialGraph(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open essential graph file: " << filename << std::endl;
            return false;
        }
        
        spanning_tree_edges_.clear();
        covisibility_edges_.clear();
        
        std::string line;
        std::getline(file, line); // Skip header
        
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;
            
            std::istringstream iss(line);
            std::string edge_type;
            int source_id, target_id;
            double weight, info_scale;
            
            if (iss >> edge_type >> source_id >> target_id >> weight >> info_scale) {
                Edge edge;
                edge.source_id = source_id;
                edge.target_id = target_id;
                edge.weight = weight;
                
                if (edge_type == "SPANNING_TREE") {
                    edge.type = Edge::SPANNING_TREE;
                    // Set parent-child relationship
                    if (keyframes_.find(source_id) != keyframes_.end() && 
                        keyframes_.find(target_id) != keyframes_.end()) {
                        
                        keyframes_[target_id].parent_id = source_id;
                        
                        const KeyFrame& kf_source = keyframes_[source_id];
                        const KeyFrame& kf_target = keyframes_[target_id];
                        
                        // Relative pose from source to target
                        edge.rel_rotation = kf_source.rotation.conjugate() * kf_target.rotation;
                        edge.rel_translation = kf_source.rotation.conjugate() * 
                                            (kf_target.position - kf_source.position);
                        edge.rel_scale = 1.0;
                        
                        spanning_tree_edges_.push_back(edge);
                    }
                }
                else if (edge_type == "COVISIBILITY") {
                    edge.type = Edge::COVISIBILITY;
                    // Add covisibility relationship
                    if (keyframes_.find(source_id) != keyframes_.end() && 
                        keyframes_.find(target_id) != keyframes_.end()) {
                        
                        keyframes_[source_id].covisible_keyframes[target_id] = weight;
                        keyframes_[target_id].covisible_keyframes[source_id] = weight;
                        
                        const KeyFrame& kf_source = keyframes_[source_id];
                        const KeyFrame& kf_target = keyframes_[target_id];
                        
                        // Relative pose from source to target
                        edge.rel_rotation = kf_source.rotation.conjugate() * kf_target.rotation;
                        edge.rel_translation = kf_source.rotation.conjugate() * 
                                            (kf_target.position - kf_source.position);
                        edge.rel_scale = 1.0;
                        
                        covisibility_edges_.push_back(edge);
                    }
                }
            }
        }
        
        file.close();
        std::cout << "Loaded " << spanning_tree_edges_.size() << " spanning tree edges and " 
                  << covisibility_edges_.size() << " covisibility edges" << std::endl;
        return true;
    }
    
    // Load connection changes - new connections after loop closure
    bool LoadConnectionChanges(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open connection changes file: " << filename << std::endl;
            return false;
        }
        
        connection_changes_.clear();
        
        std::string line;
        std::getline(file, line); // Skip header
        
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;
            
            std::istringstream iss(line);
            int source_id, target_id, is_new;
            
            if (iss >> source_id >> target_id >> is_new) {
                if (is_new == 1) {
                    connection_changes_.push_back(std::make_pair(source_id, target_id));
                    loop_connections_[source_id].insert(target_id);
                }
            }
        }
        
        file.close();
        std::cout << "Loaded " << connection_changes_.size() << " new connections" << std::endl;
        return true;
    }
    
    // Load Sim3 transformations
    bool LoadSim3Transformations(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open Sim3 transforms file: " << filename << std::endl;
            return false;
        }
        
        sim3_non_corrected_map_.clear();
        sim3_corrected_map_.clear();
        
        std::string line;
        std::getline(file, line); // Skip header
        
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;
            
            std::istringstream iss(line);
            int kf_id;
            double s_orig, tx_orig, ty_orig, tz_orig, qx_orig, qy_orig, qz_orig, qw_orig;
            double s_corr, tx_corr, ty_corr, tz_corr, qx_corr, qy_corr, qz_corr, qw_corr;
            int propagation_source;
            
            if (iss >> kf_id
                   >> s_orig >> tx_orig >> ty_orig >> tz_orig >> qx_orig >> qy_orig >> qz_orig >> qw_orig
                   >> s_corr >> tx_corr >> ty_corr >> tz_corr >> qx_corr >> qy_corr >> qz_corr >> qw_corr
                   >> propagation_source) {
                
                // Create non-corrected Sim3
                Eigen::Quaterniond q_orig(qw_orig, qx_orig, qy_orig, qz_orig);
                Eigen::Vector3d t_orig(tx_orig, ty_orig, tz_orig);
                Sim3 orig_sim3(q_orig, t_orig, s_orig);
                sim3_non_corrected_map_[kf_id] = orig_sim3;
                
                // Create corrected Sim3
                Eigen::Quaterniond q_corr(qw_corr, qx_corr, qy_corr, qz_corr);
                Eigen::Vector3d t_corr(tx_corr, ty_corr, tz_corr);
                Sim3 corr_sim3(q_corr, t_corr, s_corr);
                sim3_corrected_map_[kf_id] = corr_sim3;
            }
        }
        
        file.close();
        std::cout << "Loaded " << sim3_non_corrected_map_.size() << " Sim3 transformations" << std::endl;
        return !sim3_non_corrected_map_.empty();
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
        std::getline(file, line); // Skip header
        
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;
            
            std::istringstream iss(line);
            int id, ref_kf_id;
            double x, y, z;
            int is_bad;
            
            if (iss >> id >> x >> y >> z >> ref_kf_id >> is_bad) {
                MapPoint mp;
                mp.id = id;
                mp.position = Eigen::Vector3d(x, y, z);
                mp.reference_kf_id = ref_kf_id;
                mp.is_bad = (is_bad == 1);
                mp.corrected_by_kf = -1;
                mp.corrected_reference = -1;
                
                map_points_.push_back(mp);
            }
        }
        
        file.close();
        std::cout << "Loaded " << map_points_.size() << " map points" << std::endl;
        return !map_points_.empty();
    }
    
    // Get max KeyFrame ID
    int GetMaxKFId() const {
        int max_id = 0;
        for (const auto& kf_pair : keyframes_) {
            if (kf_pair.first > max_id) {
                max_id = kf_pair.first;
            }
        }
        return max_id;
    }
    
    // Optimize the essential graph - CORE FUNCTION
    bool OptimizeEssentialGraph(const std::string& output_filename, int loop_kf_id = 1, bool fix_scale = true) {
        // Setup the Ceres problem
        ceres::Problem problem;
        ceres::LossFunction* loss_function = new ceres::HuberLoss(1.345);
        
        // Setup optimization parameters - similar to ORBSLAM3
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        options.minimizer_progress_to_stdout = true;
        options.max_num_iterations = 20;  // Same as ORBSLAM3
        options.function_tolerance = 1e-6;
        options.gradient_tolerance = 1e-10;
        options.parameter_tolerance = 1e-8;
        options.initial_trust_region_radius = 1e-16;  // Similar to ORBSLAM3's userLambdaInit
        
        std::cout << "Setting up optimization problem..." << std::endl;
        
        // Get max KeyFrame ID to allocate vectors (like ORBSLAM3)
        int nMaxKFid = GetMaxKFId();
        
        // Create parameter blocks for each keyframe - using Sim3 [scale, rotation, translation]
        std::map<int, double*> sim3_blocks;
        std::map<int, Sim3> vScw;  // Store original Sim3 (like ORBSLAM3)
        std::map<int, Sim3> vCorrectedSwc;  // For storing corrected inverse Sim3
        
        const int minFeat = 100;  // Minimum features for covisibility, same as ORBSLAM3
        
        // Add keyframe vertices - following ORBSLAM3's approach
        for(auto& kf_pair : keyframes_) {
            int id = kf_pair.first;
            const KeyFrame kf = kf_pair.second; 
            
            if (kf.is_bad) continue;
            
            // Create Sim3 parameter block [s, qw, qx, qy, qz, tx, ty, tz]
            double* sim3_block = new double[8];
            
            // Check if it's a corrected keyframe
            auto it = sim3_corrected_map_.find(id);
            if(it != sim3_corrected_map_.end()) {
                // This keyframe has a corrected Sim3
                const Sim3& corrected_sim3 = it->second;
                
                sim3_block[0] = corrected_sim3.scale();
                sim3_block[1] = corrected_sim3.rotation().w();
                sim3_block[2] = corrected_sim3.rotation().x();
                sim3_block[3] = corrected_sim3.rotation().y();
                sim3_block[4] = corrected_sim3.rotation().z();
                sim3_block[5] = corrected_sim3.translation().x();
                sim3_block[6] = corrected_sim3.translation().y();
                sim3_block[7] = corrected_sim3.translation().z();
                
                vScw[id] = corrected_sim3;
            } else {
                // Use current pose and scale=1.0
                sim3_block[0] = 1.0;  // Scale
                sim3_block[1] = kf.rotation.w();
                sim3_block[2] = kf.rotation.x();
                sim3_block[3] = kf.rotation.y();
                sim3_block[4] = kf.rotation.z();
                sim3_block[5] = kf.position.x();
                sim3_block[6] = kf.position.y();
                sim3_block[7] = kf.position.z();
                
                Sim3 Siw(kf.rotation, kf.position, 1.0);
                vScw[id] = Siw;
            }
            
            sim3_blocks[id] = sim3_block;
            
            // Add parameter block to problem
            // 替换为:
            ceres::Manifold* sim3_parameterization = new Sim3Parameterization(fix_scale);
            problem.AddParameterBlock(sim3_block, 8, sim3_parameterization);
            
            // Fix the initial KF (similar to ORBSLAM3)
            if(id == loop_kf_id) {
                problem.SetParameterBlockConstant(sim3_block);
            }
        }
        
        // Track inserted edges to avoid duplicates (like ORBSLAM3)
        std::set<std::pair<long unsigned int, long unsigned int>> sInsertedEdges;
        
        // Add loop edges - similar to ORBSLAM3
        std::cout << "Adding loop edges..." << std::endl;
        for(const auto& entry : loop_connections_) {
            int source_id = entry.first;
            const std::set<int>& connected_kfs = entry.second;
            
            if(sim3_blocks.find(source_id) == sim3_blocks.end())
                continue;
                
            const Sim3& Siw = vScw[source_id];
            const Sim3 Swi = Siw.inverse();
            
            for(int target_id : connected_kfs) {
                if(sim3_blocks.find(target_id) == sim3_blocks.end())
                    continue;
                    
                // Skip if this connection has already been processed
                if(sInsertedEdges.count(std::make_pair(std::min(source_id, target_id), 
                                                    std::max(source_id, target_id))))
                    continue;
                
                // Skip if weight is too low (similar to ORBSLAM3)
                auto weight_it = keyframes_[source_id].covisible_keyframes.find(target_id);
                if((source_id != current_kf_id || target_id != loop_kf_id) && 
                   (weight_it == keyframes_[source_id].covisible_keyframes.end() || weight_it->second < minFeat))
                    continue;
                
                const Sim3& Sjw = vScw[target_id];
                const Sim3 Sji = Sjw * Swi;
                
                // Add Sim3 edge
                ceres::CostFunction* cost_function = 
                    new ceres::AutoDiffCostFunction<Sim3Error, 7, 8, 8>(
                        new Sim3Error(Sji, 1.0));
                
                problem.AddResidualBlock(cost_function,
                                         loss_function,
                                         sim3_blocks[source_id],
                                         sim3_blocks[target_id]);
                                         
                sInsertedEdges.insert(std::make_pair(std::min(source_id, target_id), 
                                                    std::max(source_id, target_id)));
            }
        }
        
        // Add spanning tree edges - similar to ORBSLAM3
        std::cout << "Adding spanning tree edges..." << std::endl;
        for(const auto& kf_pair : keyframes_) {
            int target_id = kf_pair.first;
            KeyFrame kf = kf_pair.second; 
            
            if(kf.is_bad || kf.parent_id < 0)
                continue;
                
            int source_id = kf.parent_id;
            
            if(sim3_blocks.find(source_id) == sim3_blocks.end() || 
               sim3_blocks.find(target_id) == sim3_blocks.end())
                continue;
                
            // Skip if this connection has already been processed
            if(sInsertedEdges.count(std::make_pair(std::min(source_id, target_id), 
                                                std::max(source_id, target_id))))
                continue;
            
            // Get the relative Sim3 transformation
            Sim3 Sji;
            
            // Check if source has a non-corrected Sim3
            auto src_it = sim3_non_corrected_map_.find(source_id);
            if(src_it != sim3_non_corrected_map_.end()) {
                const Sim3& Swi = (src_it->second).inverse();
                
                // Check if target has a non-corrected Sim3
                auto tgt_it = sim3_non_corrected_map_.find(target_id);
                if(tgt_it != sim3_non_corrected_map_.end()) {
                    const Sim3& Sjw = tgt_it->second;
                    Sji = Sjw * Swi;
                } else {
                    const Sim3& Sjw = vScw[target_id];
                    Sji = Sjw * Swi;
                }
            } else {
                const Sim3& Swi = vScw[source_id].inverse();
                
                // Check if target has a non-corrected Sim3
                auto tgt_it = sim3_non_corrected_map_.find(target_id);
                if(tgt_it != sim3_non_corrected_map_.end()) {
                    const Sim3& Sjw = tgt_it->second;
                    Sji = Sjw * Swi;
                } else {
                    const Sim3& Sjw = vScw[target_id];
                    Sji = Sjw * Swi;
                }
            }
            
            // Add Sim3 edge with higher weight for spanning tree
            ceres::CostFunction* cost_function = 
                new ceres::AutoDiffCostFunction<Sim3Error, 7, 8, 8>(
                    new Sim3Error(Sji, 3.0));  // Higher weight (3.0) for spanning tree
            
            problem.AddResidualBlock(cost_function,
                                     loss_function,
                                     sim3_blocks[source_id],
                                     sim3_blocks[target_id]);
                                     
            sInsertedEdges.insert(std::make_pair(std::min(source_id, target_id), 
                                                std::max(source_id, target_id)));
        }
        
        // Add loop edges (from previous loop detections)
        for(const auto& kf_pair : keyframes_) {
            int source_id = kf_pair.first;
            KeyFrame kf = kf_pair.second; 
            
            if(kf.is_bad) continue;
            
            for(int target_id : kf.loop_edges) {
                if(target_id < source_id) continue; // Process each edge only once
                
                if(sim3_blocks.find(source_id) == sim3_blocks.end() || 
                   sim3_blocks.find(target_id) == sim3_blocks.end() ||
                   keyframes_[target_id].is_bad)
                    continue;
                    
                // Skip if this connection has already been processed
                if(sInsertedEdges.count(std::make_pair(source_id, target_id)))
                    continue;
                    
                // Get the relative Sim3 transformation (similar to spanning tree)
                Sim3 Sji;
                
                auto src_it = sim3_non_corrected_map_.find(source_id);
                if(src_it != sim3_non_corrected_map_.end()) {
                    const Sim3& Swi = (src_it->second).inverse();
                    
                    auto tgt_it = sim3_non_corrected_map_.find(target_id);
                    if(tgt_it != sim3_non_corrected_map_.end()) {
                        const Sim3& Sjw = tgt_it->second;
                        Sji = Sjw * Swi;
                    } else {
                        const Sim3& Sjw = vScw[target_id];
                        Sji = Sjw * Swi;
                    }
                } else {
                    const Sim3& Swi = vScw[source_id].inverse();
                    
                    auto tgt_it = sim3_non_corrected_map_.find(target_id);
                    if(tgt_it != sim3_non_corrected_map_.end()) {
                        const Sim3& Sjw = tgt_it->second;
                        Sji = Sjw * Swi;
                    } else {
                        const Sim3& Sjw = vScw[target_id];
                        Sji = Sjw * Swi;
                    }
                }
                
                // Add Sim3 edge with loop edge weight
                ceres::CostFunction* cost_function = 
                    new ceres::AutoDiffCostFunction<Sim3Error, 7, 8, 8>(
                        new Sim3Error(Sji, 5.0));  // Higher weight (5.0) for loop edges
                
                problem.AddResidualBlock(cost_function,
                                         loss_function,
                                         sim3_blocks[source_id],
                                         sim3_blocks[target_id]);
                                         
                sInsertedEdges.insert(std::make_pair(source_id, target_id));
            }
        }
        
        // Add covisibility edges
        std::cout << "Adding covisibility edges..." << std::endl;
        for(const auto& kf_pair : keyframes_) {
            int source_id = kf_pair.first;
            KeyFrame kf = kf_pair.second; 
            
            if(kf.is_bad) continue;
            
            for(const auto& cov_pair : kf.covisible_keyframes) {
                int target_id = cov_pair.first;
                int weight = cov_pair.second;
                
                // Process each edge only once
                if(target_id < source_id) continue;
                
                // Skip if target KF is bad
                if(keyframes_.find(target_id) == keyframes_.end() || 
                   keyframes_[target_id].is_bad)
                    continue;
                
                // Skip if already have an edge between these KFs
                if(sInsertedEdges.count(std::make_pair(source_id, target_id)))
                    continue;
                    
                // Skip if weight is too low
                if(weight < minFeat)
                    continue;
                    
                if(sim3_blocks.find(source_id) == sim3_blocks.end() || 
                   sim3_blocks.find(target_id) == sim3_blocks.end())
                    continue;
                    
                // Get the relative Sim3 transformation (similar to spanning tree)
                Sim3 Sji;
                
                auto src_it = sim3_non_corrected_map_.find(source_id);
                if(src_it != sim3_non_corrected_map_.end()) {
                    const Sim3& Swi = (src_it->second).inverse();
                    
                    auto tgt_it = sim3_non_corrected_map_.find(target_id);
                    if(tgt_it != sim3_non_corrected_map_.end()) {
                        const Sim3& Sjw = tgt_it->second;
                        Sji = Sjw * Swi;
                    } else {
                        const Sim3& Sjw = vScw[target_id];
                        Sji = Sjw * Swi;
                    }
                } else {
                    const Sim3& Swi = vScw[source_id].inverse();
                    
                    auto tgt_it = sim3_non_corrected_map_.find(target_id);
                    if(tgt_it != sim3_non_corrected_map_.end()) {
                        const Sim3& Sjw = tgt_it->second;
                        Sji = Sjw * Swi;
                    } else {
                        const Sim3& Sjw = vScw[target_id];
                        Sji = Sjw * Swi;
                    }
                }
                
                // Add Sim3 edge with appropriate weight for covisibility
                ceres::CostFunction* cost_function = 
                    new ceres::AutoDiffCostFunction<Sim3Error, 7, 8, 8>(
                        new Sim3Error(Sji, 1.0));  // Standard weight for covisibility
                
                problem.AddResidualBlock(cost_function,
                                         loss_function,
                                         sim3_blocks[source_id],
                                         sim3_blocks[target_id]);
                                         
                sInsertedEdges.insert(std::make_pair(source_id, target_id));
            }
        }
        
        // Add connection changes edges
        std::cout << "Adding connection changes..." << std::endl;
        for(const auto& conn : connection_changes_) {
            int source_id = conn.first;
            int target_id = conn.second;
            
            // Skip if already have an edge between these KFs
            if(sInsertedEdges.count(std::make_pair(std::min(source_id, target_id), 
                                                  std::max(source_id, target_id))))
                continue;
                
            if(sim3_blocks.find(source_id) == sim3_blocks.end() || 
               sim3_blocks.find(target_id) == sim3_blocks.end())
                continue;
                
            // Compute relative Sim3 transformation
            const Sim3& Siw = vScw[source_id];
            const Sim3& Sjw = vScw[target_id];
            const Sim3 Sji = Sjw * Siw.inverse();
            
            // Add Sim3 edge with weight for new connections
            ceres::CostFunction* cost_function = 
                new ceres::AutoDiffCostFunction<Sim3Error, 7, 8, 8>(
                    new Sim3Error(Sji, 2.0));  // Weight = 2.0 for new connections
            
            problem.AddResidualBlock(cost_function,
                                     loss_function,
                                     sim3_blocks[source_id],
                                     sim3_blocks[target_id]);
                                     
            sInsertedEdges.insert(std::make_pair(std::min(source_id, target_id), 
                                                std::max(source_id, target_id)));
        }
        
        // Solve the problem
        std::cout << "Solving optimization problem..." << std::endl;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        
        std::cout << summary.BriefReport() << std::endl;
        
        // SE3 Pose Recovering (similar to ORBSLAM3)
        // Sim3:[s,R,t] -> SE3:[R,t/s]
        for(auto& kf_pair : keyframes_) {
            int id = kf_pair.first;
            KeyFrame kf = kf_pair.second; 
            
            if(sim3_blocks.find(id) == sim3_blocks.end())
                continue;
                
            const double* sim3_block = sim3_blocks[id];
            double s = sim3_block[0];
            
            // Store corrected Sim3 for map point correction
            Eigen::Quaterniond corrected_rot(sim3_block[1], sim3_block[2], sim3_block[3], sim3_block[4]);
            Eigen::Vector3d corrected_trans(sim3_block[5], sim3_block[6], sim3_block[7]);
            Sim3 CorrectedSiw(corrected_rot, corrected_trans, s);
            vCorrectedSwc[id] = CorrectedSiw.inverse();
            
            // Convert to SE3 by dividing translation by scale
            kf.rotation = Eigen::Quaterniond(sim3_block[1], sim3_block[2], sim3_block[3], sim3_block[4]);
            kf.position = Eigen::Vector3d(sim3_block[5]/s, sim3_block[6]/s, sim3_block[7]/s);
        }
        
        // Correct map points (similar to ORBSLAM3)
        for(MapPoint& mp : map_points_) {
            if(mp.is_bad)
                continue;
            
            // Get reference keyframe
            int nIDr;
            if(mp.corrected_by_kf == current_kf_id) {
                nIDr = mp.corrected_reference;
            } else {
                nIDr = mp.reference_kf_id;
            }
            
            if(vScw.find(nIDr) == vScw.end() || vCorrectedSwc.find(nIDr) == vCorrectedSwc.end())
                continue;
                
            // Get the original and corrected Sim3 transformations
            Sim3 Srw = vScw[nIDr];
            Sim3 correctedSwr = vCorrectedSwc[nIDr];
            
            // Transform the point from world to reference KF and back with corrected pose
            Eigen::Vector3d eigP3Dw = mp.position;
            Eigen::Vector3d eigCorrectedP3Dw = correctedSwr.map(Srw.map(eigP3Dw));
            
            // Update the map point position
            mp.position = eigCorrectedP3Dw;
        }
        
        // Clean up
        for(auto& block_pair : sim3_blocks) {
            delete[] block_pair.second;
        }
        
        // Save optimized trajectory
        SaveTrajectory(output_filename);
        
        return true;
    }
    
    // Save the trajectory
    void SaveTrajectory(const std::string& filename) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open output file: " << filename << std::endl;
            return;
        }
        
        // Create ordered keyframes by timestamp
        std::vector<const KeyFrame*> ordered_kfs;
        for (const auto& kf_pair : keyframes_) {
            if (!kf_pair.second.is_bad) {
                ordered_kfs.push_back(&kf_pair.second);
            }
        }
        
        std::sort(ordered_kfs.begin(), ordered_kfs.end(),
                 [](const KeyFrame* a, const KeyFrame* b) {
                     return a->timestamp < b->timestamp;
                 });
        
        for (const KeyFrame* kf : ordered_kfs) {
            file << std::fixed << std::setprecision(9)
                 << kf->timestamp * 1e9 << " "
                 << kf->position.x() << " "
                 << kf->position.y() << " "
                 << kf->position.z() << " "
                 << kf->rotation.x() << " "
                 << kf->rotation.y() << " "
                 << kf->rotation.z() << " "
                 << kf->rotation.w() << std::endl;
        }
        
        file.close();
        std::cout << "Saved optimized trajectory to: " << filename << std::endl;
    }
    
    // Save optimized map points
    void SaveMapPoints(const std::string& filename) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open output file: " << filename << std::endl;
            return;
        }
        
        file << "# ID X Y Z RefKF IsBad CorrectedByKF CorrectedRef" << std::endl;
        
        for (const MapPoint& mp : map_points_) {
            if (mp.is_bad) continue;
            
            file << mp.id << " "
                 << mp.position.x() << " "
                 << mp.position.y() << " "
                 << mp.position.z() << " "
                 << mp.reference_kf_id << " "
                 << (mp.is_bad ? 1 : 0) << " "
                 << mp.corrected_by_kf << " "
                 << mp.corrected_reference << std::endl;
        }
        
        file.close();
        std::cout << "Saved optimized map points to: " << filename << std::endl;
    }
    
    // Member variables
    std::map<int, KeyFrame> keyframes_;
    std::map<double, int> timestamp_to_id_;
    std::vector<Edge> loop_edges_;
    std::vector<Edge> spanning_tree_edges_;
    std::vector<Edge> covisibility_edges_;
    std::vector<std::pair<int, int>> connection_changes_;
    std::vector<MapPoint> map_points_;
    
    // Maps for Sim3 transformations (similar to ORBSLAM3's NonCorrectedSim3 and CorrectedSim3)
    std::map<int, Sim3> sim3_non_corrected_map_;
    std::map<int, Sim3> sim3_corrected_map_;
    
    // Loop connections (similar to ORBSLAM3's LoopConnections)
    std::map<int, std::set<int>> loop_connections_;
    
    // Current KF and Loop KF IDs
    int current_kf_id;
    int loop_kf_id;
};

int main(int argc, char** argv) {
    std::string input_dir = "/Datasets/CERES_Work/input";
    std::string output_file = "/Datasets/CERES_Work/build/ceres_optimized_trajectory.txt";
    
    if (argc > 1) input_dir = argv[1];
    if (argc > 2) output_file = argv[2];
    
    EssentialGraphOptimizer optimizer;
    
    // Load the Sim3 transformed trajectory (this is the input data after Sim3 transformation but before optimization)
    std::cout << "Loading Sim3 transformed trajectory..." << std::endl;
    if (!optimizer.LoadTrajectory(input_dir + "/sim3_transformed_trajectory.txt")) {
        std::cerr << "Failed to load trajectory!" << std::endl;
        return 1;
    }
    
    // Load constraints and graph structure (essential graph)
    std::cout << "Loading constraints..." << std::endl;
    optimizer.LoadLoopConstraints(input_dir + "/metadata/loop_constraints.txt");
    optimizer.LoadEssentialGraph(input_dir + "/pre/essential_graph.txt");
    optimizer.LoadConnectionChanges(input_dir + "/metadata/connection_changes.txt");
    
    // Attempt to load Sim3 transformations, but continue if not available
    std::cout << "Loading Sim3 transformations..." << std::endl;
    optimizer.LoadSim3Transformations(input_dir + "/metadata/sim3_transforms.txt");
    
    // Attempt to load map points, but continue if not available
    std::cout << "Loading map points..." << std::endl;
    optimizer.LoadMapPoints(input_dir + "/transformed/mappoints_sim3_transformed.txt");
    
    // Set loop keyframe ID (usually KF 1)
    optimizer.loop_kf_id = 1;
    optimizer.current_kf_id = optimizer.GetMaxKFId();
    
    // Optimize the essential graph
    std::cout << "Optimizing essential graph..." << std::endl;
    optimizer.OptimizeEssentialGraph(output_file, optimizer.loop_kf_id, true);
    
    // Save updated map points if they were loaded
    if (!optimizer.map_points_.empty()) {
        std::cout << "Saving optimized map points..." << std::endl;
        optimizer.SaveMapPoints(output_file.substr(0, output_file.find_last_of('.')) + "_mappoints.txt");
    }
    
    std::cout << "Essential Graph optimization completed!" << std::endl;
    return 0;
}
