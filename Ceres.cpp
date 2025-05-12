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

// Define KeyFrame structure
struct KeyFrame {
    int id;
    double timestamp;
    Eigen::Quaterniond rotation;
    Eigen::Vector3d position;
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

// Cost function for quaternion rotation errors
struct RotationError {
    RotationError(const Eigen::Quaterniond& relative_rotation, double weight = 1.0)
        : rel_rot_(relative_rotation), weight_(weight) {}

    template <typename T>
    bool operator()(const T* const quat_i, const T* const quat_j, T* residuals) const {
        Eigen::Quaternion<T> q_i(quat_i[0], quat_i[1], quat_i[2], quat_i[3]);
        Eigen::Quaternion<T> q_j(quat_j[0], quat_j[1], quat_j[2], quat_j[3]);
        
        // Normalize quaternions
        q_i.normalize();
        q_j.normalize();
        
        // Compute the relative transformation between vertices
        Eigen::Quaternion<T> q_ij = q_i.conjugate() * q_j;
        
        // Convert relative rotation to template type
        Eigen::Quaternion<T> rel_rot_t(
            T(rel_rot_.w()), T(rel_rot_.x()), T(rel_rot_.y()), T(rel_rot_.z()));
        rel_rot_t.normalize();
        
        // Compute the residual: (observed - predicted)
        Eigen::Quaternion<T> delta_rot = rel_rot_t.conjugate() * q_ij;
        
        // Convert to axis-angle representation
        Eigen::AngleAxis<T> angle_axis(delta_rot);
        
        // Scale by weight
        residuals[0] = weight_ * angle_axis.axis()[0] * angle_axis.angle();
        residuals[1] = weight_ * angle_axis.axis()[1] * angle_axis.angle();
        residuals[2] = weight_ * angle_axis.axis()[2] * angle_axis.angle();
        
        return true;
    }

private:
    const Eigen::Quaterniond rel_rot_;
    const double weight_;
};

// Cost function for translation errors
struct TranslationError {
    TranslationError(const Eigen::Vector3d& relative_translation, double weight = 1.0)
        : rel_trans_(relative_translation), weight_(weight) {}

    template <typename T>
    bool operator()(const T* const quat_i, const T* const trans_i,
                    const T* const quat_j, const T* const trans_j,
                    T* residuals) const {
        // Convert quaternions
        Eigen::Quaternion<T> q_i(quat_i[0], quat_i[1], quat_i[2], quat_i[3]);
        q_i.normalize();
        
        // Convert translations
        Eigen::Matrix<T, 3, 1> t_i(trans_i[0], trans_i[1], trans_i[2]);
        Eigen::Matrix<T, 3, 1> t_j(trans_j[0], trans_j[1], trans_j[2]);
        
        // Compute the relative translation in source frame
        Eigen::Matrix<T, 3, 1> t_ij = q_i.conjugate() * (t_j - t_i);
        
        // Convert relative translation to template type
        Eigen::Matrix<T, 3, 1> rel_trans_t(
            T(rel_trans_.x()), T(rel_trans_.y()), T(rel_trans_.z()));
        
        // Compute the residual: (observed - predicted)
        Eigen::Matrix<T, 3, 1> delta_trans = t_ij - rel_trans_t;
        
        // Scale by weight
        residuals[0] = weight_ * delta_trans(0);
        residuals[1] = weight_ * delta_trans(1);
        residuals[2] = weight_ * delta_trans(2);
        
        return true;
    }

private:
    const Eigen::Vector3d rel_trans_;
    const double weight_;
};

class EssentialGraphOptimizer {
public:
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
                
                keyframes_[kf.id] = kf;
                timestamp_to_id_[kf.timestamp] = kf.id;
            }
        }
        
        file.close();
        std::cout << "Loaded " << keyframes_.size() << " keyframes" << std::endl;
        return !keyframes_.empty();
    }
    
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
            }
        }
        
        file.close();
        std::cout << "Loaded " << loop_edges_.size() << " loop constraints" << std::endl;
        return !loop_edges_.empty();
    }
    
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
                    // Compute relative transformation
                    if (keyframes_.find(source_id) != keyframes_.end() && 
                        keyframes_.find(target_id) != keyframes_.end()) {
                        
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
                    // Only add significant covisibility edges
                    if (weight >= 100) { // Threshold for significant covisibility
                        // Compute relative transformation
                        if (keyframes_.find(source_id) != keyframes_.end() && 
                            keyframes_.find(target_id) != keyframes_.end()) {
                            
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
        }
        
        file.close();
        std::cout << "Loaded " << spanning_tree_edges_.size() << " spanning tree edges and " 
                  << covisibility_edges_.size() << " covisibility edges" << std::endl;
        return true;
    }
    
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
                }
            }
        }
        
        file.close();
        std::cout << "Loaded " << connection_changes_.size() << " new connections" << std::endl;
        return true;
    }
    
    bool OptimizeEssentialGraph(const std::string& output_filename, int loop_kf_id = 1, bool /* fix_scale */ = true) {
        // Setup the Ceres problem
        ceres::Problem problem;
        ceres::LossFunction* loss_function = new ceres::HuberLoss(1.345);
        
        // Setup optimization parameters
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        options.minimizer_progress_to_stdout = true;
        options.max_num_iterations = 20;
        options.function_tolerance = 1e-6;
        options.gradient_tolerance = 1e-10;
        options.parameter_tolerance = 1e-8;
        
        std::cout << "Setting up optimization problem..." << std::endl;
        
        // Create parameter blocks for each keyframe - split into rotation and position
        std::map<int, double*> rotation_blocks;
        std::map<int, double*> position_blocks;
        
        for (auto& kf_pair : keyframes_) {
            int id = kf_pair.first;
            KeyFrame& kf = kf_pair.second;
            
            // Quaternion parameter block [qw, qx, qy, qz]
            double* rotation_block = new double[4];
            rotation_block[0] = kf.rotation.w();
            rotation_block[1] = kf.rotation.x();
            rotation_block[2] = kf.rotation.y();
            rotation_block[3] = kf.rotation.z();
            rotation_blocks[id] = rotation_block;
            
            // Position parameter block [tx, ty, tz]
            double* position_block = new double[3];
            position_block[0] = kf.position.x();
            position_block[1] = kf.position.y();
            position_block[2] = kf.position.z();
            position_blocks[id] = position_block;
            
            // Add parameter blocks to problem
            auto* quaternion_manifold = new ceres::EigenQuaternionManifold();
            problem.AddParameterBlock(rotation_block, 4, quaternion_manifold);
            problem.AddParameterBlock(position_block, 3);
            
            // Fix the loop keyframe (usually keyframe 1 or the origin)
            if (id == loop_kf_id) {
                problem.SetParameterBlockConstant(rotation_block);
                problem.SetParameterBlockConstant(position_block);
            }
        }
        
        // Add loop edge constraints
        std::cout << "Adding loop edge constraints..." << std::endl;
        for (const Edge& edge : loop_edges_) {
            if (rotation_blocks.find(edge.source_id) != rotation_blocks.end() &&
                rotation_blocks.find(edge.target_id) != rotation_blocks.end()) {
                
                // Add rotation constraint
                ceres::CostFunction* rot_cost_function =
                    new ceres::AutoDiffCostFunction<RotationError, 3, 4, 4>(
                        new RotationError(edge.rel_rotation, 5.0));
                
                problem.AddResidualBlock(rot_cost_function,
                                        loss_function,
                                        rotation_blocks[edge.source_id],
                                        rotation_blocks[edge.target_id]);
                
                // Add translation constraint
                ceres::CostFunction* trans_cost_function =
                    new ceres::AutoDiffCostFunction<TranslationError, 3, 4, 3, 4, 3>(
                        new TranslationError(edge.rel_translation, 5.0));
                
                problem.AddResidualBlock(trans_cost_function,
                                        loss_function,
                                        rotation_blocks[edge.source_id],
                                        position_blocks[edge.source_id],
                                        rotation_blocks[edge.target_id],
                                        position_blocks[edge.target_id]);
            }
        }
        
        // Add spanning tree edge constraints
        std::cout << "Adding spanning tree edge constraints..." << std::endl;
        for (const Edge& edge : spanning_tree_edges_) {
            if (rotation_blocks.find(edge.source_id) != rotation_blocks.end() &&
                rotation_blocks.find(edge.target_id) != rotation_blocks.end()) {
                
                // Add rotation constraint
                ceres::CostFunction* rot_cost_function =
                    new ceres::AutoDiffCostFunction<RotationError, 3, 4, 4>(
                        new RotationError(edge.rel_rotation, 3.0));
                
                problem.AddResidualBlock(rot_cost_function,
                                        loss_function,
                                        rotation_blocks[edge.source_id],
                                        rotation_blocks[edge.target_id]);
                
                // Add translation constraint
                ceres::CostFunction* trans_cost_function =
                    new ceres::AutoDiffCostFunction<TranslationError, 3, 4, 3, 4, 3>(
                        new TranslationError(edge.rel_translation, 3.0));
                
                problem.AddResidualBlock(trans_cost_function,
                                        loss_function,
                                        rotation_blocks[edge.source_id],
                                        position_blocks[edge.source_id],
                                        rotation_blocks[edge.target_id],
                                        position_blocks[edge.target_id]);
            }
        }
        
        // Add covisibility edge constraints
        std::cout << "Adding covisibility edge constraints..." << std::endl;
        for (const Edge& edge : covisibility_edges_) {
            if (rotation_blocks.find(edge.source_id) != rotation_blocks.end() &&
                rotation_blocks.find(edge.target_id) != rotation_blocks.end()) {
                
                // Add rotation constraint
                ceres::CostFunction* rot_cost_function =
                    new ceres::AutoDiffCostFunction<RotationError, 3, 4, 4>(
                        new RotationError(edge.rel_rotation, 1.0));
                
                problem.AddResidualBlock(rot_cost_function,
                                        loss_function,
                                        rotation_blocks[edge.source_id],
                                        rotation_blocks[edge.target_id]);
                
                // Add translation constraint
                ceres::CostFunction* trans_cost_function =
                    new ceres::AutoDiffCostFunction<TranslationError, 3, 4, 3, 4, 3>(
                        new TranslationError(edge.rel_translation, 1.0));
                
                problem.AddResidualBlock(trans_cost_function,
                                        loss_function,
                                        rotation_blocks[edge.source_id],
                                        position_blocks[edge.source_id],
                                        rotation_blocks[edge.target_id],
                                        position_blocks[edge.target_id]);
            }
        }
        
        // Add connection changes constraints
        std::cout << "Adding connection changes constraints..." << std::endl;
        for (const auto& conn : connection_changes_) {
            int source_id = conn.first;
            int target_id = conn.second;
            
            if (rotation_blocks.find(source_id) != rotation_blocks.end() &&
                rotation_blocks.find(target_id) != rotation_blocks.end()) {
                
                // Compute relative transformation
                const KeyFrame& kf_source = keyframes_[source_id];
                const KeyFrame& kf_target = keyframes_[target_id];
                
                Eigen::Quaterniond rel_rotation = kf_source.rotation.conjugate() * kf_target.rotation;
                Eigen::Vector3d rel_translation = kf_source.rotation.conjugate() * 
                                                (kf_target.position - kf_source.position);
                
                // Add rotation constraint
                ceres::CostFunction* rot_cost_function =
                    new ceres::AutoDiffCostFunction<RotationError, 3, 4, 4>(
                        new RotationError(rel_rotation, 2.0));
                
                problem.AddResidualBlock(rot_cost_function,
                                        loss_function,
                                        rotation_blocks[source_id],
                                        rotation_blocks[target_id]);
                
                // Add translation constraint
                ceres::CostFunction* trans_cost_function =
                    new ceres::AutoDiffCostFunction<TranslationError, 3, 4, 3, 4, 3>(
                        new TranslationError(rel_translation, 2.0));
                
                problem.AddResidualBlock(trans_cost_function,
                                        loss_function,
                                        rotation_blocks[source_id],
                                        position_blocks[source_id],
                                        rotation_blocks[target_id],
                                        position_blocks[target_id]);
            }
        }
        
        // Solve the problem
        std::cout << "Solving optimization problem..." << std::endl;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        
        std::cout << summary.BriefReport() << std::endl;
        
        // Update keyframe poses
        for (auto& kf_pair : keyframes_) {
            int id = kf_pair.first;
            KeyFrame& kf = kf_pair.second;
            
            const double* rotation_block = rotation_blocks[id];
            const double* position_block = position_blocks[id];
            
            kf.rotation = Eigen::Quaterniond(rotation_block[0], rotation_block[1], 
                                            rotation_block[2], rotation_block[3]).normalized();
            kf.position = Eigen::Vector3d(position_block[0], position_block[1], position_block[2]);
        }
        
        // Clean up
        for (auto& block_pair : rotation_blocks) {
            delete[] block_pair.second;
        }
        for (auto& block_pair : position_blocks) {
            delete[] block_pair.second;
        }
        
        // Save optimized trajectory
        SaveTrajectory(output_filename);
        
        return true;
    }
    
    void SaveTrajectory(const std::string& filename) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open output file: " << filename << std::endl;
            return;
        }
        
        // Create ordered keyframes by timestamp
        std::vector<const KeyFrame*> ordered_kfs;
        for (const auto& kf_pair : keyframes_) {
            ordered_kfs.push_back(&kf_pair.second);
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
    
private:
    std::map<int, KeyFrame> keyframes_;
    std::map<double, int> timestamp_to_id_;
    std::vector<Edge> loop_edges_;
    std::vector<Edge> spanning_tree_edges_;
    std::vector<Edge> covisibility_edges_;
    std::vector<std::pair<int, int>> connection_changes_;
};

int main(int argc, char** argv) {
    std::string input_dir = "/Datasets/CERES_Work/input";
    std::string output_file = "/Datasets/CERES_Work/build/ceres_optimized_trajectory.txt";
    
    if (argc > 1) input_dir = argv[1];
    if (argc > 2) output_file = argv[2];
    
    EssentialGraphOptimizer optimizer;
    
    // Load the Sim3 transformed trajectory
    std::cout << "Loading Sim3 transformed trajectory..." << std::endl;
    if (!optimizer.LoadTrajectory(input_dir + "/sim3_transformed_trajectory.txt")) {
        std::cerr << "Failed to load trajectory!" << std::endl;
        return 1;
    }
    
    // Load constraints and graph structure
    std::cout << "Loading constraints..." << std::endl;
    optimizer.LoadLoopConstraints(input_dir + "/metadata/loop_constraints.txt");
    optimizer.LoadEssentialGraph(input_dir + "/pre/essential_graph.txt");
    optimizer.LoadConnectionChanges(input_dir + "/metadata/connection_changes.txt");
    
    // Optimize the essential graph
    std::cout << "Optimizing essential graph..." << std::endl;
    optimizer.OptimizeEssentialGraph(output_file, 1, true);
    
    std::cout << "Essential Graph optimization completed!" << std::endl;
    return 0;
}
