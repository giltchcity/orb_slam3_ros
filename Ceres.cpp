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
#include <iomanip>
#include <algorithm> // 用于sort

// 前向声明
class KeyFrame;
class MapPoint;
class Map;

// 定义SE3Pose类型
typedef std::pair<Eigen::Quaterniond, Eigen::Vector3d> SE3Pose;

// 定义KeyFrameAndPose类型
typedef std::map<KeyFrame*, SE3Pose> KeyFrameAndPose;

// 简化的KeyFrame类
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
    
    // 姿态作为旋转和平移分别存储
    Eigen::Matrix3f mRcw;  // 旋转
    Eigen::Vector3f mtcw;  // 平移

    // 方法
    bool isBad() const { return mbBad; }
    
    // 获取/设置姿态方法
    void GetPose(Eigen::Matrix3f& R, Eigen::Vector3f& t) const {
        R = mRcw;
        t = mtcw;
    }
    
    void SetPose(const Eigen::Matrix3f& R, const Eigen::Vector3f& t) {
        mRcw = R;
        mtcw = t;
    }
    
    // 转换为逆变换
    void GetPoseInverse(Eigen::Matrix3f& Rwc, Eigen::Vector3f& twc) const {
        Rwc = mRcw.transpose();
        twc = -Rwc * mtcw;
    }
    
    KeyFrame* GetParent() { return mpParent; }
    bool hasChild(KeyFrame* /* pKF */) const { return false; }  // 简化版
    std::set<KeyFrame*> GetLoopEdges() const { return mspLoopEdges; }
    std::vector<KeyFrame*> GetCovisiblesByWeight(int /* minWeight */) const {
        return mvpOrderedConnectedKeyFrames;  // 简化版
    }
    
    int GetWeight(KeyFrame* pKF) const { 
        // 在mvpOrderedConnectedKeyFrames中查找pKF并返回权重
        for(size_t i=0; i<mvpOrderedConnectedKeyFrames.size(); i++) {
            if(mvpOrderedConnectedKeyFrames[i] == pKF)
                return mvOrderedWeights[i];
        }
        return 0;
    }
};

// 简化的MapPoint类
class MapPoint {
public:
    unsigned long mnId;
    bool mbBad;
    Eigen::Vector3f mWorldPos;
    KeyFrame* mpRefKF;
    unsigned long mnCorrectedByKF;
    unsigned long mnCorrectedReference;

    // 方法
    bool isBad() const { return mbBad; }
    Eigen::Vector3f GetWorldPos() const { return mWorldPos; }
    void SetWorldPos(const Eigen::Vector3f& pos) { mWorldPos = pos; }
    KeyFrame* GetReferenceKeyFrame() { return mpRefKF; }
    void UpdateNormalAndDepth() { /* 如需实现 */ }
};

// 简化的Map类
class Map {
public:
    unsigned long mnId;
    unsigned long mnInitKFid;
    unsigned long mnMaxKFid;
    bool mbImuInitialized;
    std::mutex mMutexMapUpdate;

    // 方法
    int GetId() const { return mnId; }
    unsigned long GetInitKFid() const { return mnInitKFid; }
    unsigned long GetMaxKFid() const { return mnMaxKFid; }
    bool IsInertial() const { return false; }  // 简化版
    bool isImuInitialized() const { return mbImuInitialized; }
    void IncreaseChangeIndex() { /* 如需实现 */ }
    
    // 数据访问
    std::vector<KeyFrame*> GetAllKeyFrames() const { return mvpKeyFrames; }
    std::vector<MapPoint*> GetAllMapPoints() const { return mvpMapPoints; }
    int KeyFramesInMap() const { return mvpKeyFrames.size(); }
    int MapPointsInMap() const { return mvpMapPoints.size(); }

    // 设置数据
    void SetKeyFrames(const std::vector<KeyFrame*>& vpKFs) { mvpKeyFrames = vpKFs; }
    void SetMapPoints(const std::vector<MapPoint*>& vpMPs) { mvpMapPoints = vpMPs; }

private:
    std::vector<KeyFrame*> mvpKeyFrames;
    std::vector<MapPoint*> mvpMapPoints;
};

// 用于更好数值稳定性的角轴转换
template <typename T>
void StabilizedRotationToMatrix(const T* angle_axis, T* R) {
    T angle = sqrt(angle_axis[0]*angle_axis[0] + 
                  angle_axis[1]*angle_axis[1] + 
                  angle_axis[2]*angle_axis[2]);
    
    // 特别处理小角度避免除以零
    if (angle < T(1e-10)) {
        // 对小角度使用一阶近似
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
    
    // 标准罗德里格斯公式
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

// 用于CERES的SE3边代价函数
struct SE3EdgeCostFunction {
    SE3EdgeCostFunction(const Eigen::Matrix3d& R_meas, const Eigen::Vector3d& t_meas, double weight = 1.0) 
        : R_measurement(R_meas), t_measurement(t_meas), weight_(weight) {
        // 验证测量值的正交性
        double det = R_measurement.determinant();
        if(std::abs(det - 1.0) > 1e-6) {
            std::cout << "Warning: Non-orthogonal rotation matrix detected. Det = " << det << std::endl;
            // 使用SVD进行正交化
            Eigen::JacobiSVD<Eigen::Matrix3d> svd(R_measurement, Eigen::ComputeFullU | Eigen::ComputeFullV);
            R_measurement = svd.matrixU() * svd.matrixV().transpose();
        }
    }
    
    template <typename T>
    bool operator()(const T* const pose_i, const T* const pose_j, T* residuals) const {
        // 提取参数（角轴+平移）
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> rot_i(pose_i);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> trans_i(pose_i + 3);
        
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> rot_j(pose_j);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> trans_j(pose_j + 3);
        
        // 使用StabilizedRotationToMatrix函数将角轴转换为旋转矩阵
        T R_i[9], R_j[9];
        StabilizedRotationToMatrix(rot_i.data(), R_i);
        StabilizedRotationToMatrix(rot_j.data(), R_j);
        
        // 转换为模板类型的测量值
        Eigen::Matrix<T, 3, 3> R_meas = R_measurement.cast<T>();
        Eigen::Matrix<T, 3, 1> t_meas = t_measurement.cast<T>();
        
        // 创建旋转矩阵
        Eigen::Matrix<T, 3, 3> R_i_mat, R_j_mat;
        R_i_mat << R_i[0], R_i[1], R_i[2], 
                   R_i[3], R_i[4], R_i[5], 
                   R_i[6], R_i[7], R_i[8];
        
        R_j_mat << R_j[0], R_j[1], R_j[2], 
                   R_j[3], R_j[4], R_j[5], 
                   R_j[6], R_j[7], R_j[8];
        
        // 计算预测的相对变换
        Eigen::Matrix<T, 3, 3> R_i_inv = R_i_mat.transpose();
        Eigen::Matrix<T, 3, 1> t_i_inv = -R_i_inv * trans_i;
        
        Eigen::Matrix<T, 3, 3> R_ji_pred = R_j_mat * R_i_inv;
        Eigen::Matrix<T, 3, 1> t_ji_pred = R_j_mat * t_i_inv + trans_j;
        
        // 计算旋转误差: log(R_meas^T * R_ji_pred)
        Eigen::Matrix<T, 3, 3> R_error_mat = R_meas.transpose() * R_ji_pred;
        
        // 更稳定的对数映射实现
        Eigen::Matrix<T, 3, 1> rot_error;
        LogSO3(R_error_mat, rot_error);
        
        // 计算平移误差
        Eigen::Matrix<T, 3, 1> t_error = t_ji_pred - t_meas;
        
        // 应用权重
        T sqrt_weight = sqrt(T(weight_));
        
        // 填充残差
        residuals[0] = sqrt_weight * rot_error[0];
        residuals[1] = sqrt_weight * rot_error[1];
        residuals[2] = sqrt_weight * rot_error[2];
        residuals[3] = sqrt_weight * t_error[0];
        residuals[4] = sqrt_weight * t_error[1];
        residuals[5] = sqrt_weight * t_error[2];
        
        return true;
    }
    
    // 更稳定的SO3对数映射
    template <typename T>
    static void LogSO3(const Eigen::Matrix<T, 3, 3>& R, Eigen::Matrix<T, 3, 1>& log_r) {
        T cos_theta = (R.trace() - T(1.0)) * T(0.5);
        cos_theta = cos_theta < T(-1.0) ? T(-1.0) : (cos_theta > T(1.0) ? T(1.0) : cos_theta);
        
        T theta = acos(cos_theta);
        
        if (theta < T(1e-10)) {
            // 近似为零旋转
            log_r.setZero();
            return;
        }
        
        // 计算sin(theta)
        T sin_theta = sin(theta);
        
        if (sin_theta < T(1e-10)) {
            // 处理特殊情况：180度旋转
            Eigen::Matrix<T, 3, 3> W = (R - R.transpose()) * T(0.5);
            log_r[0] = W(2, 1);
            log_r[1] = W(0, 2);
            log_r[2] = W(1, 0);
            
            T norm = log_r.norm();
            if (norm > T(1e-10)) {
                log_r = log_r / norm * theta;
            } else {
                // 特别处理：找到R+I的行最大值来确定旋转轴
                Eigen::Matrix<T, 3, 3> RplusI = R + Eigen::Matrix<T, 3, 3>::Identity();
                Eigen::Matrix<T, 3, 1> col;
                int max_col = 0;
                col = RplusI.col(0);
                
                for (int i = 1; i < 3; ++i) {
                    if (RplusI.col(i).norm() > col.norm()) {
                        col = RplusI.col(i);
                        max_col = i;
                    }
                }
                
                col.normalize();
                log_r = col * theta;
            }
        } else {
            // 标准情况
            T scale = theta / (T(2.0) * sin_theta);
            Eigen::Matrix<T, 3, 3> W = (R - R.transpose()) * scale;
            log_r[0] = W(2, 1);
            log_r[1] = W(0, 2);
            log_r[2] = W(1, 0);
        }
    }
    
    static ceres::CostFunction* Create(const Eigen::Matrix3d& R_meas, 
                                      const Eigen::Vector3d& t_meas, 
                                      double weight = 1.0) {
        return new ceres::AutoDiffCostFunction<SE3EdgeCostFunction, 6, 6, 6>(
            new SE3EdgeCostFunction(R_meas, t_meas, weight));
    }
    
    Eigen::Matrix3d R_measurement;
    Eigen::Vector3d t_measurement;
    double weight_;
};

// Ceres 2.2中的自定义SE3流形
class SE3Manifold : public ceres::Manifold {
public:
    virtual ~SE3Manifold() {}

    // Plus操作: x_plus_delta = Plus(x, delta)
    virtual bool Plus(const double* x, const double* delta, double* x_plus_delta) const {
        // x为[角轴(3), 平移(3)]
        // delta为[角轴delta(3), 平移delta(3)]
        // x_plus_delta为结果
        
        // 处理旋转(角轴)
        Eigen::Map<const Eigen::Vector3d> angleAxis(x);
        Eigen::Map<const Eigen::Vector3d> delta_angleAxis(delta);
        Eigen::Map<Eigen::Vector3d> result_angleAxis(x_plus_delta);
        
        // 处理零角轴情况
        if(angleAxis.norm() < 1e-10) {
            result_angleAxis = delta_angleAxis;
        } else if(delta_angleAxis.norm() < 1e-10) {
            result_angleAxis = angleAxis;
        } else {
            // 转换为四元数，相乘，再转回角轴
            Eigen::AngleAxisd aa1(angleAxis.norm(), angleAxis.normalized());
            Eigen::AngleAxisd aa2(delta_angleAxis.norm(), delta_angleAxis.normalized());
            
            Eigen::Quaterniond q1(aa1);
            Eigen::Quaterniond q2(aa2);
            Eigen::Quaterniond q_res = q2 * q1;
            q_res.normalize();
            
            Eigen::AngleAxisd aa_res(q_res);
            if(aa_res.angle() < 1e-10) {
                result_angleAxis = Eigen::Vector3d::Zero();
            } else {
                result_angleAxis = aa_res.angle() * aa_res.axis();
            }
        }
        
        // 处理平移
        Eigen::Map<const Eigen::Vector3d> translation(x + 3);
        Eigen::Map<const Eigen::Vector3d> delta_translation(delta + 3);
        Eigen::Map<Eigen::Vector3d> result_translation(x_plus_delta + 3);
        
        // 计算更新后的平移
        if(delta_angleAxis.norm() > 1e-10) {
            Eigen::AngleAxisd aa2(delta_angleAxis.norm(), delta_angleAxis.normalized());
            Eigen::Matrix3d R_delta = aa2.toRotationMatrix();
            result_translation = R_delta * translation + delta_translation;
        } else {
            result_translation = translation + delta_translation;
        }
        
        return true;
    }

    // PlusJacobian
    virtual bool PlusJacobian(const double* x, double* jacobian) const {
        // SE3的Jacobian为6x6
        Eigen::Map<Eigen::Matrix<double, 6, 6, Eigen::RowMajor>> J(jacobian);
        J.setZero();
        
        // 旋转块：对小变化为单位矩阵
        J.block<3, 3>(0, 0).setIdentity();
        
        // 平移块：单位矩阵
        J.block<3, 3>(3, 3).setIdentity();
        
        // 交叉项：旋转影响平移
        Eigen::Map<const Eigen::Vector3d> angleAxis(x);
        
        if(angleAxis.norm() > 1e-10) {
            Eigen::AngleAxisd aa(angleAxis.norm(), angleAxis.normalized());
            Eigen::Matrix3d R = aa.toRotationMatrix();
            
            Eigen::Map<const Eigen::Vector3d> t(x + 3);
            Eigen::Matrix3d skew;
            skew << 0, -t(2), t(1),
                    t(2), 0, -t(0),
                    -t(1), t(0), 0;
                    
            J.block<3, 3>(3, 0) = -R * skew;
        }
        
        return true;
    }
    
    // Minus操作: y_minus_x = Minus(y, x)
    virtual bool Minus(const double* y, const double* x, double* y_minus_x) const {
        // y和x是[角轴(3), 平移(3)]
        // y_minus_x是切空间差异[旋转差异(3), 平移差异(3)]
        
        // 转换为SE3矩阵
        Eigen::Map<const Eigen::Vector3d> x_aa(x);
        Eigen::Map<const Eigen::Vector3d> x_t(x + 3);
        Eigen::Map<const Eigen::Vector3d> y_aa(y);
        Eigen::Map<const Eigen::Vector3d> y_t(y + 3);
        
        // 处理零角轴情况
        if(x_aa.norm() < 1e-10 && y_aa.norm() < 1e-10) {
            Eigen::Map<Eigen::Vector3d> result_aa(y_minus_x);
            result_aa = Eigen::Vector3d::Zero();
            Eigen::Map<Eigen::Vector3d> result_t(y_minus_x + 3);
            result_t = y_t - x_t;
            return true;
        }
        
        // 将角轴转换为旋转矩阵
        Eigen::Matrix3d R_x = Eigen::Matrix3d::Identity();
        Eigen::Matrix3d R_y = Eigen::Matrix3d::Identity();
        
        if(x_aa.norm() > 1e-10) {
            Eigen::AngleAxisd x_rotation(x_aa.norm(), x_aa.normalized());
            R_x = x_rotation.toRotationMatrix();
        }
        
        if(y_aa.norm() > 1e-10) {
            Eigen::AngleAxisd y_rotation(y_aa.norm(), y_aa.normalized());
            R_y = y_rotation.toRotationMatrix();
        }
        
        // 计算相对旋转R_rel = R_y * R_x^T
        Eigen::Matrix3d R_rel = R_y * R_x.transpose();
        
        // 转换回角轴表示
        Eigen::AngleAxisd aa_rel(R_rel);
        Eigen::Map<Eigen::Vector3d> result_aa(y_minus_x);
        
        if(aa_rel.angle() < 1e-10) {
            result_aa = Eigen::Vector3d::Zero();
        } else {
            result_aa = aa_rel.angle() * aa_rel.axis();
        }
        
        // 计算相对平移: R_x^T * (y_t - x_t)
        Eigen::Map<Eigen::Vector3d> result_t(y_minus_x + 3);
        result_t = R_x.transpose() * (y_t - x_t);
        
        return true;
    }
    
    // MinusJacobian计算Minus(y, x)关于x的导数
    virtual bool MinusJacobian(const double* x, double* jacobian) const {
        // 计算y_minus_x关于x的Jacobian
        Eigen::Map<Eigen::Matrix<double, 6, 6, Eigen::RowMajor>> J(jacobian);
        J.setZero();
        
        // 对于小扰动，我们可以对旋转使用负单位矩阵近似
        J.block<3, 3>(0, 0) = -Eigen::Matrix3d::Identity();
        
        // 将角轴转换为旋转矩阵
        Eigen::Map<const Eigen::Vector3d> x_aa(x);
        Eigen::Matrix3d R_x = Eigen::Matrix3d::Identity();
        
        if(x_aa.norm() > 1e-10) {
            Eigen::AngleAxisd aa(x_aa.norm(), x_aa.normalized());
            R_x = aa.toRotationMatrix();
        }
        
        // 对于平移，我们需要-R_x^T
        J.block<3, 3>(3, 3) = -R_x.transpose();
        
        // 交叉项：平移受旋转影响
        Eigen::Map<const Eigen::Vector3d> t(x + 3);
        Eigen::Matrix3d skew;
        skew << 0, -t(2), t(1),
                t(2), 0, -t(0),
                -t(1), t(0), 0;
                
        // 注意：这对小角度变化是一个近似
        J.block<3, 3>(3, 0) = R_x.transpose() * skew;
        
        return true;
    }

    virtual bool RightMultiplyByPlusJacobian(const double* x, 
                                           const int num_rows,
                                           const double* ambient_matrix,
                                           double* tangent_matrix) const {
        // 这个方法计算tangent_matrix = ambient_matrix * plus_jacobian
        
        // 使用ceres::Manifold提供的默认实现
        double* plus_jacobian = new double[6 * 6];
        PlusJacobian(x, plus_jacobian);
        
        // 执行矩阵乘法
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

    virtual int AmbientSize() const { return 6; }  // 环境空间维度
    virtual int TangentSize() const { return 6; }  // 切空间维度
};

// 正交化位姿函数
void OrthogonalizeSE3Poses(std::vector<Eigen::Quaterniond>& vRotations) {
    for (size_t i = 0; i < vRotations.size(); ++i) {
        if (vRotations[i].coeffs().norm() > 0) {
            vRotations[i].normalize();
        }
    }
}

// 保存轨迹到文件
void SaveTrajectory(const std::string& filename, const std::vector<KeyFrame*>& vpKFs, 
                    const std::vector<double*>& vPoseParams = std::vector<double*>()) {
    std::ofstream f;
    f.open(filename.c_str());
    f << std::fixed;
    
    for(size_t i=0; i<vpKFs.size(); i++) {
        KeyFrame* pKF = vpKFs[i];
        if(pKF->isBad()) continue;
        
        if(!vPoseParams.empty() && vPoseParams[pKF->mnId]) {
            // 从参数块保存
            double* params = vPoseParams[pKF->mnId];
            
            // 将角轴转换为旋转矩阵
            Eigen::Vector3d rot(params[0], params[1], params[2]);
            Eigen::Vector3d trans(params[3], params[4], params[5]);
            
            double angle = rot.norm();
            if(angle > 1e-10) {
                Eigen::Vector3d axis = rot / angle;
                Eigen::AngleAxisd aa(angle, axis);
                Eigen::Matrix3d Rcw = aa.toRotationMatrix();
                
                // 保存为Twc (Tcw的逆)
                Eigen::Matrix3d Rwc = Rcw.transpose();
                Eigen::Vector3d twc = -Rwc * trans;
                
                Eigen::Quaterniond q(Rwc);
                
                f << pKF->mTimeStamp << " " 
                  << twc.x() << " " << twc.y() << " " << twc.z() << " "
                  << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
            } else {
                // 退化情况 - 单位旋转
                Eigen::Vector3d twc = -trans;
                f << pKF->mTimeStamp << " " 
                  << twc.x() << " " << twc.y() << " " << twc.z() << " "
                  << 0.0 << " " << 0.0 << " " << 0.0 << " " << 1.0 << std::endl;
            }
        } else {
            // 直接从KeyFrame位姿保存
            Eigen::Matrix3f Rwc;
            Eigen::Vector3f twc;
            pKF->GetPoseInverse(Rwc, twc);
            
            Eigen::Quaternionf q(Rwc);
            
            f << pKF->mTimeStamp << " " 
              << twc.x() << " " << twc.y() << " " << twc.z() << " "
              << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
        }
    }
    
    f.close();
    std::cout << "Saved trajectory to " << filename << std::endl;
}

// 从TUM格式文件加载轨迹
std::vector<KeyFrame*> LoadTUMTrajectory(const std::string& filename) {
    std::vector<KeyFrame*> vpKFs;
    std::ifstream f(filename);
    
    if(!f.is_open()) {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return vpKFs;
    }
    
    std::string line;
    unsigned long id = 0;
    
    while(std::getline(f, line)) {
        std::istringstream iss(line);
        double timestamp, tx, ty, tz, qx, qy, qz, qw;
        
        if(!(iss >> timestamp >> tx >> ty >> tz >> qx >> qy >> qz >> qw)) {
            continue;  // Skip invalid lines
        }
        
        // 创建新的关键帧
        KeyFrame* pKF = new KeyFrame();
        pKF->mnId = id++;
        pKF->mTimeStamp = timestamp;
        
        // 设置位姿：Twc -> Tcw
        Eigen::Quaterniond q(qw, qx, qy, qz);
        q.normalize(); // 确保单位四元数
        
        Eigen::Matrix3d Rwc = q.toRotationMatrix();
        Eigen::Vector3d twc(tx, ty, tz);
        
        // 计算Tcw = inv(Twc)
        Eigen::Matrix3d Rcw = Rwc.transpose();
        Eigen::Vector3d tcw = -Rcw * twc;
        
        pKF->SetPose(Rcw.cast<float>(), tcw.cast<float>());
        pKF->mbBad = false;
        
        vpKFs.push_back(pKF);
    }
    
    f.close();
    std::cout << "Loaded " << vpKFs.size() << " keyframes from " << filename << std::endl;
    
    // 设置额外的连接关系（简化版）
    if(vpKFs.size() > 1) {
        for(size_t i = 1; i < vpKFs.size(); i++) {
            // 设置父子关系
            vpKFs[i]->mpParent = vpKFs[i-1];
            
            // 添加共视关系（相邻帧）
            int weight = 100;  // 默认权重
            if(i > 0) {
                vpKFs[i]->mvpOrderedConnectedKeyFrames.push_back(vpKFs[i-1]);
                vpKFs[i]->mvOrderedWeights.push_back(weight);
                
                vpKFs[i-1]->mvpOrderedConnectedKeyFrames.push_back(vpKFs[i]);
                vpKFs[i-1]->mvOrderedWeights.push_back(weight);
            }
        }
    }
    
    return vpKFs;
}

// 改进的OptimizeEssentialGraphCeres函数
void OptimizeEssentialGraphCeres(Map* pMap, KeyFrame* pLoopKF, KeyFrame* pCurKF,
                                const KeyFrameAndPose& NonCorrectedSE3,
                                const KeyFrameAndPose& CorrectedSE3,
                                const std::map<KeyFrame*, std::set<KeyFrame*>>& LoopConnections,
                                const bool& /* bFixScale */) {  // 未使用的参数
    
    std::cout << "Starting OptimizeEssentialGraphCeres..." << std::endl;
    
    // 输出路径
    std::string outputDir = "/Datasets/CERES_Work/output/";
    
    // 设置CERES优化器
    ceres::Problem problem;
    
    const std::vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();
    const std::vector<MapPoint*> vpMPs = pMap->GetAllMapPoints();
    
    const unsigned int nMaxKFid = pMap->GetMaxKFid();
    
    // 创建向量存储旋转和平移
    std::vector<Eigen::Quaterniond> vRotations(nMaxKFid+1);
    std::vector<Eigen::Vector3d> vTranslations(nMaxKFid+1);
    std::vector<double*> vPoseParams(nMaxKFid+1, nullptr);  // Ceres的参数
    
    // 存储优化后的姿态
    std::vector<SE3Pose> vCorrectedPoses(nMaxKFid+1);
    
    const int minFeat = 100; // 与原始代码相同
    
    std::cout << "Creating parameter blocks for " << vpKFs.size() << " keyframes" << std::endl;
    
    // 创建自定义SE3流形
    ceres::Manifold* se3_manifold = new SE3Manifold();
    
    // 为每个关键帧添加参数块 - 类似于原始代码的VertexSim3
    for(KeyFrame* pKF : vpKFs) {
        if(pKF->isBad())
            continue;
        
        const unsigned long nIDi = pKF->mnId;  // 修改为unsigned long以匹配类型
        double* pose_params = new double[6];  // 3用于旋转的角轴，3用于平移
        
        // 尝试先从CorrectedSE3获取姿态 - 类似于原始代码中的CorrectedSim3
        KeyFrameAndPose::const_iterator it = CorrectedSE3.find(pKF);
        if(it != CorrectedSE3.end()) {
            // 如果在CorrectedSE3中找到
            const SE3Pose& se3pose = it->second;
            Eigen::Quaterniond q = se3pose.first;
            q.normalize(); // 确保归一化
            Eigen::Vector3d t = se3pose.second;
            
            // 存储
            vRotations[nIDi] = q;
            vTranslations[nIDi] = t;
            
            // 转换为角轴
            Eigen::AngleAxisd aa(q);
            if(aa.angle() < 1e-10) {
                // 处理极小旋转的边缘情况
                pose_params[0] = 0.0;
                pose_params[1] = 0.0;
                pose_params[2] = 0.0;
            } else {
                pose_params[0] = aa.angle() * aa.axis()[0];
                pose_params[1] = aa.angle() * aa.axis()[1];
                pose_params[2] = aa.angle() * aa.axis()[2];
            }
            pose_params[3] = t[0];
            pose_params[4] = t[1];
            pose_params[5] = t[2];
        } else {
            // 如果未找到，使用当前关键帧姿态 - 类似于原始代码中的else分支
            Eigen::Matrix3f Rcw;
            Eigen::Vector3f tcw;
            pKF->GetPose(Rcw, tcw);
            
            // 转换为double
            Eigen::Matrix3d Rcw_d = Rcw.cast<double>();
            Eigen::Vector3d tcw_d = tcw.cast<double>();
            
            // 从旋转矩阵创建四元数
            Eigen::Quaterniond q(Rcw_d);
            q.normalize();
            
            // 存储
            vRotations[nIDi] = q;
            vTranslations[nIDi] = tcw_d;
            
            // 转换为角轴
            Eigen::AngleAxisd aa(q);
            if(aa.angle() < 1e-10) {
                pose_params[0] = 0.0;
                pose_params[1] = 0.0;
                pose_params[2] = 0.0;
            } else {
                pose_params[0] = aa.angle() * aa.axis()[0];
                pose_params[1] = aa.angle() * aa.axis()[1];
                pose_params[2] = aa.angle() * aa.axis()[2];
            }
            pose_params[3] = tcw_d[0];
            pose_params[4] = tcw_d[1];
            pose_params[5] = tcw_d[2];
        }
        
        vPoseParams[nIDi] = pose_params;
        
        // 添加带SE3流形的参数块
        problem.AddParameterBlock(pose_params, 6, se3_manifold);
        
        // 修复初始关键帧 - 类似于原始代码的setFixed
        if(pKF->mnId == pMap->GetInitKFid()) {
            problem.SetParameterBlockConstant(pose_params);
            std::cout << "Fixed initial keyframe: " << pKF->mnId << std::endl;
        }
    }
    
    // 确保所有位姿的正交性
    OrthogonalizeSE3Poses(vRotations);
    
    std::cout << "All parameter blocks created successfully" << std::endl;
    
    // 记录插入的边，避免重复 - 与原始代码相同
    std::set<std::pair<long unsigned int, long unsigned int>> sInsertedEdges;
    
    // 添加回环连接边 - 类似于原始代码的Set Loop edges部分
    int loopConnectionEdgeCount = 0;
    
    for(const auto& mit : LoopConnections) {
        KeyFrame* pKF = mit.first;
        const long unsigned int nIDi = pKF->mnId;
        const std::set<KeyFrame*> &spConnections = mit.second;
        
        double* poseParams_i = vPoseParams[nIDi];
        if(!poseParams_i) continue;
    
        // 获取逆变换 Swi
        Eigen::Quaterniond q_i = vRotations[nIDi];
        Eigen::Vector3d t_i = vTranslations[nIDi];
        
        Eigen::Quaterniond q_i_inv = q_i.conjugate();
    
        for(KeyFrame* pKFj : spConnections) {
            const long unsigned int nIDj = pKFj->mnId;
            
            // 与原始代码相同的条件检查
            if((nIDi != pCurKF->mnId || nIDj != pLoopKF->mnId) && pKF->GetWeight(pKFj) < minFeat)
                continue;
                
            double* poseParams_j = vPoseParams[nIDj];
            if(!poseParams_j) continue;
            
            // 获取关键帧j的位姿
            Eigen::Quaterniond q_j = vRotations[nIDj];
            Eigen::Vector3d t_j = vTranslations[nIDj];
            
            // 计算相对变换 Sji = Sjw * Swi
            Eigen::Quaterniond q_ji = q_j * q_i_inv;
            Eigen::Vector3d t_ji = t_j - q_j * q_i_inv * t_i;
            
            Eigen::Matrix3d R_ji = q_ji.toRotationMatrix();
            
            // 添加边约束 - 等同于原始代码中的EdgeSim3
            ceres::CostFunction* edge_se3 = SE3EdgeCostFunction::Create(R_ji, t_ji, 100.0);
            
            // 添加稳健核函数
            ceres::LossFunction* loss_function = new ceres::HuberLoss(1.0);
            problem.AddResidualBlock(edge_se3, loss_function, poseParams_i, poseParams_j);
            
            loopConnectionEdgeCount++;
            sInsertedEdges.insert(std::make_pair(std::min(nIDi, nIDj), std::max(nIDi, nIDj)));
        }
    }
    std::cout << "Added " << loopConnectionEdgeCount << " loop connection edges" << std::endl;
    
    // 添加普通边 - 类似于原始代码的Set normal edges部分
    int spanningEdgeCount = 0;
    int existingLoopEdgeCount = 0;
    int covisibilityEdgeCount = 0;
    int imuEdgeCount = 0;
    
    for(KeyFrame* pKF : vpKFs) {
        if(pKF->isBad())
            continue;
            
        const unsigned long nIDi = pKF->mnId;
        double* poseParams_i = vPoseParams[nIDi];
        if(!poseParams_i) continue;
        
        // 获取Swi (逆变换) - 与原始代码相同的逻辑
        Eigen::Quaterniond q_i = vRotations[nIDi];
        Eigen::Vector3d t_i = vTranslations[nIDi];
        
        Eigen::Quaterniond q_i_inv = q_i.conjugate();
        Eigen::Vector3d t_wi_inv = -(q_i_inv * t_i);
        
        // 检查是否在NonCorrectedSE3中有不同的位姿 - 与原始代码相同的逻辑
        Eigen::Quaterniond q_wi_inv = q_i_inv;
        
        auto iti = NonCorrectedSE3.find(pKF);
        if(iti != NonCorrectedSE3.end()) {
            const SE3Pose& se3pose = iti->second;
            q_wi_inv = se3pose.first.conjugate();
            t_wi_inv = -(q_wi_inv * se3pose.second);
        }
        
        // 添加生成树边 - 与原始代码相同的逻辑
        KeyFrame* pParentKF = pKF->GetParent();
        if(pParentKF && !pParentKF->isBad()) {
            unsigned long nIDj = pParentKF->mnId;
            double* poseParams_j = vPoseParams[nIDj];
            if(!poseParams_j) continue;
            
            // 获取父节点位姿 Sjw
            Eigen::Quaterniond q_j = vRotations[nIDj];
            Eigen::Vector3d t_j = vTranslations[nIDj];
            
            // 检查是否在NonCorrectedSE3中有不同的位姿
            auto itj = NonCorrectedSE3.find(pParentKF);
            if(itj != NonCorrectedSE3.end()) {
                const SE3Pose& se3pose = itj->second;
                q_j = se3pose.first;
                t_j = se3pose.second;
            }
            
            // 计算相对变换 Sji = Sjw * Swi
            Eigen::Quaterniond q_ji = q_j * q_wi_inv;
            Eigen::Vector3d t_ji = t_j - q_j * q_wi_inv * t_i;
            
            Eigen::Matrix3d R_ji = q_ji.toRotationMatrix();
            
            // 添加边约束
            ceres::CostFunction* edge_se3 = SE3EdgeCostFunction::Create(R_ji, t_ji, 500.0);
            
            // 添加稳健核函数
            ceres::LossFunction* loss_function = new ceres::HuberLoss(1.0);
            problem.AddResidualBlock(edge_se3, loss_function, poseParams_i, poseParams_j);
            
            spanningEdgeCount++;
        }
        
        // 添加现有回环边 - 与原始代码相同
        const std::set<KeyFrame*> sLoopEdges = pKF->GetLoopEdges();
        for(KeyFrame* pLKF : sLoopEdges) {
            if(pLKF->mnId < pKF->mnId) {  // 确保每条边只添加一次
                double* poseParams_l = vPoseParams[pLKF->mnId];
                if(!poseParams_l) continue;
                
                // 获取回环关键帧位姿 Slw
                Eigen::Quaterniond q_l = vRotations[pLKF->mnId];
                Eigen::Vector3d t_l = vTranslations[pLKF->mnId];
                
                // 检查是否在NonCorrectedSE3中有不同的位姿
                auto itl = NonCorrectedSE3.find(pLKF);
                if(itl != NonCorrectedSE3.end()) {
                    const SE3Pose& se3pose = itl->second;
                    q_l = se3pose.first;
                    t_l = se3pose.second;
                }
                
                // 计算相对变换 Sli = Slw * Swi
                Eigen::Quaterniond q_li = q_l * q_wi_inv;
                Eigen::Vector3d t_li = t_l - q_l * q_wi_inv * t_i;
                
                Eigen::Matrix3d R_li = q_li.toRotationMatrix();
                
                // 添加边约束
                ceres::CostFunction* edge_se3 = SE3EdgeCostFunction::Create(R_li, t_li, 100.0);
                ceres::LossFunction* loss_function = new ceres::HuberLoss(1.0);
                problem.AddResidualBlock(edge_se3, loss_function, poseParams_i, poseParams_l);
                
                existingLoopEdgeCount++;
            }
        }
        
        // 添加共视图边（权重调整） - 与原始代码相同
        const std::vector<KeyFrame*> vpConnectedKFs = pKF->GetCovisiblesByWeight(minFeat);
        for(KeyFrame* pKFn : vpConnectedKFs) {
            if(pKFn && pKFn != pParentKF && !pKF->hasChild(pKFn)) {
                if(!pKFn->isBad() && pKFn->mnId < pKF->mnId) {
                    // 检查是否已经添加过这条边
                    if(sInsertedEdges.count(std::make_pair(std::min(pKF->mnId, pKFn->mnId), 
                                                   std::max(pKF->mnId, pKFn->mnId))))
                        continue;
                        
                    double* poseParams_n = vPoseParams[pKFn->mnId];
                    if(!poseParams_n) continue;
                    
                    // 获取共视关键帧位姿 Snw
                    Eigen::Quaterniond q_n = vRotations[pKFn->mnId];
                    Eigen::Vector3d t_n = vTranslations[pKFn->mnId];
                    
                    // 检查是否在NonCorrectedSE3中有不同的位姿
                    auto itn = NonCorrectedSE3.find(pKFn);
                    if(itn != NonCorrectedSE3.end()) {
                        const SE3Pose& se3pose = itn->second;
                        q_n = se3pose.first;
                        t_n = se3pose.second;
                    }
                    
                    // 计算相对变换 Sni = Snw * Swi
                    Eigen::Quaterniond q_ni = q_n * q_wi_inv;
                    Eigen::Vector3d t_ni = t_n - q_n * q_wi_inv * t_i;
                    
                    Eigen::Matrix3d R_ni = q_ni.toRotationMatrix();
                    
                    // 添加边约束 - 权重基于共视关系
                    double weight = pKF->GetWeight(pKFn) * 0.5;
                    ceres::CostFunction* edge_se3 = SE3EdgeCostFunction::Create(R_ni, t_ni, weight);
                    ceres::LossFunction* loss_function = new ceres::HuberLoss(1.0);
                    problem.AddResidualBlock(edge_se3, loss_function, poseParams_i, poseParams_n);
                    
                    covisibilityEdgeCount++;
                    sInsertedEdges.insert(std::make_pair(std::min(pKF->mnId, pKFn->mnId), 
                                                 std::max(pKF->mnId, pKFn->mnId)));
                }
            }
        }
        
        // 添加IMU边（如果有） - 与原始代码相同
        if(pKF->bImu && pKF->mPrevKF) {
            double* poseParams_p = vPoseParams[pKF->mPrevKF->mnId];
            if(!poseParams_p) continue;
            
            // 获取前一关键帧位姿 Spw
            Eigen::Quaterniond q_p = vRotations[pKF->mPrevKF->mnId];
            Eigen::Vector3d t_p = vTranslations[pKF->mPrevKF->mnId];
            
            // 检查是否在NonCorrectedSE3中有不同的位姿
            auto itp = NonCorrectedSE3.find(pKF->mPrevKF);
            if(itp != NonCorrectedSE3.end()) {
                const SE3Pose& se3pose = itp->second;
                q_p = se3pose.first;
                t_p = se3pose.second;
            }
            
            // 计算相对变换 Spi = Spw * Swi
            Eigen::Quaterniond q_pi = q_p * q_wi_inv;
            Eigen::Vector3d t_pi = t_p - q_p * q_wi_inv * t_i;
            
            Eigen::Matrix3d R_pi = q_pi.toRotationMatrix();
            
            // 添加边约束
            ceres::CostFunction* edge_se3 = SE3EdgeCostFunction::Create(R_pi, t_pi, 200.0);
            ceres::LossFunction* loss_function = new ceres::HuberLoss(1.0);
            problem.AddResidualBlock(edge_se3, loss_function, poseParams_i, poseParams_p);
            
            imuEdgeCount++;
        }
    }
    
    // 打印边统计
    std::cout << "Added " << spanningEdgeCount << " spanning tree edges" << std::endl;
    std::cout << "Added " << existingLoopEdgeCount << " existing loop edges" << std::endl;
    std::cout << "Added " << covisibilityEdgeCount << " covisibility edges" << std::endl;
    std::cout << "Added " << imuEdgeCount << " IMU edges" << std::endl;
    std::cout << "Total inserted edges: " << sInsertedEdges.size() << std::endl;
    
    // 保存优化前状态
    SaveTrajectory(outputDir + "step0_before_optimization.txt", vpKFs, vPoseParams);
    
    // 配置求解器选项（更强大的设置）
    std::cout << "\nStep 4: Running optimization with improved settings..." << std::endl;
    
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 20;  // 与原始代码相同
    options.function_tolerance = 1e-9;
    options.gradient_tolerance = 1e-10;
    options.parameter_tolerance = 1e-8;
    
    // 更好的初始化策略
    options.initial_trust_region_radius = 1e5;
    options.max_trust_region_radius = 1e8;
    options.min_trust_region_radius = 1e-4;
    
    // 添加更多的内部优化选项 - Ceres 2.2的shared_ptr版本
    options.linear_solver_ordering = std::make_shared<ceres::ParameterBlockOrdering>();
    for(KeyFrame* pKF : vpKFs) {
        if(pKF->isBad()) continue;
        const unsigned long nIDi = pKF->mnId;
        double* params = vPoseParams[nIDi];
        if(params) {
            options.linear_solver_ordering->AddElementToGroup(params, 0);
        }
    }
    
    // 执行优化 - 对应于原始代码的optimizer.optimize(20)
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    
    // 打印详细的优化报告
    std::cout << "\n=== Optimization Report ===" << std::endl;
    std::cout << summary.BriefReport() << std::endl;
    
    // 保存优化后的轨迹
    SaveTrajectory(outputDir + "step1_after_optimization.txt", vpKFs, vPoseParams);
    
    // 更新关键帧位姿 - 与原始代码相同
    std::cout << "\n=== Updating KeyFrame poses ===" << std::endl;
    {
        std::unique_lock<std::mutex> lock(pMap->mMutexMapUpdate);
        
        for(KeyFrame* pKFi : vpKFs) {
            if(pKFi->isBad()) continue;
            
            const unsigned long nIDi = pKFi->mnId;
            double* params = vPoseParams[nIDi];
            
            if(!params) continue;
            
            // 获取优化后的位姿参数
            Eigen::Vector3d rot(params[0], params[1], params[2]);
            Eigen::Vector3d trans(params[3], params[4], params[5]);
            
            // 转换为旋转矩阵
            double angle = rot.norm();
            if(angle > 1e-10) {
                Eigen::AngleAxisd aa(angle, rot.normalized());
                Eigen::Matrix3d R = aa.toRotationMatrix();
                
                // 更新关键帧位姿
                Eigen::Matrix3f Rf = R.cast<float>();
                Eigen::Vector3f tf = trans.cast<float>();
                
                pKFi->SetPose(Rf, tf);
            } else {
                // 单位旋转
                pKFi->SetPose(Eigen::Matrix3f::Identity(), trans.cast<float>());
            }
        }
        
        std::cout << "Updated " << vpKFs.size() << " keyframe poses" << std::endl;
    }
    
    // 保存最终优化后的轨迹
    SaveTrajectory(outputDir + "final_optimized_trajectory.txt", vpKFs);
    
    // 更新MapPoints - 与原始代码类似
    std::cout << "\n=== Updating MapPoints ===" << std::endl;
    for(MapPoint* pMP : vpMPs) {
        if(pMP->isBad())
            continue;
            
        int nIDr;
        if(pMP->mnCorrectedByKF == pCurKF->mnId) {
            nIDr = pMP->mnCorrectedReference;
        } else {
            KeyFrame* pRefKF = pMP->GetReferenceKeyFrame();
            nIDr = pRefKF->mnId;
        }
        
        // 获取原始和优化后的参考关键帧位姿
        double* params_r = vPoseParams[nIDr];
        if(!params_r) continue;
        
        // 获取世界坐标下的点位置
        Eigen::Vector3f P3Dw = pMP->GetWorldPos();
        
        // 转换到相机坐标系下
        Eigen::Matrix3f Rcw;
        Eigen::Vector3f tcw;
        KeyFrame* pKFr = nullptr;
        
        // 查找参考关键帧
        for(KeyFrame* pKF : vpKFs) {
            if(pKF->mnId == nIDr) {
                pKFr = pKF;
                break;
            }
        }
        
        if(pKFr) {
            pKFr->GetPose(Rcw, tcw);
            // 转换到相机坐标系
            Eigen::Vector3f P3Dc = Rcw * P3Dw + tcw;
            
            // 获取优化后的相机位姿
            Eigen::Vector3d rot_r(params_r[0], params_r[1], params_r[2]);
            Eigen::Vector3d trans_r(params_r[3], params_r[4], params_r[5]);
            
            Eigen::Matrix3d R_r;
            if(rot_r.norm() > 1e-10) {
                Eigen::AngleAxisd aa_r(rot_r.norm(), rot_r.normalized());
                R_r = aa_r.toRotationMatrix();
            } else {
                R_r = Eigen::Matrix3d::Identity();
            }
            
            // 使用优化后的位姿将点变换回世界坐标系
            Eigen::Vector3d P3Dc_d = P3Dc.cast<double>();
            Eigen::Matrix3d R_r_inv = R_r.transpose();
            Eigen::Vector3d P3Dw_new = R_r_inv * (P3Dc_d - trans_r);
            
            // 更新地图点位置
            pMP->SetWorldPos(P3Dw_new.cast<float>());
            pMP->UpdateNormalAndDepth();
        }
    }
    
    // 释放内存
    for(double* params : vPoseParams) {
        if(params) delete[] params;
    }
    
    // 删除流形
    delete se3_manifold;
    
    // 增加变更索引
    pMap->IncreaseChangeIndex();
}

// 主函数
int main(int /* argc */, char** /* argv */) {  // 未使用的参数
    // 输入和输出目录
    std::string inputDir = "/Datasets/CERES_Work/input";
    std::string outputDir = "/Datasets/CERES_Work/output";
    system(("mkdir -p " + outputDir).c_str());
    
    // 使用单个TUM格式轨迹文件
    std::string tumFile = "/Datasets/CERES_Work/input/transformed/standard_trajectory_sim3_transformed.txt";
    
    std::cout << "Loading trajectory from TUM format file..." << std::endl;
    
    // 加载轨迹
    std::vector<KeyFrame*> vpKFs = LoadTUMTrajectory(tumFile);
    
    if(vpKFs.empty()) {
        std::cerr << "Failed to load trajectory." << std::endl;
        return -1;
    }
    
    // 创建一个新的Map对象
    Map* pMap = new Map();
    pMap->mnId = 0;
    pMap->mnInitKFid = 0;
    pMap->mnMaxKFid = vpKFs.size() - 1;
    pMap->SetKeyFrames(vpKFs);
    
    // 创建空的MapPoints向量
    std::vector<MapPoint*> vpMPs;
    pMap->SetMapPoints(vpMPs);
    
    // 创建NonCorrectedSE3和CorrectedSE3映射 (空，因为我们只用一个轨迹)
    KeyFrameAndPose NonCorrectedSE3;
    KeyFrameAndPose CorrectedSE3;
    
    // 创建LoopConnections映射 (设置第一帧和最后一帧之间的闭环)
    std::map<KeyFrame*, std::set<KeyFrame*>> LoopConnections;
    
    // 假设第一帧和最后一帧为回环
    if(vpKFs.size() > 10) {
        KeyFrame* pKFFirst = vpKFs[0];
        KeyFrame* pKFLast = vpKFs[vpKFs.size()-1];
        
        // 添加回环连接
        std::set<KeyFrame*> sConnections;
        sConnections.insert(pKFFirst);
        LoopConnections[pKFLast] = sConnections;
        
        // 设置回环边
        pKFFirst->mspLoopEdges.insert(pKFLast);
        pKFLast->mspLoopEdges.insert(pKFFirst);
        
        // 设置重要关键帧
        KeyFrame* pLoopKF = pKFFirst;
        KeyFrame* pCurKF = pKFLast;
        
        std::cout << "Setting LoopKF to KF " << pLoopKF->mnId << " and CurKF to KF " << pCurKF->mnId << std::endl;
        
        // 运行优化
        bool bFixScale = true;  // 对于RGBD/双目系统
        OptimizeEssentialGraphCeres(pMap, pLoopKF, pCurKF, NonCorrectedSE3, CorrectedSE3, LoopConnections, bFixScale);
    } else {
        std::cerr << "Not enough keyframes for optimization." << std::endl;
    }
    
    // 清理内存
    delete pMap;
    for(KeyFrame* pKF : vpKFs)
        delete pKF;
    
    std::cout << "Optimization completed successfully." << std::endl;
    return 0;
}
