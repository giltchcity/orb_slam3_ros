#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <set>
#include <string>
#include <cmath>
#include <limits>
#include <Eigen/Core>
#include <Eigen/Geometry>

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
    
    void GetPose(Eigen::Matrix3f& R, Eigen::Vector3f& t) const {
        R = mRcw;
        t = mtcw;
    }
    
    void SetPose(const Eigen::Matrix3f& R, const Eigen::Vector3f& t) {
        mRcw = R;
        mtcw = t;
    }
    
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

    bool isBad() const { return mbBad; }
    Eigen::Vector3f GetWorldPos() const { return mWorldPos; }
    void SetWorldPos(const Eigen::Vector3f& pos) { mWorldPos = pos; }
    KeyFrame* GetReferenceKeyFrame() { return mpRefKF; }
};

// Simplified Map class
class Map {
public:
    unsigned long mnId;
    unsigned long mnInitKFid;
    unsigned long mnMaxKFid;
    bool mbImuInitialized;
    
    int GetId() const { return mnId; }
    unsigned long GetInitKFid() const { return mnInitKFid; }
    unsigned long GetMaxKFid() const { return mnMaxKFid; }
    bool IsInertial() const { return false; }  // Simplified
    bool isImuInitialized() const { return mbImuInitialized; }
    
    std::vector<KeyFrame*> GetAllKeyFrames() const { return mvpKeyFrames; }
    std::vector<MapPoint*> GetAllMapPoints() const { return mvpMapPoints; }
    int KeyFramesInMap() const { return mvpKeyFrames.size(); }
    int MapPointsInMap() const { return mvpMapPoints.size(); }

    void SetKeyFrames(const std::vector<KeyFrame*>& vpKFs) { mvpKeyFrames = vpKFs; }
    void SetMapPoints(const std::vector<MapPoint*>& vpMPs) { mvpMapPoints = vpMPs; }

private:
    std::vector<KeyFrame*> mvpKeyFrames;
    std::vector<MapPoint*> mvpMapPoints;
};

// Function to convert quaternion and translation from file to SE3Pose
SE3Pose ConvertToSE3(double scale, double tx, double ty, double tz, double qx, double qy, double qz, double qw) {
    Eigen::Quaterniond q(qw, qx, qy, qz);
    Eigen::Vector3d t(tx, ty, tz);
    
    // Normalize quaternion
    if(std::abs(q.norm() - 1.0) > 1e-6) {
        std::cout << "Warning: Quaternion not normalized, norm = " << q.norm() << std::endl;
        q.normalize();
    }
    
    // Apply scale to translation for fixed scale case
    t = t / scale;
    
    return std::make_pair(q, t);
}

// Function to load keyframes from file
std::vector<KeyFrame*> LoadKeyFrames(const std::string& baseDir) {
    std::vector<KeyFrame*> vpKFs;
    
    // Load keyframe basic info
    std::string kfsFile = baseDir + "/keyframes.txt";
    std::ifstream f(kfsFile);
    
    if(!f.is_open()) {
        std::cerr << "Cannot open file: " << kfsFile << std::endl;
        return vpKFs;
    }
    
    std::string line;
    // Skip header
    std::getline(f, line);
    
    std::map<unsigned long, KeyFrame*> mapKFs;
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
    
    // Load keyframe poses
    std::string posesFile = baseDir + "/keyframe_poses.txt";
    std::ifstream fPoses(posesFile);
    
    if(!fPoses.is_open()) {
        std::cerr << "Cannot open file: " << posesFile << std::endl;
        return vpKFs;
    }
    
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
            float qNorm = q.norm();
            if(std::abs(qNorm - 1.0f) > 1e-6f) {
                std::cout << "Warning: Quaternion not normalized in poses file for KF " << id 
                          << ", norm = " << qNorm << std::endl;
                q.normalize();
            }
            
            Eigen::Matrix3f R = q.toRotationMatrix();
            Eigen::Vector3f t(tx, ty, tz);
            
            pKF->SetPose(R, t);
            poseCount++;
        }
    }
    
    fPoses.close();
    std::cout << "Loaded " << poseCount << " keyframe poses" << std::endl;
    
    // Load parent-child relationships
    std::string spanningFile = baseDir + "/spanning_tree.txt";
    std::ifstream fSpanning(spanningFile);
    
    if(fSpanning.is_open()) {
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
    std::string loopEdgesFile = baseDir + "/loop_edges.txt";
    std::ifstream fLoopEdges(loopEdgesFile);
    
    if(fLoopEdges.is_open()) {
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
    std::string covisFile = baseDir + "/covisibility.txt";
    std::ifstream fCovis(covisFile);
    
    if(fCovis.is_open()) {
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
    
    return vpKFs;
}

// Function to load map points from file
std::vector<MapPoint*> LoadMapPoints(const std::string& baseDir, const std::map<unsigned long, KeyFrame*>& mapKFs) {
    std::vector<MapPoint*> vpMPs;
    
    std::string mpsFile = baseDir + "/mappoints.txt";
    std::ifstream f(mpsFile);
    
    if(!f.is_open()) {
        std::cerr << "Cannot open file: " << mpsFile << std::endl;
        return vpMPs;
    }
    
    std::string line;
    // Skip header
    std::getline(f, line);
    int mpCount = 0;
    
    while(std::getline(f, line)) {
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
    
    f.close();
    std::cout << "Loaded " << mpCount << " map points" << std::endl;
    
    return vpMPs;
}

// Function to load SE3 data from file (from non_corrected_sim3.txt or corrected_sim3.txt)
KeyFrameAndPose LoadSE3Data(const std::string& filename, const std::map<unsigned long, KeyFrame*>& mapKFs) {
    KeyFrameAndPose result;
    
    std::ifstream f(filename);
    
    if(!f.is_open()) {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return result;
    }
    
    std::string line;
    // Skip header
    std::getline(f, line);
    int dataCount = 0;
    
    while(std::getline(f, line)) {
        std::istringstream iss(line);
        unsigned long id;
        double s, tx, ty, tz, qx, qy, qz, qw;
        
        iss >> id >> s >> tx >> ty >> tz >> qx >> qy >> qz >> qw;
        
        if(mapKFs.find(id) != mapKFs.end()) {
            KeyFrame* pKF = mapKFs.at(id);
            SE3Pose se3pose = ConvertToSE3(s, tx, ty, tz, qx, qy, qz, qw);
            result[pKF] = se3pose;
            dataCount++;
        }
    }
    
    f.close();
    std::cout << "Loaded " << dataCount << " SE3 poses from " << filename << std::endl;
    
    return result;
}

// Function to load loop connections from file
std::map<KeyFrame*, std::set<KeyFrame*>> LoadLoopConnections(const std::string& filename, 
                                                            const std::map<unsigned long, KeyFrame*>& mapKFs) {
    std::map<KeyFrame*, std::set<KeyFrame*>> result;
    
    std::ifstream f(filename);
    
    if(!f.is_open()) {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return result;
    }
    
    std::string line;
    // Skip header
    std::getline(f, line);
    int connCount = 0;
    
    while(std::getline(f, line)) {
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
        
        result[pKF] = sConnected;
    }
    
    f.close();
    std::cout << "Loaded " << connCount << " loop connections" << std::endl;
    
    return result;
}

// Function to validate all the input data
void ValidateInputData(const std::vector<KeyFrame*>& vpKFs, 
                      const std::vector<MapPoint*>& vpMPs, 
                      const KeyFrameAndPose& NonCorrectedSE3, 
                      const KeyFrameAndPose& CorrectedSE3, 
                      const std::map<KeyFrame*, std::set<KeyFrame*>>& LoopConnections) {
    std::cout << "\n=========== DETAILED DATA VALIDATION ===========" << std::endl;
    
    // 1. Validate KeyFrames
    std::cout << "\nValidating " << vpKFs.size() << " KeyFrames..." << std::endl;
    int invalidKF = 0;
    int smallRotationKF = 0;
    
    for(KeyFrame* pKF : vpKFs) {
        if(pKF->isBad()) continue;
        
        Eigen::Matrix3f Rcw;
        Eigen::Vector3f tcw;
        pKF->GetPose(Rcw, tcw);
        
        // Check if the rotation matrix is valid (orthogonal)
        Eigen::Matrix3f RRT = Rcw * Rcw.transpose();
        Eigen::Matrix3f I = Eigen::Matrix3f::Identity();
        float deviation = (RRT - I).norm();
        
        if(deviation > 1e-3) {
            std::cout << "WARNING: KeyFrame " << pKF->mnId << " has non-orthogonal rotation matrix. Deviation: " << deviation << std::endl;
            invalidKF++;
        }
        
        // Convert to quaternion and check normalization
        Eigen::Quaternionf q(Rcw);
        if(std::abs(q.norm() - 1.0f) > 1e-6) {
            std::cout << "WARNING: KeyFrame " << pKF->mnId << " has non-normalized quaternion. Norm: " << q.norm() << std::endl;
            invalidKF++;
        }
        
        // Check for very small rotation (can cause numerical issues)
        Eigen::AngleAxisf aa(q);
        if(aa.angle() < 1e-10) {
            std::cout << "NOTE: KeyFrame " << pKF->mnId << " has very small rotation angle: " << aa.angle() << std::endl;
            smallRotationKF++;
        }
        
        // Check for NaN values
        if(std::isnan(tcw.x()) || std::isnan(tcw.y()) || std::isnan(tcw.z()) ||
           std::isnan(Rcw(0,0)) || std::isnan(Rcw(0,1)) || std::isnan(Rcw(0,2)) ||
           std::isnan(Rcw(1,0)) || std::isnan(Rcw(1,1)) || std::isnan(Rcw(1,2)) ||
           std::isnan(Rcw(2,0)) || std::isnan(Rcw(2,1)) || std::isnan(Rcw(2,2))) {
            std::cout << "ERROR: KeyFrame " << pKF->mnId << " has NaN values in pose!" << std::endl;
            invalidKF++;
        }
    }
    
    std::cout << "KeyFrame validation summary: " << invalidKF << " invalid keyframes, " 
              << smallRotationKF << " keyframes with very small rotations" << std::endl;
    
    // 2. Validate MapPoints
    std::cout << "\nValidating " << vpMPs.size() << " MapPoints..." << std::endl;
    int invalidMP = 0;
    
    for(MapPoint* pMP : vpMPs) {
        if(pMP->isBad()) continue;
        
        Eigen::Vector3f pos = pMP->GetWorldPos();
        
        // Check for NaN values
        if(std::isnan(pos.x()) || std::isnan(pos.y()) || std::isnan(pos.z())) {
            std::cout << "ERROR: MapPoint " << pMP->mnId << " has NaN position: " 
                      << pos.x() << ", " << pos.y() << ", " << pos.z() << std::endl;
            invalidMP++;
            continue;
        }
        
        // Check for extremely large values
        if(pos.norm() > 1e6) {
            std::cout << "WARNING: MapPoint " << pMP->mnId << " has very large position norm: " << pos.norm() << std::endl;
            invalidMP++;
        }
    }
    
    std::cout << "MapPoint validation summary: " << invalidMP << " invalid map points" << std::endl;
    
    // 3. Validate NonCorrectedSE3
    std::cout << "\nValidating " << NonCorrectedSE3.size() << " NonCorrectedSE3 poses..." << std::endl;
    int invalidNonCorrected = 0;
    
    for(const auto& entry : NonCorrectedSE3) {
        KeyFrame* pKF = entry.first;
        const SE3Pose& se3pose = entry.second;
        Eigen::Quaterniond q = se3pose.first;
        Eigen::Vector3d t = se3pose.second;
        
        // Check for NaN values
        if(std::isnan(q.x()) || std::isnan(q.y()) || std::isnan(q.z()) || std::isnan(q.w()) ||
           std::isnan(t.x()) || std::isnan(t.y()) || std::isnan(t.z())) {
            std::cout << "ERROR: NonCorrectedSE3 for KF " << pKF->mnId << " has NaN values!" << std::endl;
            std::cout << "q: " << q.coeffs().transpose() << ", t: " << t.transpose() << std::endl;
            invalidNonCorrected++;
            continue;
        }
        
        // Check quaternion normalization
        double qNorm = q.norm();
        if(std::abs(qNorm - 1.0) > 1e-6) {
            std::cout << "WARNING: NonCorrectedSE3 quaternion not normalized for KF " 
                      << pKF->mnId << ": norm = " << qNorm << std::endl;
            invalidNonCorrected++;
        }
        
        // Check for small rotations
        Eigen::AngleAxisd aa(q);
        double angle = aa.angle();
        if(angle < 1e-10) {
            std::cout << "NOTE: NonCorrectedSE3 has very small rotation angle for KF " 
                      << pKF->mnId << ": " << angle << std::endl;
        }
    }
    
    std::cout << "NonCorrectedSE3 validation summary: " << invalidNonCorrected << " invalid poses" << std::endl;
    
    // 4. Validate CorrectedSE3
    std::cout << "\nValidating " << CorrectedSE3.size() << " CorrectedSE3 poses..." << std::endl;
    int invalidCorrected = 0;
    int smallRotationCorrected = 0;
    
    for(const auto& entry : CorrectedSE3) {
        KeyFrame* pKF = entry.first;
        const SE3Pose& se3pose = entry.second;
        Eigen::Quaterniond q = se3pose.first;
        Eigen::Vector3d t = se3pose.second;
        
        // Check for NaN values
        if(std::isnan(q.x()) || std::isnan(q.y()) || std::isnan(q.z()) || std::isnan(q.w()) ||
           std::isnan(t.x()) || std::isnan(t.y()) || std::isnan(t.z())) {
            std::cout << "ERROR: CorrectedSE3 for KF " << pKF->mnId << " has NaN values!" << std::endl;
            std::cout << "q: " << q.coeffs().transpose() << ", t: " << t.transpose() << std::endl;
            invalidCorrected++;
            continue;
        }
        
        // Check quaternion normalization
        double qNorm = q.norm();
        if(std::abs(qNorm - 1.0) > 1e-6) {
            std::cout << "WARNING: CorrectedSE3 quaternion not normalized for KF " 
                      << pKF->mnId << ": norm = " << qNorm << std::endl;
            invalidCorrected++;
        }
        
        // Check for very small rotations
        Eigen::AngleAxisd aa(q);
        double angle = aa.angle();
        if(angle < 1e-10) {
            std::cout << "NOTE: CorrectedSE3 has very small rotation angle for KF " 
                      << pKF->mnId << ": " << angle << std::endl;
            smallRotationCorrected++;
        }
        
        // Convert to angle-axis to check for potential issues in optimization
        Eigen::AngleAxisd angleAxis(q);
        if(angleAxis.angle() < 1e-10) {
            // When rotation angle is very small, the axis can be unstable
            std::cout << "WARNING: CorrectedSE3 for KF " << pKF->mnId << " has very small rotation angle: " 
                      << angleAxis.angle() << " (axis = " << angleAxis.axis().transpose() << ")" << std::endl;
        }
    }
    
    std::cout << "CorrectedSE3 validation summary: " << invalidCorrected << " invalid poses, " 
              << smallRotationCorrected << " poses with very small rotations" << std::endl;
    
    // 5. Validate LoopConnections
    std::cout << "\nValidating loop connections..." << std::endl;
    int totalConnections = 0;
    
    for(const auto& entry : LoopConnections) {
        KeyFrame* pKF = entry.first;
        const std::set<KeyFrame*>& sConnected = entry.second;
        
        totalConnections += sConnected.size();
        
        // Check if the loop connections make sense
        for(KeyFrame* pConnKF : sConnected) {
            if(!pConnKF || pConnKF->isBad()) {
                std::cout << "WARNING: Loop connection from KF " << pKF->mnId << " connects to invalid KF" << std::endl;
            }
            
            // Check if connection exists in both directions
            auto it = LoopConnections.find(pConnKF);
            if(it != LoopConnections.end()) {
                if(it->second.find(pKF) == it->second.end()) {
                    std::cout << "WARNING: Loop connection from KF " << pKF->mnId << " to KF " 
                              << pConnKF->mnId << " is not bidirectional" << std::endl;
                }
            }
        }
    }
    
    std::cout << "Loop connection validation: " << totalConnections << " total connections" << std::endl;
    
    // 6. Additional Cross-Validation
    std::cout << "\nPerforming cross-validation..." << std::endl;
    
    // Check if all keyframes in the SE3 data are valid
    for(const auto& entry : CorrectedSE3) {
        KeyFrame* pKF = entry.first;
        if(pKF->isBad()) {
            std::cout << "WARNING: CorrectedSE3 contains a bad keyframe with ID " << pKF->mnId << std::endl;
        }
    }
    
    for(const auto& entry : NonCorrectedSE3) {
        KeyFrame* pKF = entry.first;
        if(pKF->isBad()) {
            std::cout << "WARNING: NonCorrectedSE3 contains a bad keyframe with ID " << pKF->mnId << std::endl;
        }
    }
    
    // Check correct references for map points
    for(MapPoint* pMP : vpMPs) {
        if(pMP->isBad()) continue;
        
        KeyFrame* pRefKF = pMP->GetReferenceKeyFrame();
        if(!pRefKF || pRefKF->isBad()) {
            std::cout << "WARNING: MapPoint " << pMP->mnId << " has invalid reference KeyFrame" << std::endl;
        }
        
        // If corrected, check that references exist
        if(pMP->mnCorrectedByKF != 0) {
            bool foundKF = false;
            for(KeyFrame* pKF : vpKFs) {
                if(pKF->mnId == pMP->mnCorrectedByKF) {
                    foundKF = true;
                    break;
                }
            }
            
            if(!foundKF) {
                std::cout << "WARNING: MapPoint " << pMP->mnId << " was corrected by non-existent KF " 
                          << pMP->mnCorrectedByKF << std::endl;
            }
        }
    }
    
    std::cout << "\n=========== DATA VALIDATION COMPLETE ===========" << std::endl;
}

// Main function that mimics OptimizeEssentialGraph
void ProcessDataLikeOptimizeEssentialGraph(const std::string& inputDir) {
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
    std::vector<KeyFrame*> vpKFs = LoadKeyFrames(inputDir);
    
    // Create map of KeyFrame ID to KeyFrame
    std::map<unsigned long, KeyFrame*> mapKFs;
    for(KeyFrame* pKF : vpKFs) {
        mapKFs[pKF->mnId] = pKF;
    }
    
    // Load map points
    std::vector<MapPoint*> vpMPs = LoadMapPoints(inputDir, mapKFs);
    
    // Load SE3 data from Sim3 files
    KeyFrameAndPose NonCorrectedSE3 = LoadSE3Data(inputDir + "/non_corrected_sim3.txt", mapKFs);
    KeyFrameAndPose CorrectedSE3 = LoadSE3Data(inputDir + "/corrected_sim3.txt", mapKFs);
    
    // Load loop connections
    std::map<KeyFrame*, std::set<KeyFrame*>> LoopConnections = 
        LoadLoopConnections(inputDir + "/loop_connections.txt", mapKFs);
    
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
        return;
    }
    
    // Create Map (as in the original code)
    Map* pMap = new Map();
    pMap->mnId = 0;
    pMap->mnInitKFid = nInitKFid;
    pMap->mnMaxKFid = nMaxKFid;
    pMap->SetKeyFrames(vpKFs);
    pMap->SetMapPoints(vpMPs);
    
    // Validate all the data
    ValidateInputData(vpKFs, vpMPs, NonCorrectedSE3, CorrectedSE3, LoopConnections);
    
    // Examine the specific relationship between pLoopKF and pCurKF
    std::cout << "\n====== LOOP DETAILS ======" << std::endl;
    std::cout << "Loop KF ID: " << pLoopKF->mnId << std::endl;
    std::cout << "Current KF ID: " << pCurKF->mnId << std::endl;
    
    // Check if there's a direct connection in LoopConnections
    bool directLoopConnection = false;
    auto it = LoopConnections.find(pCurKF);
    if(it != LoopConnections.end()) {
        if(it->second.find(pLoopKF) != it->second.end()) {
            directLoopConnection = true;
        }
    }
    
    std::cout << "Direct loop connection between pCurKF and pLoopKF: " << (directLoopConnection ? "YES" : "NO") << std::endl;
    
    // Check if both keyframes have entries in the Sim3 maps
    bool inNonCorrected = NonCorrectedSE3.find(pLoopKF) != NonCorrectedSE3.end() && 
                          NonCorrectedSE3.find(pCurKF) != NonCorrectedSE3.end();
    bool inCorrected = CorrectedSE3.find(pLoopKF) != CorrectedSE3.end() && 
                       CorrectedSE3.find(pCurKF) != CorrectedSE3.end();
    
    std::cout << "Both keyframes in NonCorrectedSE3: " << (inNonCorrected ? "YES" : "NO") << std::endl;
    std::cout << "Both keyframes in CorrectedSE3: " << (inCorrected ? "YES" : "NO") << std::endl;
    
    // Print some additional information about these important keyframes
    std::cout << "\nLoop KF details:" << std::endl;
    Eigen::Matrix3f RLoopKF;
    Eigen::Vector3f tLoopKF;
    pLoopKF->GetPose(RLoopKF, tLoopKF);
    std::cout << "Position: " << tLoopKF.transpose() << std::endl;
    
    std::cout << "\nCurrent KF details:" << std::endl;
    Eigen::Matrix3f RCurKF;
    Eigen::Vector3f tCurKF;
    pCurKF->GetPose(RCurKF, tCurKF);
    std::cout << "Position: " << tCurKF.transpose() << std::endl;
    
    // Clean up
    for(KeyFrame* pKF : vpKFs)
        delete pKF;
    
    for(MapPoint* pMP : vpMPs)
        delete pMP;
    
    delete pMap;
    
    std::cout << "\nData analysis complete." << std::endl;
}

int main(int argc, char** argv) {
    std::string inputDir = "/Datasets/CERES_Work/input/optimization_data";
    
    if(argc > 1) {
        inputDir = argv[1];
    }
    
    std::cout << "Analyzing data in: " << inputDir << std::endl;
    ProcessDataLikeOptimizeEssentialGraph(inputDir);
    
    return 0;
}
