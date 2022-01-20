#pragma once
#include <string>
#include <vector>
#include <memory>

#include <core/session/onnxruntime_cxx_api.h>

class HerwigClusterDecayer {
public:
    struct Config
    {
        std::string inputMLModelDir;
        std::vector<float> clusterMin;
        std::vector<float> clusterMax;
        std::vector<float> hadronMin;
        std::vector<float> hadronMax;
        int numInputFeatures = 4;
    };
    
    HerwigClusterDecayer(const Config& config);
    virtual ~HerwigClusterDecayer() {}

    // It takes the cluster 4 vector as input
    // and produces two hadrons' 4vectors as output.
    // 4 vectors are in the order of [energy, px, py, pz]
    // hadrons are assumed to be pions of mass 0.135 GeV
    void getDecayProducts(
        std::vector<float>& cluster4Vec,
        std::vector<float>& hadronOne4Vec,
        std::vector<float>& hadronTwo4Vec
    );

private:
    void initTrainedModels();
    void runSessionWithIoBinding(
      Ort::Session& sess,
      std::vector<const char*>& inputNames,
      std::vector<Ort::Value> & inputData,
      std::vector<const char*>& outputNames,
      std::vector<Ort::Value>&  outputData) const;

private:
    Config m_cfg;
    std::unique_ptr<Ort::Env> m_env;
    std::unique_ptr<Ort::Session> m_sess;
};