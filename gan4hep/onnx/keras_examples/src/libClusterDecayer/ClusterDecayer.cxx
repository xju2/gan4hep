#include <ClusterDecayer.hpp>

#include <iostream>
#include <math.h>
#include <random>
#include <assert.h>
#include <algorithm>
#include <iterator>

HerwigClusterDecayer::HerwigClusterDecayer(const Config& config): m_cfg(config){
    std::cout << "Constructing HerwigClusterDcayer" << std::endl;

    // by default, the number of input features are the cluster 4 vectors
    assert(m_cfg.clusterMin.size() == m_cfg.numInputFeatures);
    assert(m_cfg.clusterMax.size() == m_cfg.numInputFeatures);

    initTrainedModels();

    m_numEvts = 1; // batch size
}

void HerwigClusterDecayer::getDecayProducts(
    std::vector<float>& cluster4Vec,
    float hadronMassOne, float hadronMassTwo,
    std::vector<float>& hadronOne4Vec,
    std::vector<float>& hadronTwo4Vec)
{
    assert(cluster4Vec.size() == m_cfg.numInputFeatures);
    // does it live here?
    Ort::AllocatorWithDefaultOptions allocator;
    auto memoryInfo = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator,
        OrtMemType::OrtMemTypeDefault
    );

    // lambda functions to scale inputs and scale back outputs
    // they may not live here
    float scaledMin = -1, scaledMax = 1;
    auto scalerFrd = [=](float x, float xMin, float xMax){
        // assuming the data is scaled to [-1, 1]
        return (x - xMin) / (xMax - xMin) * (scaledMax - scaledMin) + scaledMin;
    };
    auto scalerInv = [=](float& xScaled, float xMin, float xMax){
        xScaled = (xScaled - scaledMin) / (scaledMax - scaledMin) * (xMax - xMin) + xMin;
    };
    std::vector<float> scaledCluster4Vec;
    for(unsigned int idx=0; idx < cluster4Vec.size(); idx++){
        scaledCluster4Vec.push_back(
            scalerFrd(cluster4Vec[idx], m_cfg.clusterMin[idx], m_cfg.clusterMax[idx])
        );
    }

    // generate noises from a norm distribution
    std::random_device rd;
    std::mt19937 gen{rd()};
    std::normal_distribution<> normal_dis{0, 1};

    std::vector<float> inputTensorValues = std::move(scaledCluster4Vec);
    for (int idx = 0; idx < m_noiseDims; idx++) {
        inputTensorValues.push_back(normal_dis(gen));
    }

    std::vector<int64_t> inputShape{m_numEvts, m_totalInputFeatures};
    const char* inputName = m_sess->GetInputName(0, allocator);
    std::vector<const char*> inputNames{inputName};
    std::vector<Ort::Value> inputTensor;
    inputTensor.push_back(
        Ort::Value::CreateTensor<float>(
            memoryInfo, inputTensorValues.data(), inputTensorValues.size(),
            inputShape.data(), inputShape.size())
    );


    // prepare outputs
    std::vector<float> outputData(m_numEvts * m_numOfOutFeatures);
    std::vector<int64_t> outputShape{m_numEvts, m_numOfOutFeatures};
    std::vector<Ort::Value> outputTensor;
    outputTensor.push_back(
        Ort::Value::CreateTensor<float>(
            memoryInfo, outputData.data(), outputData.size(),
            outputShape.data(), outputShape.size())
    );
    const char* outputName = m_sess->GetOutputName(0, allocator);
    std::vector<const char*> outputNames{outputName};

    // runSessionWithIoBinding(*m_sess, inputNames, inputTensor, outputNames, outputTensor);

    m_sess->Run(Ort::RunOptions{nullptr}, inputNames.data(),
        inputTensor.data(), 1, outputNames.data(), outputTensor.data(), 1);

    std::cout << "original outputs from GAN: ";
    std::copy(outputData.begin(), outputData.end(), std::ostream_iterator<float>(std::cout, " "));
    std::cout << std::endl;

    // convert the three output vectors
    for(unsigned int idx=0; idx < m_numOfOutFeatures; idx++){
        scalerInv(outputData[idx], m_cfg.hadronMin[idx], m_cfg.hadronMax[idx]);
    }
    // std::cout << "after scaling: ";
    // std::copy(outputData.begin(), outputData.end(), std::ostream_iterator<float>(std::cout, " "));
    // std::cout << std::endl;

    float phi = outputData[0];
    float theta = outputData[1];

    // inverse boost, i.e. boost them back to the lab frame.
    // first calculate direction of the cluster and lorentz factor
    float p2 = cluster4Vec[1]*cluster4Vec[1] \
        + cluster4Vec[2]*cluster4Vec[2] \
        + cluster4Vec[3]*cluster4Vec[3];
    float mass = sqrt(cluster4Vec[0]*cluster4Vec[0] - p2);
    float gamma = cluster4Vec[0] / mass;
    float v_mag = sqrt(p2) / gamma / mass;
    std::vector<float> n{
        cluster4Vec[1]/gamma/mass/v_mag, 
        cluster4Vec[2]/gamma/mass/v_mag,
        cluster4Vec[3]/gamma/mass/v_mag,
        };

    // Calculate the 4vector of the two outgoing pions in the cluster's frame, in which
    // they are back-to-back.
    float energy = (mass*mass + hadronMassOne*hadronMassOne - hadronMassTwo*hadronMassTwo) / (2*mass);
    float momentum = sqrt(energy*energy - hadronMassOne*hadronMassOne);
    float px = momentum * sin(theta) * sin(phi);
    float py = momentum * sin(theta) * cos(phi);
    float pz = momentum * cos(theta);

    auto invBoost = [&](float& energy_, float& px_, float& py_, float& pz_){
        float n_dot_p = n[0]*px_ + n[1]*py_ + n[2]*pz_;
        float new_energy = gamma * (energy_ + v_mag * n_dot_p);
        px_ = px_ + (gamma - 1) * n_dot_p * n[0] + gamma * energy_ * v_mag * n[0];
        py_ = py_ + (gamma - 1) * n_dot_p * n[1] + gamma * energy_ * v_mag * n[1];
        pz_ = pz_ + (gamma - 1) * n_dot_p * n[2] + gamma * energy_ * v_mag * n[2];
        energy_ = new_energy;
    };

    
    float h1_e=energy, h1_px=px, h1_py=py, h1_pz=pz;
    invBoost(h1_e, h1_px, h1_py, h1_pz);
    hadronOne4Vec.clear();
    hadronOne4Vec.push_back(h1_e);
    hadronOne4Vec.push_back(h1_px);
    hadronOne4Vec.push_back(h1_py);
    hadronOne4Vec.push_back(h1_pz);

    // now use 4 vector conservation to calculate the other hadron
    float h2_e = mass-energy, h2_px=-px, h2_py=-py, h2_pz=-pz;
    invBoost(h2_e, h2_px, h2_py, h2_pz);
    hadronTwo4Vec.clear();
    hadronTwo4Vec.push_back(h2_e);
    hadronTwo4Vec.push_back(h2_px);
    hadronTwo4Vec.push_back(h2_py);
    hadronTwo4Vec.push_back(h2_pz);
}


void HerwigClusterDecayer::initTrainedModels()
{

    std::cout << "Initializing Trained ML Models" << std::endl;
    std::cout << "Model input directory:  " << m_cfg.inputMLModelDir << std::endl;

    m_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "GAN4Herwig");

    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    if (m_cfg.useCuda) {
        OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0);
    }
    
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    m_sess = std::make_unique<Ort::Session>(*m_env, m_cfg.inputMLModelDir.c_str(), session_options);

    // extract some dimensionality information
    // inputs
    Ort::TypeInfo inputTypeInfo = m_sess->GetInputTypeInfo(0);
    auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> inputDims = inputTensorInfo.GetShape();
    m_totalInputFeatures = inputDims[1];
    m_noiseDims = m_totalInputFeatures - m_cfg.numInputFeatures;

    // outputs
    Ort::TypeInfo outputTypeInfo = m_sess->GetOutputTypeInfo(0);
    auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> outputDims = outputTensorInfo.GetShape();

    m_numOfOutFeatures = outputDims[1]; // phi, theta, energy
}

void HerwigClusterDecayer::runSessionWithIoBinding(
    Ort::Session& sess,
    std::vector<const char*>& inputNames,
    std::vector<Ort::Value> & inputData,
    std::vector<const char*>& outputNames,
    std::vector<Ort::Value>&  outputData) const
{
    // std::cout <<"In the runSessionWithIoBinding" << std::endl;
    if (inputNames.size() < 1) {
        throw std::runtime_error("Onnxruntime input data maping cannot be empty");
    }
    assert(inputNames.size() == inputData.size());

    Ort::IoBinding iobinding(sess);
    for(size_t idx = 0; idx < inputNames.size(); ++idx){
        iobinding.BindInput(inputNames[idx], inputData[idx]);
    }


    for(size_t idx = 0; idx < outputNames.size(); ++idx){
        iobinding.BindOutput(outputNames[idx], outputData[idx]);
    }

    sess.Run(Ort::RunOptions{nullptr}, iobinding);
}
