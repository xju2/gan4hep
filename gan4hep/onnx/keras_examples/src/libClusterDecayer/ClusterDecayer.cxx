#include <ClusterDecayer.hpp>

#include <iostream>
#include <math.h>
#include <random>
#include <assert.h>

HerwigClusterDecayer::HerwigClusterDecayer(const Config& config): m_cfg(config){
    std::cout << "Constructing HerwigClusterDcayer" << std::endl;
    initTrainedModels();
}

void HerwigClusterDecayer::getDecayProducts(
    std::vector<float>& cluster4Vec,
    std::vector<float>& hadronOne4Vec,
    std::vector<float>& hadronTwo4Vec)
{
    // does it live here?
    Ort::AllocatorWithDefaultOptions allocator;
    auto memoryInfo = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator,
        OrtMemType::OrtMemTypeDefault
    );

    // prepare inputs
    std::random_device rd;
    std::mt19937 gen{rd()};
    std::normal_distribution<> normal_dis{0, 1};

    std::vector<float> inputTensorValues = cluster4Vec;
    for (int idx = 0; idx < m_cfg.noiseDims; idx++) {
        inputTensorValues.push_back(normal_dis(gen));
    }

    int64_t numEvts = 1;
    int64_t totalFeatures = 4 + m_cfg.noiseDims;
    std::vector<int64_t> inputShape{numEvts, totalFeatures};
    const char* inputName = m_sess->GetInputName(0, allocator);
    std::vector<const char*> inputNames{inputName};
    std::vector<Ort::Value> inputTensor;
    inputTensor.push_back(
        Ort::Value::CreateTensor<float>(
            memoryInfo, inputTensorValues.data(), inputTensorValues.size(),
            inputShape.data(), inputShape.size())
    );

    // prepare outputs
    int64_t numOfOutFeatures = 3; // phi, theta, energy
    std::vector<float> outputData(numEvts * numOfOutFeatures);
    std::vector<int64_t> outputShape{numEvts, numOfOutFeatures};
    std::vector<Ort::Value> outputTensor;
    outputTensor.push_back(
        Ort::Value::CreateTensor<float>(
            memoryInfo, outputData.data(), outputData.size(),
            outputShape.data(), outputShape.size())
    );
    const char* outputName = m_sess->GetOutputName(0, allocator);
    std::vector<const char*> outputNames{outputName};

    runSessionWithIoBinding(*m_sess, inputNames, inputTensor, outputNames, outputTensor);
    std::cout << "output data size: " << outputData.size() << " " << outputData[0] << std::endl;

    // convert the three output vectors
    const float pionMass = 0.135;
    float phi = outputData[0];
    float theta = outputData[1];
    float energy = outputData[2];

    float momentum = sqrt(energy*energy - pionMass*pionMass);
    float px = momentum * sin(theta) * cos(phi);
    float py = momentum * sin(theta) * sin(phi);
    float pz = momentum * cos(theta);
    hadronOne4Vec.clear();
    hadronOne4Vec.push_back(energy);
    hadronOne4Vec.push_back(px);
    hadronOne4Vec.push_back(py);
    hadronOne4Vec.push_back(pz);

    // now use 4 vector conservation to calculate the other hadron
    hadronTwo4Vec.clear();
    hadronTwo4Vec.push_back(cluster4Vec[0] - energy);
    hadronTwo4Vec.push_back(cluster4Vec[1] - px);
    hadronTwo4Vec.push_back(cluster4Vec[2] - py);
    hadronTwo4Vec.push_back(cluster4Vec[3] - pz);
}


void HerwigClusterDecayer::initTrainedModels()
{

    std::cout << "Initializing Trained ML Models" << std::endl;
    std::cout << "Model input directory:  " << m_cfg.inputMLModelDir << std::endl;
    std::cout << "Input Noise Dimensions: " << m_cfg.noiseDims << std::endl;

    m_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "GAN4Herwig");

    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    m_sess = std::make_unique<Ort::Session>(*m_env, m_cfg.inputMLModelDir.c_str(), session_options);
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
