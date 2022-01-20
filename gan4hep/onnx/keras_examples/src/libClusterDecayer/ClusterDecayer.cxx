#include <ClusterDecayer.hpp>

#include <iostream>
#include <math.h>
#include <random>
#include <assert.h>
#include <algorithm>
#include <iterator>

HerwigClusterDecayer::HerwigClusterDecayer(const Config& config): m_cfg(config){
    std::cout << "Constructing HerwigClusterDcayer" << std::endl;
    assert(m_cfg.clusterMin.size() == m_cfg.numInputFeatures);
    assert(m_cfg.clusterMax.size() == m_cfg.numInputFeatures);
    initTrainedModels();
}

void HerwigClusterDecayer::getDecayProducts(
    std::vector<float>& cluster4Vec,
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

    // figure out noise dimensions
    Ort::TypeInfo inputTypeInfo = m_sess->GetInputTypeInfo(0);
    auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> inputDims = inputTensorInfo.GetShape();
    int64_t totalFeatures = inputDims[1];
    int64_t numEvts = 1;
    int noiseDims = totalFeatures - m_cfg.numInputFeatures;

    // generate noises from a norm distribution
    std::random_device rd;
    std::mt19937 gen{rd()};
    std::normal_distribution<> normal_dis{0, 1};

    std::vector<float> inputTensorValues = std::move(scaledCluster4Vec);
    for (int idx = 0; idx < noiseDims; idx++) {
        inputTensorValues.push_back(normal_dis(gen));
    }

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
    Ort::TypeInfo outputTypeInfo = m_sess->GetOutputTypeInfo(0);
    auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> outputDims = outputTensorInfo.GetShape();

    int64_t numOfOutFeatures = outputDims[1]; // phi, theta, energy
    assert(numOfOutFeatures == m_cfg.hadronMax.size());
    assert(numOfOutFeatures == m_cfg.hadronMin.size());

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
    std::cout << "original output: ";
    std::copy(outputData.begin(), outputData.end(), std::ostream_iterator<float>(std::cout, " "));
    std::cout << std::endl;

    // convert the three output vectors
    const float pionMass = 0.135;
    for(unsigned int idx=0; idx < numOfOutFeatures; idx++){
        scalerInv(outputData[idx], m_cfg.hadronMin[idx], m_cfg.hadronMax[idx]);
    }
    std::cout << "after scaling: ";
    std::copy(outputData.begin(), outputData.end(), std::ostream_iterator<float>(std::cout, " "));
    std::cout << std::endl;

    float phi = outputData[0];
    float theta = outputData[1];
    float energy = outputData[2];

    float momentum = sqrt(energy*energy - pionMass*pionMass);
    float px = momentum * sin(theta) * sin(phi);
    float py = momentum * sin(theta) * cos(phi);
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
