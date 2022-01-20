#include "ClusterDecayer.hpp"

#include <vector>
#include <iostream>
#include <iterator>

int main(int argc, char* argv[])
{
    HerwigClusterDecayer::Config config;
    config.inputMLModelDir = "../../data/models/mlp_gan.onnx";
    config.noiseDims = 4;

    HerwigClusterDecayer clusterDecayer{config};

    std::vector<float> cluster4vec{1, 0., 0., 0};
    std::vector<float> hadron4vec1;
    std::vector<float> hadron4vec2;

    clusterDecayer.getDecayProducts(cluster4vec, hadron4vec1, hadron4vec2);

    std::cout << "incoming cluster with 4 vector: ";
    std::copy(cluster4vec.begin(), cluster4vec.end(),
        std::ostream_iterator<float>(std::cout, " "));
    std::cout << std::endl;
    std::cout <<" produced two hadrons [pions] with 4 vector: ";
    std::copy(hadron4vec1.begin(), hadron4vec1.end(),
        std::ostream_iterator<float>(std::cout, " "));
    std::cout << std::endl;
    std::copy(hadron4vec2.begin(), hadron4vec2.end(),
        std::ostream_iterator<float>(std::cout, " "));
    std::cout << std::endl;

    return 0;
}