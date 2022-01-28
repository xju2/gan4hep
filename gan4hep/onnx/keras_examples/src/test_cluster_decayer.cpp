#include "ClusterDecayer.hpp"

#include <vector>
#include <iostream>
#include <iterator>
#include <unistd.h> // to parse input args.

int main(int argc, char* argv[])
{
    std::string modelFilepath{"../../data/models/cluster_decayer.onnx"};
    int opt;
    bool useCUDA = false;
    bool help = false;

    while ((opt = getopt(argc, argv, "hcm:")) != -1){
        switch(opt){
            case 'm':
                modelFilepath = optarg;
                break;
            case 'c':
                useCUDA = true;
                break;
            default:
                fprintf(stderr, "Usage %s [-c]\n", argv[0]);
            if (help){
                printf("   -c useCUDA: use cuda\n");
                printf("   -m MODELFILE: onnx model file name\n");
            }
            exit(EXIT_FAILURE);
        }
    }

    HerwigClusterDecayer::Config config;
    config.inputMLModelDir = std::move(modelFilepath);

    // these for scaling back the output [phi, eta, energy]
    config.clusterMin = std::move(std::vector<float>{0.652201, -35.2036, -29.6485, -35.7964});
    config.clusterMax = std::move(std::vector<float>{38.8865, 31.0066, 31.8602, 33.0058});
    config.hadronMin = std::move(std::vector<float>{-1.5707537, -1.570766, 0.135401 });
    config.hadronMax = std::move(std::vector<float>{1.5707537, 1.570766, 30.9537});
    config.useCuda = useCUDA;

    HerwigClusterDecayer clusterDecayer{config};

    std::vector<float> cluster4vec{3.84263,-3.27865,1.36946,0.841847};
    std::vector<float> hadron4vec1; // from Herwig: 1.56275,-1.5339,0.248784,-0.0960216
    std::vector<float> hadron4vec2; // from Herwig: 2.27988,-1.74475,1.12067,0.937869

    clusterDecayer.getDecayProducts(cluster4vec, hadron4vec1, hadron4vec2);

    std::cout << "incoming cluster with 4 vector: ";
    std::copy(cluster4vec.begin(), cluster4vec.end(),
        std::ostream_iterator<float>(std::cout, " "));
    std::cout << std::endl;
    std::cout <<" produced two hadrons [pions] with 4 vector:\n\n[GAN]    ";
    std::copy(hadron4vec1.begin(), hadron4vec1.end(),
        std::ostream_iterator<float>(std::cout, " "));
    std::cout << std::endl << "[Herwig] 1.56275,-1.5339,0.248784,-0.0960216\n\n[GAN]    ";

    std::copy(hadron4vec2.begin(), hadron4vec2.end(),
        std::ostream_iterator<float>(std::cout, " "));
    std::cout << std::endl << "[Herwig] 2.27988,-1.74475,1.12067,0.937869\n";

    return 0;
}