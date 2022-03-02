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

    // these for scaling back the output [phi, eta]
    config.clusterMin = std::move(std::vector<float>{0.652201, -35.2036, -29.6485, -35.7964});
    config.clusterMax = std::move(std::vector<float>{38.8865, 31.0066, 31.8602, 33.0058});
    config.hadronMin = std::move(std::vector<float>{-1.5707537, -1.570766});
    config.hadronMax = std::move(std::vector<float>{1.5707537, 1.570766});
    config.useCuda = useCUDA;
    config.massDecayer1 = 0.134978;
    config.massDecayer2 = 0.134978;

    HerwigClusterDecayer clusterDecayer{config};

    std::vector<float> cluster4vec{3.84263,-3.27865,1.36946,0.841847};
    std::vector<float> hadron4vec1; // from Herwig: 1.56275,-1.5339,0.248784,-0.0960216
    std::vector<float> hadron4vec2; // from Herwig: 2.27988,-1.74475,1.12067,0.937869

    clusterDecayer.getDecayProducts(cluster4vec, hadron4vec1, hadron4vec2);

    auto print_checks = [](const std::vector<float>& cluster,
        const std::vector<float>& h1, const std::vector<float>& h1_herwig,
        const std::vector<float>& h2, const std::vector<float>& h2_herwig){
        std::cout << "incoming cluster with 4 vector: ";
        std::copy(cluster.begin(), cluster.end(),
            std::ostream_iterator<float>(std::cout, " "));
        std::cout << std::endl;
        std::cout <<" produced two hadrons [pions] with 4 vector:\n\n[GAN]    ";
        std::copy(h1.begin(), h1.end(),
            std::ostream_iterator<float>(std::cout, " "));
        std::cout << "\n[Herwig] ";
        std::copy(h1_herwig.begin(), h1_herwig.end(),
            std::ostream_iterator<float>(std::cout, " "));

        std::cout << "\n\n[GAN]    ";
        std::copy(h2.begin(), h2.end(),
            std::ostream_iterator<float>(std::cout, " "));
        std::cout << "\n[Herwig] ";
        std::copy(h2_herwig.begin(), h2_herwig.end(),
            std::ostream_iterator<float>(std::cout, " "));

        std::cout << std::endl;
    };
    std::vector<float> h1Herwig{1.56275,-1.5339,0.248784,-0.0960216};
    std::vector<float> h2Herwig{2.27988,-1.74475,1.12067,0.937869};
    print_checks(cluster4vec, hadron4vec1, h1Herwig, hadron4vec2, h2Herwig);


    // Config the decayer for cases in which there are at least one quark with Pert=1
    HerwigClusterDecayer::Config config_pert;
    config_pert.inputMLModelDir = "../../data/models/cluster_decayer_pert.onnx";

    // these for scaling back the output [phi, eta]
    config_pert.clusterMin = std::move(std::vector<float>{0.692856, -44.578400, -44.212502, -44.866402});
    config_pert.clusterMax = std::move(std::vector<float>{45.605202, 44.833000, 45.033901, 44.893398});
    config_pert.hadronMin = std::move(std::vector<float>{-1.570768, -1.570728});
    config_pert.hadronMax = std::move(std::vector<float>{1.570796, 1.570796});
    config_pert.useCuda = useCUDA;
    config_pert.massDecayer1 = 0.134978;
    config_pert.massDecayer2 = 0.134978;

    HerwigClusterDecayer clusterDecayerPert{config_pert};

    std::vector<float> cluster4vecPert{25.5166,17.3116,0.866833,18.6806};
    std::vector<float> hadron4vec1Pert; // from Herwig: 19.0023,12.8332,1.19091,13.9629
    std::vector<float> hadron4vec2Pert; // from Herwig: 6.5143,4.4784,-0.324079,4.71771
    clusterDecayer.getDecayProducts(cluster4vecPert, hadron4vec1Pert, hadron4vec2Pert);

    std::vector<float> h1PertHerwig{19.0023,12.8332,1.19091,13.9629};
    std::vector<float> h2PertHerwig{6.5143,4.4784,-0.324079,4.71771};
    print_checks(cluster4vecPert, hadron4vec1Pert, h1PertHerwig,hadron4vec2Pert, h2PertHerwig);

    return 0;
}