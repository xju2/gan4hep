# Compile the code in a docker container
As a starting point, one can play with the code with the docker container by running

```bash
docker run -it --rm -v $PWD:$PWD -w $PWD --gpus all docexoty/mltools:ubuntu20.04
```

then `mkdir build && cd build && cmake .. && make`. 
It will compile the code and produce an executable `test_cluster_decayer`;
Run `./bin/test_cluster_decayer`

A typical output can be
```
Constructing HerwigClusterDcayer
Initializing Trained ML Models
Model input directory:  ../../data/models/cluster_decayer.onnx
original outputs from GAN: -0.125014 0.383769 
incoming cluster with 4 vector: 3.84263 -3.27865 1.36946 0.841847 
 produced two hadrons [pions] with 4 vector:

[GAN]    2.80672 -2.27986 1.24949 1.04902 
[Herwig] 1.56275 -1.5339 0.248784 -0.0960216 

[GAN]    1.03591 -0.998792 0.11997 -0.207177 
[Herwig] 2.27988 -1.74475 1.12067 0.937869 
Constructing HerwigClusterDcayer
Initializing Trained ML Models
Model input directory:  ../../data/models/cluster_decayer_pert.onnx
original outputs from GAN: 0.421606 0.875085 
incoming cluster with 4 vector: 25.5166 17.3116 0.866833 18.6806 
 produced two hadrons [pions] with 4 vector:

[GAN]    19.974 13.6974 1.15834 14.4908 
[Herwig] 19.0023 12.8332 1.19091 13.9629 

[GAN]    5.54256 3.61422 -0.291505 4.18976 
[Herwig] 6.5143 4.4784 -0.324079 4.71771 
```


# Install onnxruntime on ubuntu

* One can follow the commands in this docker file [Install OnnxRunTime](https://github.com/leimao/ONNX-Runtime-Inference/blob/main/docker/onnxruntime-cuda.Dockerfile#L115);
* Or, download the precompiled packages [ONNXRunTime](https://onnxruntime.ai/docs/how-to/install.html#inference).

I did not test the second option. The first is tested and used to build the previous image.