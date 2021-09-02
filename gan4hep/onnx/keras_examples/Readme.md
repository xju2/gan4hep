# Compile the code in a docker container
As a starting point, one can play with the code with the docker container by running

```bash
docker run -it --rm -v $PWD:$PWD -w $PWD --gpus all docexoty/mltools:ubuntu20.04
```

then `mkdir build && cd build && cmake .. && make`. 
It will compile the code and produce an executable `inference`;
Run `./bin/inference`


# Install onnxruntime on ubuntu

* One can follow the commands in this docker file [Install OnnxRunTime](https://github.com/leimao/ONNX-Runtime-Inference/blob/main/docker/onnxruntime-cuda.Dockerfile#L115);
* Or, download the precompiled packages [ONNXRunTime](https://onnxruntime.ai/docs/how-to/install.html#inference).

I did not test the second option. The first is tested and used to build the previous image.