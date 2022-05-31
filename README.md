# gan4hep
This repostiory houses generative models developed to generate HEP events.
You can find introductions of training
[Generative Adversial Network](gan4hep/gan/README.md) 
and  [Normaling Flow](gan4hep/nf/README.md). 
These modules are more or less standalone.
However, it needs a `reader` that reads the input files.

You can also find examples
of converting trained GAN models into onnx format and performing the
inference in C++ with [onnxruntime](gan4hep/onnx/keras_examples/Readme.md).

# Installation
Install the repository to get supports on reading files.

```
pip install -e . 
```

# Examples
Please check the wiki page for examples.