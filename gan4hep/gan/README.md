# Introduction

Scripts in this module only depend on the `gnn4hep.preprocess`, 
which reads a given file and prepare input tensors for training.

The entrance script for training a GAN is `train_gan.py`. 
Use `train_gan.py -h` to check various options.

# Examples

## Dimuon events
To train a GAN for dimuon events:
```bash
python train_gan.py GAN mc16d_364100_100Kentries_dimuon.output DiMuonsReader
    --cond-dim 0 --gen-output-dim 6
```
which assumes an input file of `mc16d_364100_100Kentries_dimuon.output` and 
the GAN is a normal MLP-based GAN.


## Herwig events
Please create softlinks to the two scripts: `herwig.py` in `gan4hep/io/herwig.py` 
and `train_gan.py` in `gan4hep/gan/train_gan.py`. Then convert the events. 
```bash
python herwig.py data/cluster_ML_2PI0.txt data/cluster_ML_2PI0_mode2 -m 2
```
Then start the training
```bash
python train_gan.py GAN data/cluster_ML_2PI0_mode2_mode2.npz \
    HerwigReader --cond-dim 4 --gen-output-dim 2
```