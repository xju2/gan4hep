Scripts in this module only depend on the `gnn4hep.preprocess`, 
which reads a given file and prepare input tensors for training these generative models.

The entrance script for training a GAN is `train_gan.py`. 
Use `train_gan.py -h` to check various options.

To train a GAN for dimuon events:
```bash
python train_gan.py GAN mc16d_364100_100Kentries_dimuon.output \
    --data dimuon_inclusive \
    --cond-dim 0 --gen-output-dim 6
```
which assumes an input file of `mc16d_364100_100Kentries_dimuon.output` and 
the GAN is a normal MLP-based GAN.