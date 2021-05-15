# gan4hep
Use GAN to generate HEP events

# Instructions
Following command trains a GNN-based GAN:
```bash
train_gan4hep.py inputs gnn_gnn/v1 --gan-type gnn_gnn_gan --max-epochs 7 --batch-size 128 --shuffle-size -1 --noise-dim 8 --log-freq
1000 --with-disc-reg --gamma-reg 1.0 --disc-lr 0.0003 --gen-lr 0.0001 --input-frac 0.2 --warm-up --disc-batches 100 --disable-tqdm --val-batches 100
```
where `inputs` contains input dataset and `gnn_gnn/v1` is the output directory.
