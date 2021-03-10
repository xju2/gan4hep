# gan4hep
Use GAN to generate HEP events

# Instructions
Following command trains the selected model
```bash
train_gan4hep.py inputs mlpGan --gan-type mlp_gan --max-epochs 1 --batch-size 4 --shuffle-size -1 --noise-dim 32 --log-freq 1000 --warm-up --disc-batches 5 --with-disc-reg --gamma-reg 5.0 --disc-lr 0.0001 --gen-lr 0.00001 --debug
```
where `inputs` contains input dataset and `mlpGan` is the output directory.
