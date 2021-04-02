# gan4hep
Use GAN to generate HEP events

# Instructions
Following command trains the selected model
```bash
train_gan4hep.py inputs mlpGan --gan-type mlp_gan --max-epochs 10 --batch-size 100 --shuffle-size -1 --noise-dim 12 --log-freq 10 --with-disc-reg --gamma-reg 5.0 --disc-lr 0.00003 --gen-lr 0.00001 --input-frac 0.1 --use-pt-eta-phi-e --debug 
```
where `inputs` contains input dataset and `mlpGan` is the output directory.