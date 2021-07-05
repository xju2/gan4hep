#!/bin/bash
#SBATCH -J gan
#SBATCH -C gpu
#SBATCH -G 1
#SBATCH -N 1
#SBATCH -c 10
#SBATCH -t 4:00:00
#SBATCH -A m1759
#SBATCH -o %x-%j.out
#SBATCH --mail-type=END
#SBATCH --mail-user=xju@lbl.gov

which python
#train_mlp_gan.py inputs mlpGAN/v1 --max-epochs 10 --batch-size 2048 --shuffle-size -1 --noise-dim 16 --log-freq 100 --warm-up --with-disc-reg --gamma-reg 0.1 

#train_rnn_mlp_gan.py inputs rnnGAN/v0 --max-epochs 10 --batch-size 2048 --shuffle-size -1 --noise-dim 16 --log-freq 100 --warm-up --with-disc-reg --gamma-reg 3.0

#train_rnn_mlp_gan.py inputs rnnGAN/v1 --max-epochs 10 --batch-size 2048 --shuffle-size -1 --noise-dim 8 --log-freq 1000 --warm-up --disc-batches 1000 --with-disc-reg --gamma-reg 1.0 --disc-lr 0.00001 --gen-lr 0.0001

#train_rnn_mlp_gan.py inputs rnnGAN/v2 --max-epochs 10 --batch-size 2048 --shuffle-size -1 --noise-dim 32 --log-freq 1000 --warm-up --disc-batches 1000 --with-disc-reg --gamma-reg 1.0 --disc-lr 0.00001 --gen-lr 0.0001
#train_rnn_mlp_gan.py inputs rnnGAN/v2 --max-epochs 10 --batch-size 2048 --shuffle-size -1 --noise-dim 32 --log-freq 1000 --warm-up --disc-batches 1000 --with-disc-reg --gamma-reg 1.0 --disc-lr 0.001 --gen-lr 0.0001
#train_rnn_mlp_gan.py inputs rnnGAN/v2 --max-epochs 10 --batch-size 2048 --shuffle-size -1 --noise-dim 32 --log-freq 1000 --warm-up --disc-batches 1000 --with-disc-reg --gamma-reg 1.0 --disc-lr 0.0001 --gen-lr 0.00001

# added tanh in last layer; force the energy term to the positive; the regularization term is very small, put a larger scale
#train_rnn_mlp_gan.py inputs rnnGAN/v3 --max-epochs 10 --batch-size 2048 --shuffle-size -1 --noise-dim 32 --log-freq 1000 --warm-up --disc-batches 1000 --with-disc-reg --gamma-reg 5.0 --disc-lr 0.0001 --gen-lr 0.00001

# now remove the tanh in last layer
#train_rnn_mlp_gan.py inputs rnnGAN/v4 --max-epochs 10 --batch-size 2048 --shuffle-size -1 --noise-dim 32 --log-freq 1000 --warm-up --disc-batches 1000 --with-disc-reg --gamma-reg 1.0 --disc-lr 0.00001 --gen-lr 0.0001

# inputs has 11M training events; Running for 4 hours only running 2.4 epochs. To get 100 epochs, it would need to run the job for about one week (167 hours). 
# to check how the epochs affect the result, take 10% of inputs
# because I am sharing the code with others, moving to a new repo.
#train_gan4hep.py inputs rnnGAN/v1 --gan-type rnn_mlp_gan --max-epochs 15 --batch-size 2048 --shuffle-size 4096 --noise-dim 32 --log-freq 1000 --warm-up --disc-batches 1000 --with-disc-reg --gamma-reg 5.0 --disc-lr 0.001 --gen-lr 0.0001 --input-frac 0.1
# it takes 3 hours for runing 20 epochs
#train_gan4hep.py inputs rnnGAN/v1 --gan-type rnn_mlp_gan --max-epochs 20 --batch-size 2048 --shuffle-size 4096 --noise-dim 32 --log-freq 1000 --with-disc-reg --gamma-reg 5.0 --disc-lr 0.001 --gen-lr 0.0001 --input-frac 0.1
# the update are not quiet visible, change learning rate one order of magnitude
#train_gan4hep.py inputs rnnGAN/v1 --gan-type rnn_mlp_gan --max-epochs 20 --batch-size 2048 --shuffle-size 4096 --noise-dim 32 --log-freq 1000 --with-disc-reg --gamma-reg 5.0 --disc-lr 0.0001 --gen-lr 0.00001 --input-frac 0.1


# Now I increase the input dataset by a factor of 5, 3 hours would process 4 epochs
# I think I changed to smaller learning rate that made the discriminator unstable??
# train_gan4hep.py inputs rnnGAN/v2 --gan-type rnn_mlp_gan --max-epochs 4 --batch-size 2048 --shuffle-size 4096 --noise-dim 32 --log-freq 1000 --with-disc-reg --gamma-reg 1.0 --disc-lr 0.0001 --gen-lr 0.0001 --input-frac 0.5

# I sense the regularization term is too strong.
# and it makes GAN unstable.
#train_gan4hep.py inputs rnnGAN/v3 --gan-type rnn_mlp_gan --max-epochs 4 --batch-size 2048 --shuffle-size 4096 --noise-dim 32 --log-freq 1000 --with-disc-reg --gamma-reg 0.01 --disc-lr 0.001 --gen-lr 0.0001 --input-frac 0.5

# I continue with v2 and change back lr and use larger gamma
# train_gan4hep.py inputs rnnGAN/v4 --gan-type rnn_mlp_gan --max-epochs 4 --batch-size 2048 --shuffle-size -1 --noise-dim 32 --log-freq 1000 --with-disc-reg --gamma-reg 5.0 --disc-lr 0.001 --gen-lr 0.0001 --input-frac 0.5

# hyo from graphGAN
#train_gan4hep.py inputs rnnGAN/v5 --gan-type rnn_mlp_gan --max-epochs 4 --batch-size 2048 --shuffle-size -1 --noise-dim 64 --log-freq 1000 --with-disc-reg --gamma-reg 5.0 --disc-lr 0.00003 --gen-lr 0.00001 --input-frac 0.5

# Now RNN-RNN GAN
#train_gan4hep.py inputs rnn_rnn_GAN/v0 --gan-type rnn_rnn_gan --max-epochs 8 --batch-size 2048 --shuffle-size -1 --noise-dim 12 --log-freq 1000 --with-disc-reg --gamma-reg 5.0 --disc-lr 0.00003 --gen-lr 0.00001 --input-frac 0.2

#train_gan4hep.py inputs rnn_rnn_GAN/v1 --gan-type rnn_rnn_gan --max-epochs 8 --batch-size 2048 --shuffle-size -1 --noise-dim 12 --log-freq 1000 --with-disc-reg --gamma-reg 5.0 --disc-lr 0.00003 --gen-lr 0.00001 --input-frac 0.2 --use-pt-eta-phi-e

#train_gan4hep.py inputs rnn_rnn_GAN/v2 --gan-type rnn_rnn_gan --max-epochs 8 --batch-size 2048 --shuffle-size -1 --noise-dim 8 --log-freq 200 --with-disc-reg --gamma-reg 5.0 --disc-lr 0.003 --gen-lr 0.001 --input-frac 0.2 --warm-up --disc-batches 10 --disable-tqdm

# >>>>>>>>>>>>>>>>>>>> GNN >>>>>>>>>>>>>>>>>>>> 
#train_gan4hep.py inputs gnn_gnn/v0 --gan-type gnn_gnn_gan --max-epochs 7 --batch-size 128 --shuffle-size -1 --noise-dim 8 --log-freq 1000 --with-disc-reg --gamma-reg 1.0 --disc-lr 0.0003 --gen-lr 0.0001 --input-frac 0.2 --warm-up --disc-batches 50 --disable-tqdm --val-batches 50

# noise generated from uniform distribution [-1, 1]
# new GNN.. the previous one would not work
train_gan4hep.py inputs gnn_gnn/v1 --gan-type gnn_gnn_gan --max-epochs 7 --batch-size 128 --shuffle-size -1 --noise-dim 8 --log-freq 1000 --with-disc-reg --gamma-reg 1.0 --disc-lr 0.0003 --gen-lr 0.0001 --input-frac 0.2 --warm-up --disc-batches 100 --disable-tqdm --val-batches 100

# use pt-eta-phi
#train_gan4hep.py inputs rnnGAN/v6 --gan-type rnn_mlp_gan --max-epochs 10 --batch-size 2048 --shuffle-size -1 --noise-dim 12 --log-freq 1000 --with-disc-reg --gamma-reg 5.0 --disc-lr 0.00003 --gen-lr 0.00001 --input-frac 0.1 --use-pt-eta-phi-e
