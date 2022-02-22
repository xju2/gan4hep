#!/user/bin/env python
import time
import os
import tqdm
import numpy as np

from scipy import stats
import tensorflow as tf

from tensorflow.compat.v1 import logging
logging.info("TF Version:{}".format(tf.__version__))
gpus = tf.config.experimental.list_physical_devices("GPU")
logging.info("found {} GPUs".format(len(gpus)))
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

from tensorflow import keras
from gan import GAN
# from aae import AAE
# from cgan import CGAN
# from wgan import WGAN

all_gans = ['GAN', "AAE", 'CGAN', 'WGAN']

from gan4hep.preprocess import herwig_angles
from gan4hep.preprocess import herwig_angles2
from gan4hep.preprocess import dimuon_inclusive


from utils import evaluate, log_metrics
from gan4hep.utils_plot import compare

cross_entropy = keras.losses.BinaryCrossentropy(from_logits=False)
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return tf.reduce_mean(total_loss)

def generator_loss(fake_output):
    return tf.reduce_mean(cross_entropy(tf.ones_like(fake_output), fake_output))


def train(train_truth, test_truth, model, gen_lr, disc_lr, batch_size,
    max_epochs, log_dir, xlabels, disable_tqdm=False, train_in=None, test_in=None):

    noise_dim = model.noise_dim
    generator = model.generator
    discriminator = model.discriminator
    
    # ======================================
    # construct testing data once for all
    # ======================================
    AUTO = tf.data.experimental.AUTOTUNE
    noise = np.random.normal(loc=0., scale=1., size=(test_truth.shape[0], noise_dim))
    test_in = np.concatenate(
        [test_in, noise], axis=1).astype(np.float32) if test_in is not None else noise
    testing_data = tf.data.Dataset.from_tensor_slices(
        (test_in, test_truth)).batch(batch_size, drop_remainder=True).prefetch(AUTO)

    # ====================================
    # Checkpoints and model summary
    # ====================================
    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    checkpoint = tf.train.Checkpoint(
        generator=generator,
        discriminator=discriminator)
    ckpt_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=None)
    logging.info("Loading latest checkpoint from: {}".format(checkpoint_dir))
    _ = checkpoint.restore(ckpt_manager.latest_checkpoint).expect_partial()

    time_stamp = time.strftime('%Y%m%d-%H%M%S', time.localtime())
    summary_dir = os.path.join(log_dir, "logs", f"{time_stamp}")
    summary_writer = tf.summary.create_file_writer(summary_dir)

    img_dir = os.path.join(log_dir, 'img', f'{time_stamp}')
    os.makedirs(img_dir, exist_ok=True)

    # ======================
    # Create optimizers
    # ======================
    end_lr = 1e-6
    # gen_lr = keras.optimizers.schedules.PolynomialDecay(gen_lr, max_epochs, end_lr, power=4)
    # disc_lr = keras.optimizers.schedules.PolynomialDecay(disc_lr, max_epochs, end_lr, power=1.0)
    generator_optimizer = keras.optimizers.Adam(gen_lr)
    discriminator_optimizer = keras.optimizers.Adam(disc_lr)

    @tf.function
    def train_step(gen_in_4vec, truth_4vec):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_out_4vec = generator(gen_in_4vec, training=True)

            real_output = discriminator(truth_4vec, training=True)
            fake_output = discriminator(gen_out_4vec, training=True)

            gen_loss = generator_loss(fake_output)
            disc_loss = discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

        return disc_loss, gen_loss

    @tf.function
    def train_disc_only(gen_in_4vec, truth_4vec):
        with tf.GradientTape() as disc_tape:
            gen_out_4vec = generator(gen_in_4vec, training=False)

            real_output = discriminator(truth_4vec, training=True)
            fake_output = discriminator(gen_out_4vec, training=True)

            disc_loss = discriminator_loss(real_output, fake_output)
            gen_loss = generator_loss(fake_output)

        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
        return disc_loss, gen_loss

    best_wdis = 9999
    best_epoch = -1
    plaeto_threashold = 100000
    num_no_improve = 0
    do_disc_only = False
    num_disc_only_epochs = 0
    i_disc_only = 0
    summary_logfile = os.path.join(summary_dir, f'results_{time_stamp}.txt')
    
    with tqdm.trange(max_epochs, disable=disable_tqdm) as t0:
        for epoch in t0:
            # compose the training dataset by generating different noises for each epochs
            noise = np.random.normal(loc=0., scale=1., size=(train_truth.shape[0], noise_dim))
            train_inputs = np.concatenate(
                [train_in, noise], axis=1).astype(np.float32) if train_in is not None else noise


            dataset = tf.data.Dataset.from_tensor_slices(
                (train_inputs, train_truth)).shuffle(2*batch_size).batch(
                    batch_size, drop_remainder=False).prefetch(AUTO)

            tot_loss = []
            
                
            # train_fn = train_disc_only if do_disc_only else train_step
            train_fn = train_step
            i_disc_only += do_disc_only

            for data_batch in dataset:
                tot_loss.append(list(train_fn(*data_batch)))

            tot_loss = np.array(tot_loss)
            avg_loss = np.sum(tot_loss, axis=0)/tot_loss.shape[0]
            loss_dict = dict(D_loss=avg_loss[0], G_loss=avg_loss[1])

            predictions, truths = evaluate(generator, testing_data)
            tot_wdis = sum([stats.wasserstein_distance(predictions[:, idx], truths[:, idx])\
                    for idx in range(truths.shape[1])])


            with summary_writer.as_default():
                tf.summary.experimental.set_step(epoch)
                tf.summary.scalar("tot_wasserstein_dis",
                    tot_wdis, description="total wasserstein distance")
                for key,val in loss_dict.items():
                    tf.summary.scalar(key, val)

            if tot_wdis < best_wdis:
                ckpt_manager.save()
                generator.save(os.path.join(log_dir, "generator"))
                best_wdis = tot_wdis
                best_epoch = epoch
                outname = os.path.join(img_dir, f"{epoch}.png")
                compare(predictions, truths, outname, xlabels)
                with open(summary_logfile, 'a') as f:
                    f.write(", ".join(["{:.4f}".format(x) 
                        for x in [best_wdis, best_epoch]]) + '\n')
            else:
                num_no_improve += 1
            
            # so long since last improvement
            # train discriminator only
            if num_no_improve > plaeto_threashold:
                num_no_improve = 0
                num_disc_only_epochs += 1
                do_disc_only = True
            else:
                if i_disc_only == num_disc_only_epochs:
                    do_disc_only = False
                    i_disc_only = 0


            t0.set_postfix(**loss_dict, doOnlyDisc=do_disc_only, BestD=best_wdis, BestE=best_epoch)

    tmp_res = "Best Model in {} Epoch with a Wasserstein distance {:.4f}".format(best_epoch, best_wdis)
    logging.info(tmp_res)
    
    with open(summary_logfile, 'a') as f:
        f.write(tmp_res + "\n")


def inference(gan, test_in, test_truth, log_dir, xlabels):
    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    checkpoint = tf.train.Checkpoint(
        generator=gan.generator,
        discriminator=gan.discriminator)
    ckpt_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=None)
    logging.info("Loading latest checkpoint from: {}".format(checkpoint_dir))
    _ = checkpoint.restore(ckpt_manager.latest_checkpoint).expect_partial()

    AUTO = tf.data.experimental.AUTOTUNE
    noise = np.random.normal(loc=0., scale=1., size=(test_truth.shape[0], gan.noise_dim))
    test_in = np.concatenate(
        [test_in, noise], axis=1).astype(np.float32) if test_in is not None else noise
    testing_data = tf.data.Dataset.from_tensor_slices(
        (test_in, test_truth)).batch(batch_size, drop_remainder=True).prefetch(AUTO)

    summary_dir = os.path.join(log_dir, "logs_inference")
    summary_writer = tf.summary.create_file_writer(summary_dir)

    img_dir = os.path.join(log_dir, 'img_inference')
    os.makedirs(img_dir, exist_ok=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train The GAN')
    add_arg = parser.add_argument
    add_arg("model", choices=all_gans, help='gan model')
    add_arg("filename", help='input filename', default=None, nargs='+')
    add_arg("--epochs", help='number of maximum epochs', default=100, type=int)
    add_arg("--log-dir", help='log directory', default='log_training')
    add_arg("--num-test-evts", help='number of testing events', default=10000, type=int)
    add_arg("--inference", help='perform inference only', action='store_true')
    add_arg("-v", '--verbose', help='tf logging verbosity', default='INFO',
        choices=['WARN', 'INFO', "ERROR", "FATAL", 'DEBUG'])
    add_arg("--max-evts", help='Maximum number of events', type=int, default=None)
    add_arg("--batch-size", help='Batch size', type=int, default=512)
    add_arg("--gen-lr", help='generator learning rate', type=float, default=0.0001)
    add_arg("--disc-lr", help='discriminator learning rate', type=float, default=0.0001)
    add_arg("--data", default='herwig_angles',
        choices=['herwig_angles', 'dimuon_inclusive', 'herwig_angles2'])

    # model parameters
    add_arg("--noise-dim", type=int, default=4, help="noise dimension")
    add_arg("--gen-output-dim", type=int, default=2, help='generator output dimension')
    add_arg("--cond-dim", type=int, default=0, help='dimension of conditional input')
    add_arg("--disable-tqdm", action="store_true", help='disable tqdm')

    args = parser.parse_args()

    from tensorflow.compat.v1 import logging
    logging.set_verbosity(args.verbose)

    # prepare input data by calling those function implemented in 
    # gan4hep.preprocess.
    train_in, train_truth, test_in, test_truth, xlabels = eval(args.data)(
        args.filename, max_evts=args.max_evts)

    batch_size = args.batch_size
    gan = eval(args.model)(**vars(args))
    if args.inference:
        inference(gan, test_in, test_truth, args.log_dir, xlabels)
    else:
        train(train_truth, test_truth, gan, args.gen_lr, args.disc_lr,
            batch_size, args.epochs, args.log_dir, xlabels, args.disable_tqdm,
            train_in, test_in)