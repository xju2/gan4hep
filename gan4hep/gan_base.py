"""
MLP-based GANs
"""
from types import SimpleNamespace
import functools
from typing import Callable, Iterable, Optional, Text

import tensorflow as tf
import sonnet as snt


def sum_trainables(module):
    return sum([tf.size(v) for v in module.trainable_variables])

class GANBase(snt.Module):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.generator = None
        self.discriminator = None

    def generate(self, inputs, is_training=True):
        return self.generator(inputs, is_training)

    def discriminate(self, inputs, is_training=True):
        return self.discriminator(inputs, is_training)

    def num_trainable_vars(self):
        return sum_trainables(self.generator), sum_trainables(self.discriminator)


def Discriminator_Regularizer(p_true, grad_D_true_logits, p_gen, grad_D_gen_logits, batch_size):
    """
    Args:
        p_true: probablity from Discriminator for true events
        grad_D_true_logits: gradient of Discrimantor logits w.r.t its input variables
        p_gen: probability from Discriminator for generated events
        grad_D_gen_logits: gradient of Discrimantor logits w.r.t its input variables
    Returns:
        discriminator regularizer
    """
    grad_D_true_logits_norm = tf.norm(
        tf.reshape(grad_D_true_logits, [batch_size, -1]),
        axis=1, keepdims=True
    )
    grad_D_gen_logits_norm = tf.norm(
        tf.reshape(grad_D_gen_logits, [batch_size, -1]),
        axis=1, keepdims=True
    )
    assert grad_D_true_logits_norm.shape == p_true.shape, "{} {}".format(grad_D_true_logits_norm.shape, p_true.shape)
    assert grad_D_gen_logits_norm.shape == p_gen.shape, "{} {}".format(grad_D_gen_logits_norm.shape, p_gen.shape)
        
    reg_true = tf.multiply(tf.square(1.0 - p_true), tf.square(grad_D_true_logits_norm))
    reg_gen = tf.multiply(tf.square(p_gen), tf.square(grad_D_gen_logits_norm))
    disc_regularizer = tf.reduce_mean(reg_true + reg_gen)
    return disc_regularizer, grad_D_true_logits_norm, grad_D_gen_logits_norm, reg_true, reg_gen


class GANOptimizer(snt.Module):

    def __init__(self,
                gan, 
                batch_size=100,
                noise_dim=128,
                disc_lr=2e-4,
                gen_lr=5e-5,
                num_epochs=100,
                loss_type='logloss',
                gamma_reg=1e-3, 
                name=None, *args, **kwargs):
        super().__init__(name=name)
        self.gan = gan
        self.hyparams = SimpleNamespace(
            batch_size=batch_size,
            noise_dim=noise_dim,
            disc_lr=disc_lr,
            gen_lr=gen_lr,
            num_epochs=num_epochs,
            with_disc_reg=True,
            gamma_reg=gamma_reg,
            loss_type=loss_type,
        )

        # self.disc_lr = tf.Variable(
        #     disc_lr, trainable=False, name='disc_lr', dtype=tf.float32)
        # self.gen_lr = tf.Variable(
        #     gen_lr, trainable=False, name='gen_lr', dtype=tf.float32)

        # two different optimizers        
        self.disc_opt = snt.optimizers.SGD(learning_rate=self.hyparams.disc_lr)
        self.gen_opt = snt.optimizers.Adam(learning_rate=self.hyparams.gen_lr)

        self.num_epochs = tf.constant(num_epochs, dtype=tf.int32)

        self.loss_fn = tf.nn.sigmoid_cross_entropy_with_logits \
            if loss_type == 'logloss' else tf.compat.v1.losses.mean_squared_error


    def get_noise_batch(self):
        noise_shape = [self.hyparams.batch_size, self.hyparams.noise_dim]
        return tf.random.normal(noise_shape, dtype=tf.float32)


    def disc_step(self, truth_inputs, cond_inputs=None, lr_mult=1.0):
        gan = self.gan

        inputs = self.get_noise_batch()
        if cond_inputs is not None:
            inputs = tf.concat([cond_inputs, inputs], axis=-1)


        with tf.GradientTape() as tape, tf.GradientTape() as true_tape, tf.GradientTape() as fake_tape:
            gen_evts = gan.generate(inputs)
            if cond_inputs is not None:
                gen_evts = tf.concat([cond_inputs, gen_evts], axis=-1)

            true_tape.watch(truth_inputs)
            fake_tape.watch(gen_evts)
            real_output = gan.discriminate(truth_inputs)
            fake_output = gan.discriminate(gen_evts)

            loss = tf.reduce_mean(self.loss_fn(tf.ones_like(real_output), real_output) \
                                + self.loss_fn(tf.zeros_like(fake_output), fake_output))

            if self.hyparams.with_disc_reg:
                grad_logits_true = true_tape.gradient(real_output, truth_inputs)
                grad_logits_gen = fake_tape.gradient(fake_output, gen_evts)

                real_scores = tf.sigmoid(real_output) if self.hyparams.loss_type == "logloss"\
                    else real_output
                fake_scores = tf.sigmoid(fake_output) if self.hyparams.loss_type == "logloss"\
                    else fake_output

                regularizers = Discriminator_Regularizer(
                    real_scores,
                    grad_logits_true,
                    fake_scores,
                    grad_logits_gen,
                    self.hyparams.batch_size,
                )
                reg_loss = regularizers[0]
                assert reg_loss.shape == loss.shape
                loss += self.hyparams.gamma_reg*reg_loss

        disc_params = gan.discriminator.trainable_variables
        disc_grads = tape.gradient(loss, disc_params)

        self.disc_opt.apply(disc_grads, disc_params)
        if self.hyparams.with_disc_reg:
            return loss, *regularizers
        else:
            return loss,


    def gen_step(self, inputs_tr=None, lr_mult=1.0):
        gan = self.gan
        inputs = self.get_noise_batch()
        if inputs_tr is not None:
            inputs = tf.concat([inputs_tr, inputs], axis=-1)

        with tf.GradientTape() as tape:
            gen_graph = gan.generate(inputs)
            if inputs_tr is not None:
                gen_graph = tf.concat([inputs_tr, gen_graph], axis=-1)
            fake_output = gan.discriminate(gen_graph)

            loss = tf.reduce_mean(self.loss_fn(tf.ones_like(fake_output), fake_output))

        gen_params = gan.generator.trainable_variables
        gen_grads = tape.gradient(loss, gen_params)

        self.gen_opt.apply(gen_grads, gen_params)
        return loss

    def _get_lr_mult(self, epoch):
        # No learning rate decay
        return tf.constant(1., dtype=tf.float32)

    def step(self, targets_tr, epoch, inputs_tr):
        lr_mult = self._get_lr_mult(epoch)
        disc_loss = self.disc_step(targets_tr, inputs_tr, lr_mult=lr_mult)
        gen_loss = self.gen_step(inputs_tr, lr_mult=lr_mult)
        return disc_loss, gen_loss, lr_mult

    def cond_gen(self, inputs_tr):
        gan = self.gan
        noises = self.get_noise_batch()
        inputs = tf.concat([inputs_tr, noises], axis=-1)
        gen_evts = gan.generate(inputs)
        return gen_evts