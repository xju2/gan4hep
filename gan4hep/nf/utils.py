import tensorflow as tf

__all__= [
    "train_density_estimation",
    "train_density_estimation_cond",
    "nll"]
@tf.function
def train_density_estimation(distribution, optimizer, batch):
    """
    Train function for density estimation normalizing flows.
    :param distribution: TensorFlow distribution, e.g. tf.TransformedDistribution.
    :param optimizer: TensorFlow keras optimizer, e.g. tf.keras.optimizers.Adam(..)
    :param batch: Batch of the train data.
    :return: loss.
    """
    with tf.GradientTape() as tape:
        tape.watch(distribution.trainable_variables)
        loss = -tf.reduce_mean(distribution.log_prob(batch))  # negative log likelihood
    gradients = tape.gradient(loss, distribution.trainable_variables)
    optimizer.apply_gradients(zip(gradients, distribution.trainable_variables))
    return loss


@tf.function
def train_density_estimation_cond(distribution, optimizer, batch, condition, layers):
    """
    Train function for density estimation normalizing flows.
    :param distribution: TensorFlow distribution, e.g. tf.TransformedDistribution.
    :param optimizer: TensorFlow keras optimizer, e.g. tf.keras.optimizers.Adam(..)
    :param batch: Batch of the train data.
    :return: loss.
    """
    cond_kwargs = dict([(f"b{idx}", {"conditional_input": condition}) for idx in range(layers)])
    with tf.GradientTape() as tape:
        tape.watch(distribution.trainable_variables)
        loss = -tf.reduce_mean(distribution.log_prob(batch, bijector_kwargs=cond_kwargs))

    gradients = tape.gradient(loss, distribution.trainable_variables)
    optimizer.apply_gradients(zip(gradients, distribution.trainable_variables))
    return loss


@tf.function
def nll(distribution, data):
    """
    Computes the negative log liklihood loss for a given distribution and given data.
    :param distribution: TensorFlow distribution, e.g. tf.TransformedDistribution.
    :param data: Data or a batch from data.
    :return: Negative Log Likelihodd loss.
    """
    return -tf.reduce_mean(distribution.log_prob(data))


def save_flow(flow, ckpt_path: str,
    output_path: str, cond_data=None, layers=None):
    """Save Masked Autogressive Flow for density estimation

    Arguments:
    flow -- Masked Autogressive Flow
    ckpt_path -- Path to the checkpoint
    output_path -- Path to the output directory
    cond_data -- Conditioned data
    layers -- Number of layers of MADE
    """
    try:
        import tf2onnx
    except ImportError:
        print("pip install -U tf2onnx")
        raise ImportError("tf2onnx is required to save the flow")

    opt = tf.keras.optimizers.Adam(learning_rate=1e-3)  # optimizer
    checkpoint = tf.train.Checkpoint(optimizer=opt, model=flow)
    manager = tf.train.CheckpointManager(checkpoint, ckpt_path, max_to_keep=None)
    checkpoint.restore(manager.latest_checkpoint).expect_partial()

    num_cond = 0 if cond_data is None else cond_data.shape[1]
    ## generate one event at one time.
    ## not efficient, but that is how it works in event generator.
    if cond_data is None:
        input_signature = None
        def inference():
            return flow.sample(1)
    else:
        if layers is None:
            raise ValueError("layers is required when cond_data is given")
        input_signature = [tf.TensorSpec(shape=(1,num_cond), dtype=tf.float32)]
        def inference(x):
            cond_kwargs = dict([(f"b{idx}", {"conditional_input": x}) for idx in range(10)])
            return flow.sample(1, bijector_kwargs=cond_kwargs)

    inf_fn = tf.function(inference, input_signature=input_signature)
    inference(cond_data)
    model_proto, _ = tf2onnx.convert.from_function(
        inf_fn,
        input_signature=input_signature,
        opset=None,
        output_path=output_path,   
    )