    """
    The primary training loop
    """
    base_lr = lr
    end_lr = 1e-5
    learning_rate_fn = tfk.optimizers.schedules.PolynomialDecay(
        base_lr, max_epochs, end_lr, power=0.5)

    # initialize checkpoints
    checkpoint_directory = "{}/checkpoints".format(outdir)
    os.makedirs(checkpoint_directory, exist_ok=True)

    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn)  # optimizer
    checkpoint = tf.train.Checkpoint(optimizer=opt, model=flow_model)
    ckpt_manager = tf.train.CheckpointManager(checkpoint, checkpoint_directory, max_to_keep=None)
    _ = checkpoint.restore(ckpt_manager.latest_checkpoint).expect_partial()


    AUTO = tf.data.experimental.AUTOTUNE
    training_data = tf.data.Dataset.from_tensor_slices(
        train_truth).batch(batch_size).prefetch(AUTO)

    img_dir = os.path.join(outdir, "imgs")
