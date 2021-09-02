#!/user/bin/env python 


from gan import GAN
from aae import AdversarialAutoencoder
from cgan import CGAN
from wgan import WGAN

all_gans = ['GAN', "AdversarialAutoencoder", 'CGAN', 'WGAN']

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
    add_arg("--lr", help='learning rate', type=float, default=0.0001)
    args = parser.parse_args()

    from tensorflow.compat.v1 import logging
    logging.set_verbosity(args.verbose)


    from gan4hep.utils_gan import generate_and_save_images
    from gan4hep.preprocess import herwig_angles

    train_in, train_truth, test_in, test_truth = herwig_angles(
        args.filename, max_evts=args.max_evts)

    batch_size = args.batch_size
    gan = eval(args.model)()
    gan.train(
        train_truth, args.epochs, batch_size,
        test_truth, args.log_dir,
        generate_and_save_images,
        train_in, test_in
    )