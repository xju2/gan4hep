"""
This is a simple MLP-base conditional GAN.
Note that the conditional input is not
given to the discriminator.
"""

import numpy as np
import os


import tensorflow as tf
from tensorflow.compat.v1 import logging
logging.info("TF Version:{}".format(tf.__version__))
gpus = tf.config.experimental.list_physical_devices("GPU")
logging.info("found {} GPUs".format(len(gpus)))
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


from tensorflow import keras
from tensorflow.keras import layers

import tqdm


cross_entropy = keras.losses.BinaryCrossentropy(from_logits=False)
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return tf.reduce_mean(total_loss)

def generator_loss(fake_output):
    return tf.reduce_mean(cross_entropy(tf.ones_like(fake_output), fake_output))


class GAN():
    def __init__(self,
        noise_dim: int = 4, gen_output_dim: int = 2,
        cond_dim: int = 4, disable_tqdm=False, lr_dis=0.01,lr_gen=0.01,gen_layers=1,dis_layers=1,num_nodes=256,**kwargs):
        """
        noise_dim: dimension of the noises
        gen_output_dim: output dimension
        cond_dim: in case of conditional GAN, 
                  it is the dimension of the condition
        """
        self.noise_dim = noise_dim
        self.gen_output_dim = gen_output_dim
        self.cond_dim = cond_dim
        self.disable_tqdm = disable_tqdm

        self.gen_input_dim = self.noise_dim + self.cond_dim

        # ============
        # Optimizers
        # ============
        self.generator_optimizer = keras.optimizers.Adam(lr_gen)
        self.discriminator_optimizer = keras.optimizers.Adam(lr_dis)

        # Build the critic
        self.discriminator = self.build_critic(dis_layers,num_nodes)
        self.discriminator.summary()

        # Build the generator
        self.generator = self.build_generator(gen_layers,num_nodes)
        self.generator.summary()
   
    def build_generator(self,gen_layers,num_nodes):
        
        
        #Old version of the NN in case the new one breaks
        '''

        gen_input_dim = self.gen_input_dim

        model = keras.Sequential([
        keras.Input(shape=(gen_input_dim,)),
        layers.Dense(256),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
            
        layers.Dense(256),
        layers.BatchNormalization(),
            
        layers.Dense(self.gen_output_dim),
        layers.Activation("tanh"),
        ], name='Generator')
        return model
    
    
        '''
        gen_input_dim = self.gen_input_dim

        model = keras.Sequential([
            keras.Input(shape=(gen_input_dim,)),
            ], name='Generator')
        model.add(layers.Dense(num_nodes))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.Dense(num_nodes))
        model.add(layers.BatchNormalization())
   
        
                #Adding the extra layer if needed
        for i in range(int(gen_layers)):
            model.add(layers.Dense(num_nodes))
            model.add(layers.BatchNormalization())
            
        model.add(layers.Dense(self.gen_output_dim))
        model.add(layers.Activation("tanh"))

        return model
        
    def build_critic(self,dis_layers,num_nodes):
        '''
         #Old version of the NN in case the new one breaks
        
        # <NOTE> conditional input is not given
        gen_output_dim = self.gen_output_dim

        model = keras.Sequential([
        keras.Input(shape=(gen_output_dim,)),
        layers.Dense(256),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
            
        layers.Dense(256),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        layers.Dense(1, activation='sigmoid'),
        ], name='Discriminator')
        return model
        '''
        # <NOTE> conditional input is not given
        gen_output_dim = self.gen_output_dim

        #print(dis_layers)
        model = keras.Sequential([
            keras.Input(shape=(gen_output_dim,)),
             ], name='Discriminator')
        model.add(layers.Dense(num_nodes))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.Dense(num_nodes))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        
        #Adding the extra layer
        for i in range(int(dis_layers)):
            model.add(layers.Dense(num_nodes))
            model.add(layers.BatchNormalization())
            model.add(layers.LeakyReLU())
        
        model.add(layers.Dense(1, activation='sigmoid'))

        return model
        
    #Main Train function called at the end of train_gay.py
    def train(self,args, train_truth, epochs, batch_size, test_truth, log_dir, evaluate_samples_fn, generate_and_save_images_end_of_run,lr_dis,lr_gen,noise_type, gen_layers, dis_layers,num_nodes,gen_train_num,dis_train_num,single_step_limit,
            train_in=None, test_in=None):

            # ======================================
            # construct testing data once for all
            # ======================================
            AUTO = tf.data.experimental.AUTOTUNE


            #Selecting noise type based on user input
            if noise_type=='gaussian':
                noise = np.random.normal(loc=0., scale=1., size=(test_truth.shape[0], self.noise_dim)) 
                

            elif noise_type=='uniform':
                noise = np.random.uniform(-1, 1, size=(test_truth.shape[0], self.noise_dim)) 


            #Sanity Checks
            print(test_truth.shape[0],'test-truth dataset size')
            #print(test_in.shape[0],'test_in size') Currently Empty
            print(train_truth.shape[0],'train_truth dataset size')
            #print(train_in.shape[0],'train_in size') Currently Empty

            test_in = np.concatenate(
                [test_in, noise], axis=1).astype(np.float32) if test_in is not None else noise 

            testing_data = tf.data.Dataset.from_tensor_slices(
                (test_in, test_truth)).batch(batch_size, drop_remainder=True).prefetch(AUTO) 


            # ====================================
            # Checkpoints and model summary
            # ====================================

            checkpoint_dir = os.path.join(log_dir, "checkpoints") #Create directory to store checkpoints, path given by terminal             input log_dir

            #Train using the generator and dicriminator and save each step as checkpoints
            checkpoint = tf.train.Checkpoint(
                generator=self.generator,
                discriminator=self.discriminator)

            ckpt_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=None) #Where to save checkpoints

            #Next two lines have been commented out because I think they were breaking the code whenever I added extra layers to the GAN 
            #logging.info("Loading latest checkpoint from: {}".format(checkpoint_dir)) #Logs a message with level INFO on the root             logger. The arguments are interpreted as for debug().

            #_ = checkpoint.restore(ckpt_manager.latest_checkpoint)#.expect_partial() # Restore the checkpointed values to the model

            summary_dir = os.path.join(log_dir, "logs")
            summary_writer = tf.summary.create_file_writer(summary_dir) #Creates a summary file writer for the given log directory.

            img_dir = os.path.join(log_dir, 'img')
            os.makedirs(img_dir, exist_ok=True)


            import time
            import pathlib
            import datetime

            #Making seperate folders for each run to store plots
            #Get time and date of current run
            current_date_and_time = datetime.datetime.now()
            current_date_and_time_string = str(current_date_and_time)

            #Create directory path
            run_dir = os.path.join(img_dir, 'img')

            #Add GAN paramaters to time and date to create file name for current run
            new_run_folder=run_dir+current_date_and_time_string+' Epoch_num: '+ str(args.epochs)+' Batch_Size: '+str(batch_size) +' Number of Events: '+str(args.num_test_evts) +' GAN Type: '+str(args.model) + ' Max Number of Events: ' +str(args.max_evts) + ' Learning Rate: ' + str(args.lr_dis) + ' Noise Type: ' + str(args.noise_type) + ' Number of Noise Dimensions: ' + str(args.noise_dim) # + ' Number of Extra G Layers: ' + str(args.gen_layers)  + ' Number of Extra D Layers: ' + str(args.dis_layers)  #  + ' Fraction of testing events ' + str(args.test_frac)# + ' Number of Nodes per Layer: ' + str(args.num_nodes)

            #Make Directory
            os.makedirs(new_run_folder, exist_ok=True)
            @tf.function

            #Functions to train Generator and Discriminator seperately
            #new
            
            #Train Generator
            def gen_opt(gen_in_4vec, truth_4vec,i):
                with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                    real_output = self.discriminator(truth_4vec, training=True)
                    gen_out_4vec = self.generator(gen_in_4vec, training=True)
                    fake_output = self.discriminator(gen_out_4vec, training=True)
                    gen_loss = generator_loss(fake_output) # Calculate Loss

                    return gen_loss
                
            #Train Discriminator  
            def dis_opt(gen_in_4vec, truth_4vec):
                with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                    gen_out_4vec = self.generator(gen_in_4vec, training=True)
                    real_output = self.discriminator(truth_4vec, training=True)
                    fake_output = self.discriminator(gen_out_4vec, training=True)
                    disc_loss = discriminator_loss(real_output, fake_output) #Calculate Loss
                  
                    return disc_loss

            #Train step for trainig generator and discriminator for each batch
            def train_step(gen_in_4vec, truth_4vec,single_step_limit,epoch): 
                with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                    #new
                    gen_loss=0
                    disc_loss=0
                  
                    if epoch <single_step_limit:
                        #print('single_step',single_step_limit)
                        gen_out_4vec = self.generator(gen_in_4vec, training=True)

                        real_output = self.discriminator(truth_4vec, training=True)
                        fake_output = self.discriminator(gen_out_4vec, training=True)

                        gen_loss = generator_loss(fake_output) # Calculate Loss

                        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables) #Automatic differentiation?????

                        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))

                        disc_loss = discriminator_loss(real_output, fake_output) #Calculate Loss
                        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

                        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables)) 
                        return disc_loss,gen_loss
                            
                            
                    else:
                        #print('multi_step')    
                        for i in range(gen_train_num): #Train generator a certain number of times and get loss

                            gen_loss=gen_opt(gen_in_4vec, truth_4vec,i)

                        for j in range(dis_train_num):

                            disc_loss=dis_opt(gen_in_4vec, truth_4vec) #Train discriminator a certain number of times and get loss


                        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables) 
                        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))

                        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables) #Get Gradient
                        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables)) #Apply Gradient
                        return disc_loss, gen_loss

                #Old method if new breaks
                '''
                gen_out_4vec = self.generator(gen_in_4vec, training=True)

                real_output = self.discriminator(truth_4vec, training=True)
                fake_output = self.discriminator(gen_out_4vec, training=True)

                gen_loss = generator_loss(fake_output) # Calculate Loss

                gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables) #Automatic differentiation?????

                self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))

                disc_loss = discriminator_loss(real_output, fake_output) #Calculate Loss
                gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

                self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

                '''
                    

            best_wdis = 9999
            best_epoch = -1

            #Defining lists to store loss and wasserstein data for plots
            loss_all_epochs_0=[] #Disc Loss
            loss_all_epochs_1=[] #Gen Loss
            accuracy_list=[] #Should be called discriminator accuracy
            gen_accuracy=[]
            best_was_dist=[]

            with tqdm.trange(epochs, disable=self.disable_tqdm) as t0:
                for epoch in t0:  #performing each epoch

                    # compose the training dataset by generating different noises for each epochs
                    if noise_type=='gaussian':
                        noise = np.random.normal(loc=0., scale=1., size=(train_truth.shape[0], self.noise_dim)) 

                    elif noise_type=='uniform':
                        noise = np.random.uniform(-1, 1, size=(train_truth.shape[0], self.noise_dim)) 

                    #Contains the original output file and the noise for the training process
                    train_inputs = np.concatenate(
                        [train_in, noise], axis=1).astype(np.float32) if train_in is not None else noise

                    #Sanity Check
                    print(' ', train_inputs.shape[0],'train_in dataset size')
                    print(train_truth.shape[0],'train_truth dataset size')

                    #Create dataset from train inputs and train truth, shuffles it, and splits it into batches according to given                     batch size
                    dataset = tf.data.Dataset.from_tensor_slices(
                        (train_inputs, train_truth)).shuffle(2*batch_size).batch(batch_size, drop_remainder=True).prefetch(AUTO)


                    #Calculate statistics for the train dataset created above
                    tot_loss = []
                    for data_batch in dataset:

                        tot_loss.append(list(train_step(*data_batch,single_step_limit,epoch)))

                    tot_loss = np.array(tot_loss)
                    avg_loss = np.sum(tot_loss, axis=0)/tot_loss.shape[0]
                    loss_dict = dict(D_loss=avg_loss[0], G_loss=avg_loss[1])
                    #Append to list for end of run plots
                    loss_all_epochs_0.append(avg_loss[0])
                    loss_all_epochs_1.append(avg_loss[1])

                    gen_output_dim=args.gen_output_dim
                    #Calculate wasserstein difference and plot distrubutions
                    tot_wdis,accuracy_list,gen_accuracy = evaluate_samples_fn(
                        self.generator, epoch, testing_data,
                        summary_writer, img_dir,new_run_folder,loss_all_epochs_0,loss_all_epochs_1,self.discriminator,gen_accuracy,accuracy_list,gen_output_dim, **loss_dict)
                    
                    # Check if the current is the best epoch if it has the lowest wasserstein difference
                    if tot_wdis < best_wdis:
                        ckpt_manager.save()
                        self.generator.save("generator")
                        best_wdis = tot_wdis
                        best_epoch = epoch
                    print('Best Wasserstein Distance: ' ,best_wdis)
                    t0.set_postfix(**loss_dict, BestD=best_wdis, BestE=best_epoch)
                    #Append to list for end of run plots
                    best_was_dist.append(best_wdis)

             #Write text file listing all the parameters and best wasserstein difference
            tmp_res = "Best Model in {} Epoch with a Wasserstein distance {:.4f}".format(best_epoch, best_wdis)
            logging.info(tmp_res)
            summary_logfile = os.path.join(summary_dir, 'results.txt')
            summary_logfile = os.path.join(new_run_folder, 'results.txt')

            with open(summary_logfile, 'a') as f:
                f.write(tmp_res + "\n")
                f.write(' Epoch_num: '+ str(args.epochs)+' Batch_Size: '+str(batch_size) +' Number of Events: '+str(args.num_test_evts)+ "\n")
                f.write(' GAN Type: '+str(args.model) + ' Max Number of Events: ' +str(args.max_evts) + ' Learning Rate Discriminator: ' + str(args.lr_dis) + ' Learning Rate Generator: ' + str(args.lr_gen)+ "\n") 
                f.write(' Noise Type: ' + str(args.noise_type) + ' Number of Noise Dimensions: ' + str(args.noise_dim)) 
                f.write(' Number of Extra G Layers: ' + str(args.gen_layers)  + ' Number of Extra D Layers: ' + str(args.dis_layers)+ "\n")  
                f.write(' Fraction of testing events ' + str(args.test_frac)+ ' Number of Nodes per Layer: ' + str(args.num_nodes)+ "\n")    
                f.write(' Number of times training generator per epoch: ' + str(args.gen_train_num)+ ' Number of times training discriminator per epoch: ' + str(args.dis_train_num)+ "\n")
                f.write(' Number of epochs including single training steps until multiple training per epoch: ' + str(args.single_step_limit)+ "\n")
                
                        
            #Print log_loss etc graphs at end of run
            generate_and_save_images_end_of_run( epoch,img_dir,new_run_folder,loss_all_epochs_0,loss_all_epochs_1,accuracy_list,gen_accuracy,best_was_dist)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train The GAN')
    add_arg = parser.add_argument
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

    logging.set_verbosity(args.verbose)


    from gan4hep.utils_gan import generate_and_save_images
    from gan4hep.preprocess import herwig_angles

    train_in, train_truth, test_in, test_truth = herwig_angles(
        args.filename, max_evts=args.max_evts)

    batch_size = args.batch_size
    gan = GAN()
    gan.train(
        train_truth, args.epochs, batch_size,
        test_truth, args.log_dir,
        generate_and_save_images,
        train_in, test_in
    )


