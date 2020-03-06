import sys
import tensorflow as tf
import coloredlogs, logging
import os, ipdb
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re


coloredlogs.install()
anomaly_classification_feature_selection_folder='/home/user/baxter_ws/src/SPAI/smach_based_introspection_framework/introspection_data_folder.AC_offline_test/anomaly_classification_feature_selection_folder'


def get_anomalous_samples():
    logger = logging.getLogger()
    logger.info("load_csv_data_from_filtered_scheme")
    
    folders = glob.glob(os.path.join(
        anomaly_classification_feature_selection_folder,
        'No.* filtering scheme',
        'anomalies_grouped_by_type',
        'anomaly_type_(*)',
    )) 

    data_by_type = []
    types = []
    for folder in folders: # for each anomaly_type
        samples = []
        logger.info(folder)
        path_postfix = os.path.relpath(folder, anomaly_classification_feature_selection_folder).replace("anomalies_grouped_by_type"+os.sep, "")

        prog = re.compile(r'anomaly_type_\(?([^\(\)]+)\)?')
        anomaly_type = prog.search(path_postfix).group(1)
        csvs = glob.glob(os.path.join(folder, '*', '*.csv'))
        for j in csvs:
            df =  pd.read_csv(j, sep = ',')
            # delete the 1st column with is time index
            df = df.drop(['Unnamed: 0'], axis = 1)
            samples.append(df.values.T)
        data = np.stack(samples)
        logger.info('Successfully! load the data of %s'%anomaly_type)
        # np.save("./anomalies/"+anomaly_type+".npy", data)
        
        data_by_type.append(data)
        types.append(anomaly_type)
    return data_by_type, types

def feed_dict(batch_size, samples_type=None):
    if samples_type == 'train':
        indeces = np.random.randint(num_train_samples, size=batch_size)
        samples = train_samples
        
    elif samples_type == 'valid':
        indeces = np.random.randint(num_valid_samples, size=batch_size)
        samples = valid_samples
        
    elif samples_type == 'test':
        indeces = np.random.randint(num_test_samples, size=batch_size)
        samples = test_samples
        
    return np.take(samples, indeces, axis=0)


if __name__ == '__main__':

    x_in_dim = 6
    beta_1 = 0.05 # adam parameters
    beta_2 = 0.001 # adam parameters
    num_epochs = 3 # i.e. iterations 40000
    batch_size = 1
    num_hidden_units = 500
    learning_rate_1 = 2e-5
    learning_rate_2 = 1e-5
    num_epochs_to_diff_learn_rate = int(num_epochs/2) # change the learning rate after half of the epochs
    num_epochs_to_save_model = 1000 # save model after each 1000 iterations
    #decay_rate = .7

    z_dim = 20
    
    data_by_type, types = get_anomalous_samples()

    for i, samples in enumerate(data_by_type):
        anomaly_type = types[i]
        time_steps = samples.shape[1]
        train_samples = samples
        valid_samples = samples
        test_samples  = samples
        num_train_samples = train_samples.shape[0] 
        num_valid_samples = valid_samples.shape[0] 
        num_test_samples = test_samples.shape[0] 

        network_params =  ''.join([
                          'time_steps={}-'.format(time_steps),          
                          'latent_dim={}-'.format(z_dim),          
                          'dataset={}'.format(anomaly_type)])

        # Dir structure : /base_dir/network_params/run_xx/train_or_test/
        log_root = './anomalies_%s'%anomaly_type
        log_base_dir = os.path.join(log_root, network_params)
    
        # Check for previous runs
        if not os.path.isdir(log_base_dir):
            os.makedirs(log_base_dir)

        previous_runs = os.listdir(log_base_dir)

        if len(previous_runs) == 0:
            run_number = 1
        else:
            run_number = max([int(str.split(s,'run_')[1]) for s in previous_runs if 'run' in s]) + 1

        log_dir = os.path.join(log_base_dir,'run_{0:02d}'.format(run_number))

        train_summary_writer = tf.summary.FileWriter(log_dir + '/train')
        valid_summary_writer = tf.summary.FileWriter(log_dir + '/valid')
        test_summary_writer = tf.summary.FileWriter(log_dir + '/test')
        model_save_path = log_dir + '/models'
        figure_save_path = log_dir + '/figures'

        #############################
        # Setup graph
        #############################
        tf.reset_default_graph()
        X = tf.placeholder(tf.float32, shape=(batch_size, x_in_dim, time_steps))

        # time_slices containts input x at time t across batches.
        x_in = time_steps * [None]
        x_out = time_steps * [None]
        h_enc = time_steps * [None]
        h_dec = (time_steps + 1) * [None]

        for t in range(time_steps):
            x_in[t] = tf.squeeze(tf.slice(X,begin=[0,0,t],size=[-1,-1,1]),axis=2)


        ###### Encoder network ###########
        with tf.variable_scope('encoder_rnn'):
            cell_enc = tf.nn.rnn_cell.BasicRNNCell(num_hidden_units,activation=tf.nn.tanh)
            h_enc[0] = tf.zeros([batch_size,num_hidden_units], dtype=tf.float32) # Initial state is 0

            # h_t+1 = tanh(Wenc*h_t + Win*x_t+1 + b )
            #Most basic RNN: output = new_state = act(W * input + U * state + B).
            #https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/python/ops/rnn_cell_impl.py
            for t in range(time_steps-1):
                _ , h_enc[t+1] = cell_enc(inputs=x_in[t+1], state=h_enc[t])


        mu_enc = tf.layers.dense(h_enc[-1], z_dim, activation=None, name='mu_enc')
        log_sigma_enc = tf.layers.dense(h_enc[-1], z_dim, activation=None, name='log_sigma_enc')

        ###### Reparametrize ##############
        eps = tf.random_normal(tf.shape(log_sigma_enc))
        z = mu_enc + tf.exp(log_sigma_enc) * eps

        ##### Decoder network ############
        with tf.variable_scope('decoder_rnn'):
            W_out = tf.get_variable('W_out',shape=[num_hidden_units, x_in_dim])
            b_out = tf.get_variable('b_out',shape=[x_in_dim])

            cell_dec = tf.nn.rnn_cell.BasicRNNCell(num_hidden_units,activation=tf.nn.tanh)
            h_dec[0] = tf.layers.dense(z, num_hidden_units, activation=tf.nn.tanh)

            for t in range(time_steps):
                x_out[t] = tf.nn.sigmoid(tf.matmul(h_dec[t], W_out) + b_out)
                if t < time_steps - 1:
                    _, h_dec[t+1] = cell_dec(inputs=x_out[t], state=h_dec[t])

        ##### Loss #####################
        with tf.variable_scope('loss'):
            # Latent loss: -KL[q(z|x)|p(z)]
            with tf.variable_scope('latent_loss'):
                sigma_sq_enc = tf.square(tf.exp(log_sigma_enc))
                latent_loss = -.5 * tf.reduce_mean(tf.reduce_sum((1 + tf.log(1e-10 + sigma_sq_enc)) - tf.square(mu_enc) - sigma_sq_enc, axis=1),axis=0)
                latent_loss_summ = tf.summary.scalar('latent_loss',latent_loss)

            # Reconstruction Loss: log(p(x|z))    
            with tf.variable_scope('recon_loss'):    
                for i in range(time_steps):
                    if i == 0:
                        recon_loss_ = x_in[i] * tf.log(1e-10 + x_out[i]) + (1 - x_in[i]) * tf.log(1e-10+1-x_out[i])
                    else:
                        recon_loss_ += x_in[i] * tf.log(1e-10 + x_out[i]) + (1 - x_in[i]) * tf.log(1e-10+1-x_out[i])

                #collapse the loss, mean across a sample across all x_dim and time points, mean over batches
                recon_loss = -tf.reduce_mean(tf.reduce_mean(recon_loss_/(time_steps),axis=1),axis=0)


            recon_loss_summ = tf.summary.scalar('recon_loss', recon_loss)

            with tf.variable_scope('total_loss'):
                total_loss = latent_loss + recon_loss

            total_loss_summ = tf.summary.scalar('total_loss', total_loss)

        global_step = tf.Variable(0,name='global_step') 

        #learning_rate = tf.train.exponential_decay(initial_learning_rate, epoch_num, num_epochs, decay_rate, staircase=False)
        learning_rate = tf.Variable(learning_rate_1,name='learning_rate')
        train_step = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta_1, beta2=beta_2).minimize(total_loss,global_step=global_step)    
        scalar_summaries = tf.summary.merge([latent_loss_summ, recon_loss_summ, total_loss_summ])
        #image_summaries = tf.summary.merge()

        train_summary_writer.add_graph(tf.get_default_graph())


        #############################
        # Training/Logging
        #############################
        num_batches = int(num_train_samples/batch_size)
        global_step_op = tf.train.get_global_step()
        saver = tf.train.Saver()

        avg_loss_epoch = []
        avg_latent_loss_epoch = []
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(num_epochs):
                epoch_loss = 0.
                epoch_latent_loss = 0.
                for batch in range(num_batches):
                    batch_num = sess.run(global_step_op)

                    if epoch < num_epochs_to_diff_learn_rate:
                        curr_learning_rate = learning_rate_1
                    else:
                        curr_learning_rate = learning_rate_2

                    #_ , loss, scalar_train_summaries, x_out_, x_in_,learning_rate_,latent_loss_ = \
                    #sess.run([train_step, total_loss, scalar_summaries, x_out, x_in,learning_rate, latent_loss],feed_dict={X: feed_dict(batch_size,'train'), learning_rate: curr_learning_rate})

                    _ , loss, scalar_train_summaries, learning_rate_, latent_loss_ = \
                    sess.run([train_step, total_loss, scalar_summaries, learning_rate, latent_loss],feed_dict={X: feed_dict(batch_size,'train'), learning_rate: curr_learning_rate})

                    # Check for NaN
                    if np.isnan(loss):
                        sys.exit("Loss during training at epoch: {}".format(epoch))

                    epoch_loss += loss
                    epoch_latent_loss += latent_loss_

                print('Average loss epoch {0}: {1}'.format(epoch, epoch_loss/num_batches)) 
                print('Average latent loss epoch {0}: {1}'.format(epoch, epoch_latent_loss/num_batches)) 
                print('Learning Rate {}'.format(learning_rate_))
                print('')
                avg_loss_epoch.append(epoch_loss/num_batches)
                avg_latent_loss_epoch.append(epoch_latent_loss/num_batches)

                # Write train summaries once a epoch
                scalar_train_summaries = sess.run(scalar_summaries,feed_dict={X: feed_dict(batch_size,'train')})
                train_summary_writer.add_summary(scalar_train_summaries, global_step=batch_num)

                # Write validation summaries
                scalar_valid_summaries = sess.run(scalar_summaries,feed_dict={X: feed_dict(batch_size,'valid')})
                valid_summary_writer.add_summary(scalar_valid_summaries, global_step=batch_num)

                # Write test summaries
                scalar_test_summaries = sess.run(scalar_summaries,feed_dict={X: feed_dict(batch_size,'test')})
                test_summary_writer.add_summary(scalar_test_summaries, global_step=batch_num)

                # Save the models
                if epoch % num_epochs_to_save_model == 0:
                    save_path = saver.save(sess, model_save_path + '/epoch_{}.ckpt'.format(epoch))

        # plotting
        fig, ax = plt.subplots()
        ax.plot(avg_loss_epoch, label='avg_loss_epoch')
        ax.plot(avg_latent_loss_epoch, label='avg_latent_loss_epoch')
        plt.legend()

        if not os.path.isdir(figure_save_path):
            os.makedirs(figure_save_path)
        plt.savefig(figure_save_path +'/loss.eps', format='eps', dpi=300)
        plt.show()
