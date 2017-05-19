import tensorflow as tf;
import pickle;
import numpy as np;
import time;
from DataPrep_main import DataPrep;
import tarfile;

#gets the training

class Helper(object):

    @staticmethod
    def get_datadict(filename = "Master_file.pickle"):
        with open(filename,"rb") as reader:
            datasetdict = pickle.load(reader);

        return datasetdict;

    @staticmethod
    def reshape_data(feature,label):
        feature_data = feature.reshape((-1,28*28)) #flatten each record
        label_data = (label == range(10)).astype(np.float32); #one-hot encoding
        perm = np.random.permutation(label.shape[0])
        return feature_data[perm,:],label_data[perm,:]

    @staticmethod
    def relu(x,deriv=False):
        x_shape = tf.shape(x);
        if deriv:
            return tf.where(tf.greater(x,0.),tf.ones_like(x),tf.zeros(x_shape,tf.float32))
        return tf.where(tf.greater(x,0.),x,tf.zeros(x_shape,tf.float32));

    @staticmethod
    def softmax_tf(x):
        return tf.div(tf.exp(x),(tf.reduce_sum(tf.exp(x),reduction_indices=1,keep_dims=True)));

    @staticmethod
    def compute_loss(y_prob,y_true):
        inside = tf.multiply(y_true,tf.log(y_prob));
        return -tf.reduce_mean(tf.reduce_sum(inside,axis=1));

    @staticmethod
    def accuracy(y_probs,y_true):
        bool_capture = tf.equal(tf.argmax(y_probs,1),tf.argmax(y_true,1));
        return tf.reduce_mean(tf.cast(bool_capture,tf.float32))

    @staticmethod
    def batch_gen(features,labels,b_size):
        length = features.shape[0];
        while True:
            labs = np.arange(length);
            np.random.shuffle(labs);
            for i in range(length//b_size):
                yield features[labs[i*b_size:(i+1)*b_size],:],labels[labs[i*b_size:(i+1)*b_size],:]


class ann_model(object):

    logdir = "default_tensorboard";

    def __init__(self,train_features,train_labels,valid_features,valid_labels):
        self.train_features = train_features;
        self.train_labels = train_labels;
        self.num_feat = train_features.shape[1];
        self.num_class = train_labels.shape[1];
        self.valid_features = valid_features;
        self.valid_labels = valid_labels;
        self.num_epochs = [20]
        self.hidden_layer = 100
        self.lambda_val = [1e-4];
        self.learning_rate = [1e-3]
        self.batch_size = [256]

    def modify_params(self,num_hidden,num_epochs,lr_list,lambda_list,batch_sizes,logdir):
        self.num_epochs = num_epochs;
        self.lr_list = lr_list;
        self.lambda_list = lambda_list;
        self.batch_sizes = batch_sizes;
        self.logdir = logdir;
        self.num_hidden = num_hidden

    def run(self):
        graph = tf.Graph();
        with graph.as_default():
            x_train = tf.placeholder(tf.float32,(None,self.num_feat));
            y_train = tf.placeholder(tf.float32,(None,self.num_class));

            #HyperParameters
            lrate = tf.placeholder(tf.float32);
            lambda_val = tf.placeholder(tf.float32);

            #Initiate weights and bias

            #hidden layer
            w1 = tf.Variable(tf.truncated_normal((self.num_feat,self.num_hidden)),tf.float32,name='W1');
            b1 = tf.Variable(tf.ones((1,self.num_hidden)),tf.float32,name='B1');

            #output layer
            w2 = tf.Variable(tf.truncated_normal((self.num_hidden,self.num_class)),tf.float32,name='W2');
            b2 = tf.Variable(tf.ones((1,self.num_class)),tf.float32,name='B2');

            #Front Propogation

            with tf.name_scope("Front_Prop"):
                #hidden layer
                a1 = tf.matmul(x_train,w1) + b1;
                z1 = Helper.relu(a1);

                #output layer
                a2 = tf.matmul(z1,w2) + b2;
                #out_probs = Helper.softmax_tf(a2);
                out_probs = tf.nn.softmax(a2);

            with tf.name_scope("Compute_loss"):
                loss = Helper.compute_loss(out_probs,y_train);
                accuracy = Helper.accuracy(out_probs,y_train);

            #Back Propogation

            with tf.name_scope("Back_prop"):

                #output layer
                l2_delta = out_probs - y_train #after simplification differentiting (l2_error * softmax(a2)

                #hidden layer
                l1_error = tf.matmul(l2_delta,tf.transpose(w2));
                l1_delta = tf.multiply(l1_error,Helper.relu(a1,deriv=True));

                clipped_delta_2 = tf.clip_by_value(tf.matmul(tf.transpose(z1),l2_delta),-10,10)
                clipped_delta_1 = tf.clip_by_value(tf.matmul(tf.transpose(x_train),l1_delta),-10,10)

                w2_reg = tf.multiply(lambda_val,w2);
                w1_reg = tf.multiply(lambda_val,w1)

                w2_delta = w2 - ((lrate * tf.matmul(tf.transpose(z1),l2_delta))+w2_reg);
                w1_delta = w1 - ((lrate * tf.matmul(tf.transpose(x_train),l1_delta))+w1_reg);

                #w2_delta = w2 - (lrate * clipped_delta_2);
                #w1_delta = w1 - (lrate * clipped_delta_1);

                b2_delta = b2 - (lrate * tf.reduce_sum(l2_delta,reduction_indices=0,keep_dims=True));
                b1_delta = b1 - (lrate * tf.reduce_sum(l1_delta,reduction_indices=0,keep_dims=True));

            with tf.name_scope("Assignment_stage"):
                w2_assign = w2.assign(w2_delta);
                w1_assign = w1.assign(w1_delta);
                b2_assign = b2.assign(b2_delta);
                b1_assign = b1.assign(b1_delta);

            tf.summary.scalar("Loss",loss);
            tf.summary.scalar("Accuracy",accuracy)


        metrics_dict = {}

        with tf.Session(graph=graph) as sess:

            t0 = time.time();
            tf.global_variables_initializer().run();
            feed_dict_valid = {x_train:self.valid_features, y_train:self.valid_labels}

            for each_epoch in self.num_epochs:
                for each_size in self.batch_sizes:
                    total_iters = each_epoch * (self.train_features.shape[0] // each_size)
                    batchgen = Helper.batch_gen(self.train_features,self.train_labels,each_size);
                    for each_lambda in self.lambda_list:
                        for each_lr in self.lr_list:
                            tf.global_variables_initializer().run(); #reset the weights for each combination of hyper params
                            print("***Run for lambda: {} & HiddenNodes: {}".format(each_lambda,self.num_hidden))
                            #TensorBoard initializers
                            merger = tf.summary.merge_all()
                            writer = tf.summary.FileWriter("{}/size_{}_lr_{}_l_{}_epochs_{}_hid_{}".
                                                           format(self.logdir,each_size,each_lr,each_lambda,
                                                                  each_epoch,self.num_hidden));
                            writer.add_graph(graph);

                            prev_loss = 100;

                            for i in range(total_iters):
                                batches = next(batchgen);
                                feed_dict = {x_train:batches[0],
                                             y_train:batches[1],
                                             lrate:each_lr,
                                             lambda_val: each_lambda}

                                if i%500 == 0:
                                    loss_v = round(loss.eval(feed_dict_valid),2);
                                    acc_v = round(100*accuracy.eval(feed_dict_valid),2)
                                    if loss_v < prev_loss:
                                        metrics_dict["{}_{}_{}".format(each_size,each_lambda,each_lr)] = \
                                            {"step":i,"valid_Loss":loss_v,"valid_acc":acc_v,
                                                                                    "W1":w1.eval(),"W2":w2.eval(),"B1":b1.eval(),
                                                                                    "B2":b2.eval()};
                                    prev_loss = loss_v;
                                    merger_s = sess.run(merger,feed_dict_valid);
                                    writer.add_summary(merger_s,i)

                                if i%8000 == 0:
                                    print("Training Loss: %.2f" %loss.eval(feed_dict));
                                    print("Validation Loss: {:.2f}".format(loss.eval(feed_dict_valid)))
                                    print("Training accuracy = {:.2f}".format(100 * accuracy.eval(feed_dict)))
                                    print("Validation accuracy = {:.2f}".format(100 * accuracy.eval(feed_dict_valid)))
                                    print('\n');

                                sess.run([w2_assign,w1_assign,b2_assign,b1_assign],feed_dict);

                            print(metrics_dict.keys());

                            time_taken = (time.time() - t0)/60
                            print("Time Taken for each combination: {:.2f}min\n".format(time_taken))

                            print("*"*60);

                time_taken = (time.time() - t0)/60
                print("Time Taken for each hidden_node try: {:.2f}min\n".format(time_taken))


            with open("tensorboard_3_hnodes-{}.pickle".format(self.num_hidden),'wb') as writer:
                pickle.dump(metrics_dict,writer,pickle.HIGHEST_PROTOCOL)

            return metrics_dict

class ann_model_test(object):

    graph_test = tf.Graph();

    def __init__(self,test_features,test_labels,metrics):
        with self.graph_test.as_default():
            self.x_test = tf.constant(test_features,tf.float32);
            self.y_label = tf.constant(test_labels,tf.float32);
            self.w1 = tf.constant(metrics["W1"],tf.float32);
            self.b1 = tf.constant(metrics["B1"],tf.float32);
            self.w2 = tf.constant(metrics["W2"],tf.float32);
            self.b2 = tf.constant(metrics["B2"],tf.float32);

            relu_output = Helper.relu(tf.matmul(self.x_test,self.w1) + self.b1);
            output_prob_test = Helper.softmax_tf(tf.matmul(relu_output,self.w2) + self.b2);
            self.accuracy = Helper.accuracy(output_prob_test,self.y_label);

    def run_test(self):
        with tf.Session(graph=self.graph_test) as sess_test:
            calculated_accuracy = self.accuracy.eval();
            print("Accuracy of the model with test set is {}".format(100 * calculated_accuracy));
            return calculated_accuracy;


if __name__ == "__main__":

    try:
        tar_obj = tarfile.open("Master_file.tar.gz")
        print("Tar file found.. Extracting..")
        tar_obj.extractall();

    except FileNotFoundError:
        #default sizes used, valid_percent=.20,test_percent = .02,
        #for any different file sizes, pls pass it in the class below with force=True
        dataprep = DataPrep();

        #CAUTION: invoking it with a different name will start the time consuming process of download, extract, and read each images if the master
        #file is not already present.
        #default file name is Master_file.pickle and present in the path
        dataprep.create_master_pickle();

    datasetdict = Helper.get_datadict();

    #reload & reshape the data
    train_feat,train_lab = Helper.reshape_data(datasetdict["train_features"],datasetdict["train_labels"]);
    valid_feat,valid_lab = Helper.reshape_data(datasetdict["valid_features"],datasetdict["valid_labels"]);
    test_feat, test_lab = Helper.reshape_data(datasetdict["test_features"], datasetdict["test_labels"]);

    #below values
    num_epochs = [20]
    hidden_layer = 100
    lambda_val = [1e-4];
    learning_rate = [1e-3]
    batch_size = [256]

    my_model = ann_model(train_feat,train_lab,valid_feat,valid_lab);
    my_model.modify_params(hidden_layer,num_epochs,learning_rate,lambda_val,batch_size,"class_tensorboard_1"); #invoke this to supply any parameters other than default
    metrics_dict = my_model.run();

    print("Running Test on the model..")

    ann_test = ann_model_test(test_feat,test_lab,list(metrics_dict.values())[0]);
    ann_test.run_test();