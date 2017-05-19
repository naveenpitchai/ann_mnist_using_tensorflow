from DataPrep_helper import Data;
import pickle;
import os

class DataPrep(object):

    total_size = 480000

    def __init__(self,valid_percent=.20,test_percent = .02,force=False):

        self.valid_size = int(DataPrep.total_size * valid_percent);
        self.test_size = int(DataPrep.total_size * test_percent);
        self.train_size = (DataPrep.total_size - (self.valid_size + self.test_size))
        self.force = force;

    def create_master_pickle(self,filename = "Master_file.pickle"):

        if self.force or not os.path.isfile(filename):

            training_file_path = Data.download("notMNIST_large.tar.gz");
            testing_file_path = Data.download("notMNIST_small.tar.gz");

            #Extracting the file
            test_class_folders = Data.extract(testing_file_path);
            train_class_folders = Data.extract(training_file_path);

            #check the files
            print("Training and testing folder locations..\n")
            print(test_class_folders);
            print(train_class_folders);
            print('\n');

            #Read image files
            print('Starting to read images.. \n')
            test_pickle_names = Data.image_read(test_class_folders,pickleit=True);
            train_pickle_names = Data.image_read(train_class_folders,pickleit=True);

            print("\nPickled file folders..")
            print(test_pickle_names);
            print(train_pickle_names);

            print("\nStarting to Merge the classes in to a single Dataset")

            train_features,train_labels,valid_features,valid_labels = Data.create_datasets(train_pickle_names,self.train_size,self.valid_size)
            test_features,test_labels,_,_ = Data.create_datasets(test_pickle_names,self.test_size);

            print("\nMerging Done...\n")

            print("Size of All DataSets\n")
            print("Training Features:",train_features.shape);
            print("Training Labels:",train_labels.shape,'\n');

            print("Validation Features: ",valid_features.shape);
            print("Validation Labels: ",valid_labels.shape,'\n');

            print("Test Features: ",test_features.shape);
            print("Test Labels: ",test_labels.shape)

            #Check some sample images

            #Data.image_check(test_features,5);
            #Data.image_check(train_features,5);

            datasets = { "train_features": train_features,
                         "train_labels": train_labels,
                         "valid_features": valid_features,
                         "valid_labels": valid_labels,
                         "test_features":test_features,
                         "test_labels": test_labels}


            #pickle everything in to a one master file
            with open(filename,'wb') as writer:
                pickle.dump(datasets,writer,pickle.HIGHEST_PROTOCOL);

            print("{} file created".format(filename));

        else:
            print("{} file alredy exists".format(filename));


