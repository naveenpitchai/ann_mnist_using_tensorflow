import numpy as np;
import requests;
import six.moves.urllib.request as request;
import cloudpickle as cpickle;
import sys,os;
import tarfile;
from scipy import ndimage;
#from PIL import Image;
import pickle;
from matplotlib import pyplot as plt;

class Data:
    URL = 'http://commondatastorage.googleapis.com/books1000/'
    #head = "/Users/naveenpitchai/PycharmProjects/ANN_MNIST/MetaData/";
    head = "./MetaData" #main folder where all the data to be stored
    #fullpath = "/Users/naveenpitchai/PycharmProjects/ANN_MNIST/MetaData/"
    img_size = 28;

    @staticmethod
    def report_hook(count,chunk_size,total_size):
        percent = int((count/(total_size/chunk_size)) * 100);
        total_count = int(total_size/chunk_size)

        if count%(total_count//10) == 0:
            sys.stdout.write('%s..'%(str(percent)+'%'));
            sys.stdout.flush();
        elif count%(total_count//5) == 0:
            sys.stdout.write('.');
            sys.stdout.flush();

    @staticmethod
    def download(filename,force=False):
        dest_filename = os.path.join(Data.head,filename);
        if not os.path.exists(Data.head):
            print("Creating {}".format(Data.head));
            os.makedirs(Data.head);

        if force or not os.path.exists(dest_filename):
            
            print('Data download starts...\n')
            sys.stdout.write('Percent complete - ')
            f_name, _ = request.urlretrieve(Data.URL+filename,dest_filename,reporthook=Data.report_hook);
            print('\nDownloaded to %s'%f_name)
            return f_name;
        else:
            print("File already exists in %s \n"%dest_filename)
            return dest_filename;

    @staticmethod
    def extract(filepath,force=False):
        os.chdir(Data.head+"/"); #Entering in to Metadata folder
        filename = os.path.split(filepath)[1];
        extracted_folder_name = filename.split('.')[0] #getting the resulting foldername to check for
        #the availability

        if not os.path.isdir(extracted_folder_name) and not force:
            #Extract the actual file..
            print('Extracting File...')
            tar = tarfile.open(filename);
            tar.extractall();
        else:
            print("Tar already extracted, so skipping....\n")

        #getting all the class folder paths inside the extracted file.
        class_folder_paths = [os.path.join(Data.head,extracted_folder_name,each) for each in
                              sorted(os.listdir(extracted_folder_name))
                              if os.path.isdir(os.path.join(extracted_folder_name,each))]

        os.chdir('..') #reset folder location
        return class_folder_paths;

    @staticmethod
    def image_read(class_folders,pickleit=False):
        list_of_class_arrays = []
        pickle_file_names = []
        for folder in class_folders: #for each class
            pickle_file_names.append(folder+".pickle")
            if not os.path.exists(folder+".pickle"):
                files = os.listdir(folder); #list all images in each class
                data_np = np.ndarray(shape=(len(files),Data.img_size,Data.img_size),dtype=np.float32)
                num = 0;
                for image in files: #reads each image file in each class folders
                    try:
                        #reading and normalizing the data between -1 to 1
                        data_np[num,:,:] = (ndimage.imread(os.path.join(folder,image)).astype(np.float32)-(255.0/2))/255.0;
                        num += 1;
                    except IOError:
                        print('Invalid Image File.. Skipping...')
                if pickleit:
                    with open(folder+".pickle","wb") as write:
                        pickle.dump(data_np,write,pickle.HIGHEST_PROTOCOL);
                else:
                    list_of_class_arrays.append(data_np);
                print("Pickling complete...");
            else:
                print('Pickle file already Exists. Skipping..')
        if pickleit:
            return pickle_file_names; #returns the pickle file names if pickling needed
        else:
            return list_of_class_arrays; #if no pickling requested just return list of each class arrays.

    #method to merge, shuffle all classes and create training, validation and test sets.
    @staticmethod
    def create_datasets(datafiles,size,valid_size=0):

        size_each_class = size // len(datafiles); #equal sizes for all classes

        #train/test features & labels
        t_features = np.ndarray(shape=(size,Data.img_size,Data.img_size),dtype=np.float32)
        t_labels = np.ndarray(shape=(size,1),dtype=np.float32);

        if valid_size > 0:
            #validation set parameters
            size_each_class_v = valid_size // len(datafiles); #for validation set
            start_v = 0;
            end_v = size_each_class_v;

            #create the validation arrays
            v_features = np.ndarray(shape=(valid_size,Data.img_size,Data.img_size),dtype=np.float32);
            v_labels = np.ndarray(shape=(valid_size,1),dtype=np.float32);
        else:
            v_features = None;
            v_labels = None;

        start_t = 0;
        end_t = size_each_class;

        label=0; #initiate the label sequence
        for each in datafiles:
            with open(each,'rb') as reader:
                each_class = pickle.load(reader);
                t_features[start_t:end_t,:,:] = each_class[0:size_each_class]; #load the records till the size in to main dataset
                t_labels[start_t:end_t,:] = label;

                if valid_size > 0:
                    v_features[start_v:end_v,:,:] = each_class[size_each_class:size_each_class+size_each_class_v];
                    v_labels[start_v:end_v,:] = label;

                    start_v += size_each_class_v;
                    end_v += size_each_class_v;

            start_t += size_each_class; #increase to add next class
            end_t += size_each_class; #increase to add next class
            label += 1;
        return t_features, t_labels, v_features,v_labels;

    #check for image consistency after normalizing it.
    @staticmethod
    def image_check(dataset,num):
        size = dataset.shape[0];
        for i in range(num):
            rand = int(np.random.random() * size)
            plt.imshow(dataset[rand,:,:])
            print("Close the image to proceed to the next one...")
            plt.show();















