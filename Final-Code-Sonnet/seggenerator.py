import numpy as np
import os
import itertools
import pickle
from copy import deepcopy
from PIL import Image
from PIL import ImageOps
import base64

from keras.utils import to_categorical

LABEL_COUNT = 7
LABEL_BG = 0
LABEL_HAIR = 1
LABEL_FACE = 2
LABEL_UPPER_BODY = 3
LABEL_HAND = 4
LABEL_LOWER_BODY = 5
LABEL_FOOT = 6


COLOR_BG = (143, 0, 0)
COLOR_HAIR = (255, 32, 0)
COLOR_FACE = (255, 191, 0)
COLOR_UPPER_BODY = (159, 255, 96)
COLOR_HAND = (0, 80, 255)
COLOR_LOWER_BODY = (0, 255, 255)
COLOR_FOOT = (0, 0, 143)
COLOR_LEG = (0, 0, 175)


def getLabel(color):
    if color == COLOR_HAIR:
        return 1
    elif color == COLOR_FACE:
        return 2
    elif color == COLOR_UPPER_BODY:
        return 3
    elif color == COLOR_HAND:
        return 4
    elif color == COLOR_LOWER_BODY or color == COLOR_LEG:
        return 5
    elif color == COLOR_FOOT:
        return 6
    else:
        return 0

def getLabelColor(label):
    if label == LABEL_BG:
        return deepcopy(COLOR_BG)
    elif label == LABEL_HAIR:
        return deepcopy(COLOR_HAIR)
    elif label == LABEL_FACE:
        return deepcopy(COLOR_FACE)
    elif label == LABEL_UPPER_BODY:
        return deepcopy(COLOR_UPPER_BODY)
    elif label == LABEL_HAND:
        return deepcopy(COLOR_HAND)
    elif label == LABEL_LOWER_BODY:
        return deepcopy(COLOR_LOWER_BODY)
    elif label == LABEL_FOOT:
        return deepcopy(COLOR_FOOT)
    else:
        print ("WARNING: unknown label")
        return deepcopy(COLOR_BG)

def load_and_conv_image(fn, imgsize, flip=False, dtype=np.float32):
    image1 = Image.open(fn)
    image1 = image1.convert('RGB')
    image1 = image1.resize(imgsize)    
    image1 = np.asarray(image1, dtype)
    image1 /= 255
    return image1

class SegmentationGenerator(object):

    def __init__(self, dim_x = 64, dim_y = 80,  batch_size = 32, shuffle = True,
                 root='.',
                 data_path='',
                 flip=False,
                 imgsize=(64,80),
                 cache_dir=""):
        'Initialization'
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.flip = flip
        self._root = root
        self.imgsize = imgsize
        self._dtype = np.float32
        
        with open(data_path) as data_file:
            pairs = []
            for i, line in enumerate(data_file):
                pair = line.strip().split()
                if len(pair) == 1:
                    base_path, ext = os.path.splitext(pair[0])
                    mask_path = base_path + "_m.png"
                    pairs.append((pair[0], mask_path))
                elif len(pair) == 2:
                    pairs.append((pair[0], pair[1]))
                else:
                    raise ValueError(
                        'invalid format at line {} in file {}'.format(
                        i, data_path))
        self._pairs = pairs

        self._data = []
        for i in range(self.length()):
            cache_fn = os.path.join(cache_dir, str(i))
            if os.path.exists(cache_fn):
                example = pickle.load(open(cache_fn, "rb"))
            else:
                example = self.get_example(i)
                pickle.dump(example, open(cache_fn, "wb"))
            self._data.append(example)
            if (i % 100 == 0):
                print (str(i) + " / " + str(self.length()))

    def get_example(self, i):
        if self.flip and i >= len(self._pairs):
            path1, path2 = self._pairs[i - len(self._pairs)]
        else:
            path1, path2 = self._pairs[i]
            
        full_path1 = os.path.join(self._root, path1)
        flip1 = self.flip and i >= len(self._pairs)
        image1 = load_and_conv_image(full_path1, self.imgsize, flip1)
        
        full_path2 = os.path.join(self._root, path2)
        try:
            image2 = Image.open(full_path2)
        except IOError:
            image2 = Image.open(full_path1)
            
        image2 = image2.convert('RGB')
        image2 = image2.resize(self.imgsize, Image.NEAREST)

        if self.flip and i >= len(self._pairs):
            image2 = ImageOps.mirror(image2)

        #cv2.imshow('image', np.asarray(image2))
        #cv2.waitKey(0)
           
        w, h = image2.size        
        imageLabel = np.zeros((h, w), self._dtype, np.int32)
        for x, y in itertools.product(range(w), range(h)):
            (r, g, b) = image2.getpixel((x,y))
            label = getLabel((b, g, r))
            imageLabel[y, x] = label

        #image2= imageLabel.convert('RGB')
        #image1.save("image1.png")
        #image2.save("image2.png")


        #image1 = image1.transpose(2, 0, 1)
        image2 = np.asarray(imageLabel, np.int32)
        #image2 = image2[:,:,np.newaxis]
        #image2 = image2.transpose(2, 0, 1)

        #if self.mean:
        #    image1 -= self.mean


        return image1, image2

        
    def length(self):
        if self.flip:
            return len(self._pairs) * 2
        else:
            return len(self._pairs)        
        

    def generate(self):
        'Generates batches of samples'
        # Infinite loop
        while 1:
            # Generate order of exploration of dataset
            indexes = self.__get_exploration_order()
            
            # Generate batches
            imax = int(len(indexes)/self.batch_size)
            for i in range(imax):
                # Find list of IDs
                list_IDs_temp = [k for k in indexes[i*self.batch_size:(i+1)*self.batch_size]]
                
                # Generate data
                X, y = self.__data_generation(list_IDs_temp)
                yield X, y

    def __get_exploration_order(self):
        'Generates order of exploration'
        # Find exploration order
        indexes = np.arange(self.length())
        if self.shuffle == True:
            np.random.shuffle(indexes)
            
        return indexes

    def __data_generation(self,list_IDs_temp):
        'Generates data of batch_size samples' # X : (n_samples, v_size, v_size, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.dim_y, self.dim_x, 3))
        y = np.empty((self.batch_size, self.dim_y * self.dim_x, 7), dtype = int)
        
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store volume
            #img, mask = self.get_example(ID)
            img, mask = self._data[ID]
            X[i, :, :, :] = img
            
            # Store class
            flat = np.reshape(mask, (-1, 1))
            y[i] = to_categorical(flat, LABEL_COUNT)     
            
            
        return X, y  
