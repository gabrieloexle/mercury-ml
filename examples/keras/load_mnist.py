
from PIL import Image
import numpy as np
from tensorflow.python.keras.utils.data_utils import get_file
import os
import random
import sys

def download_and_convert_mnist(data_path):
    image_dir = train_dir = os.path.join(os.getcwd(),"example_data",data_path)

    train_dir = os.path.join(image_dir,"train")
    test_dir = os.path.join(image_dir,"test")
    valid_dir = os.path.join(image_dir,"valid")

    link = 'http://storage.googleapis.com/tensorflow/tf-keras-datasets/'
    file='mnist.npz'
    path = get_file(file, origin=link +file )
    with np.load(path) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']

    train = list(zip(x_train,y_train))
    test_and_valid = list( zip(x_test,y_test))
    random.shuffle(test_and_valid)
    val = test_and_valid[:len(test_and_valid)//2]
    test = test_and_valid[len(test_and_valid)//2:]


    #creating directries for images
    for i in range(0,10):
        for path in [train_dir,test_dir,valid_dir]:
            direc = os.path.join(path,str(i))
            if not os.path.exists(direc):
                os.makedirs(direc)

    print("generating png images. This will take a while...")
    #saving images to disk
    for (images,path) in [(train,train_dir), (test, test_dir), (val,valid_dir)]:
        for (i,(x,y)) in enumerate(images, start=1):

            im = Image.fromarray(x)
            #im.show()
            im.save(os.path.join(path,str(y),str(i)+'.png'))
            if i%1000 == 0:
                sys.stdout.write('.')
        print("")
        print(str(i)+ " number of images saved to "+path)





