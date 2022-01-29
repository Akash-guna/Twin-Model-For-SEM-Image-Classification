import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
def data_loader(path ,batch_size =16):
    '''
    Inputs : 
        path : Path to Dataset Folder
        batch_size : Batch size
    Outputs:
        train : Train Generator
        val  : Validation Generator
    Use: Loads the Data For Normal / LBP Model
    
    '''
    gen= ImageDataGenerator(rescale=1/255.,horizontal_flip=True,vertical_flip=True,validation_split=0.1)
    train= gen.flow_from_directory(
        path,
        target_size=(256, 256),
        class_mode="categorical",
        batch_size=batch_size,
        shuffle=True,
        subset='training',
    )

    val= gen.flow_from_directory(
        path,
        target_size=(256, 256),
        class_mode="categorical",
        batch_size=16,
        shuffle=True,
        subset='validation',
    )
    return train,val

def generate_generator_multiple(generator,dir1, dir2, batch_size, img_height,img_width,sub):
    
    '''
    Input:
        generator  ---> Keras Generator Object
        dir1       ---> SEM Data Directory
        dir2       ---> LBP Transformed SEM Data Directory
        batch_size ---> Batch size for the generator
        img_height ---> Number of rows in an image
        img_width  ---> Number of columns in an image
        sub        ---> The Subset of the generator (Training / Validation)
    Output:
        The Function acts like Keras Generator Therefore yeilds batches of data when called

    Image data generators only generate (x,y) pairs .For a two input model ([x1,x2],y) when we give input in form of generators 
    so we are making a custom image generator which produces the output tuple of form ([x1,x2],y)
    The normal images , lbp images dataset are ordered in a way that when we generate ([x1,x2],y) we get the correct tuple.
    shuffle is true and seed is set so the order of the images , its pairs is maintained'''
    # load normal images from directory
    genX1 = generator.flow_from_directory(dir1,
                                          target_size = (img_height,img_width),
                                          class_mode = 'categorical',
                                          batch_size = batch_size,
                                          shuffle=True, 
                                          subset=sub,
                                          seed=7)
    #load lbp images from directory
    genX2 = generator.flow_from_directory(dir2,
                                          target_size = (img_height,img_width),
                                          class_mode = 'categorical',
                                          batch_size = batch_size,
                                          shuffle=True, 
                                          subset=sub,
                                          seed=7)
    #custom Image Generator
    while True:
            X1i = genX1.next()
            X2i = genX2.next()
            if X1i[0].shape[0]==batch_size:
                yield [X1i[0], X2i[0]], X2i[1] 
            else:
                X1i = genX1.next()
                X2i = genX2.next()
                yield [X1i[0], X2i[0]], X2i[1] 

def data_loader_multiple(train_dir_1='Dataset/SEM_DATA',train_dir_2="Dataset/SEM_LBP",batch_size=16):
    '''
    Inputs :
            train_dir_1 --> The Path to SEM Data
            train_dir_2 --> The Path to LBP Transformed SEM Data
            batch_size  --> Batch Size for the generator
    Outputs :
            traingenerator --> The generator which is used for training
            valgenerator  --> The generator which is used for validation

    This function calls the generate_generator_multiple function and is responsible for creating a train and validation generator.
     
    '''
    # train_dir_1=r"SEM_DATA"      
    # train_dir_2=r"SEM_LBP_NEW"
    # batch_size=32
    input_imgen = ImageDataGenerator(rescale = 1./255,validation_split=0.1)
    img_height=256            
    traingenerator=generate_generator_multiple(generator=input_imgen,
                                            dir1=train_dir_1,
                                            dir2=train_dir_2,
                                            batch_size=batch_size,
                                            img_height=img_height,
                                            img_width=img_height,sub="training")     

    valgenerator=generate_generator_multiple(generator=input_imgen,
                                            dir1=train_dir_1,
                                            dir2=train_dir_2,
                                            batch_size=batch_size,
                                            img_height=img_height,
                                            img_width=img_height,sub="validation")
    return traingenerator, valgenerator
