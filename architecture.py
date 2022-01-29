import tensorflow as tf
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.layers import GlobalMaxPooling2D , Dense, Dropout , Concatenate
from tensorflow.keras import Model

def create_normal_model(categories):
    '''
    Input: 
        Categories -> Total Number of categories in the dataset.

    Output:
        model  -> A Keras Model which contains Inception V3 as its base.

    Use : Used to to Define the Normal Model.

    '''
    inception = InceptionResNetV2(  include_top=False,
                              weights="imagenet",
                              input_shape=(256,256,3),
                              pooling=None,
                            )                            
    mixed=inception.get_layer('conv_7b_ac').output
    global_pool= GlobalMaxPooling2D()(mixed)
    df0=Dense(512 , activation='relu')(global_pool)
    df1= Dense(256 , activation='relu')(df0)
    df2= Dense(128, activation='relu')(df1)
    drop=Dropout(0.3)(df2)
    df3= Dense(64, activation='relu')(drop)
    df4= Dense(32, activation='relu')(df3)
    output=Dense(categories,activation='softmax')(df4)
    model= Model(inputs=inception.input,outputs=output)
    return model

def create_lbp_model(normal_model,categories):
    '''
    Input: 
        Categories -> Total Number of categories in the dataset.
        normal_model -> A Trained Normal Model.

    Output:
        model  -> A Keras Model which containing the Normal Model as its Base.

    Use : Used to to Define the LBP Model. 

    '''
    mixed=normal_model.get_layer('conv_7b_ac').output
    global_pool= GlobalMaxPooling2D()(mixed)
    df0=Dense(512 , activation='relu')(global_pool)
    df1=Dense(256 , activation='relu')(df0)
    df2=Dense(128, activation='relu')(df1)
    df3=Dense(64, activation='relu')(df2)
    df4=Dense(32, activation='relu')(df3)
    output=Dense(categories,activation='softmax')(df4)

    model= tf.keras.Model(inputs=normal_model.input,outputs=output)
    return model

def create_multi_model(normal_model,lbp_model,categories):
    '''
    Input: 
        Categories -> Total Number of categories in the dataset.
        normal_model -> A Trained Normal Model.
        lbp_model  -> Traned LBP Model

    Output:
        model  -> Proposed Model's Architecture on concatenation of Normal Model and LBP Model

    Use : Used to to Define the Normal Model. 

    '''
    normal_model.trainable=False
    lbp_model.trainable=False

    #renaming lbp model layers to prevent confusion of compiler
    for layer in lbp_model.layers:
        layer._name = layer.name + str("_lbp")
    
    #extracting final layer before dense layers from normal and lbp models
    active_normal=normal_model.get_layer("conv_7b_ac").output
    active_lbp=lbp_model.get_layer("conv_7b_ac_lbp").output

    #concatenating the two networks
    concatted = Concatenate()([active_normal, active_lbp])
    #global max pooling layers followed by 5 dense layers
    gmp=GlobalMaxPooling2D()(concatted)
    df0=Dense(512 , activation='relu')(gmp)
    df1= Dense(256 , activation='relu')(df0)
    df2= Dense(128, activation='relu')(df1)
    drop=Dropout(0.3)(df2)
    df3=Dense(64, activation='relu')(drop)
    df4=Dense(32, activation='relu')(df3)
    output=Dense(categories,activation='softmax')(df4)

    two_input_model=tf.keras.Model([normal_model.input,lbp_model.input],output)

    return two_input_model


    