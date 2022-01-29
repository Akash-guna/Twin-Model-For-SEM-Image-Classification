import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
from data_loaders import data_loader,data_loader_multiple
from architecture import create_normal_model,create_lbp_model,create_multi_model
from Utilities import plot_history 
from options import options

d_opt = sys.argv[1]
print(d_opt)
SEM_PATH,LBP_PATH,batch_size,categories,train_len,val_len,normal_epochs,lbp_epochs,multi_epochs = options(d_opt)
print(SEM_PATH)
#loading the Normal Dataset(SEM Data)
train, val = data_loader(SEM_PATH,batch_size)
#creating the normal_model
normal_model = create_normal_model(categories)
#Compiling the normal_model
normal_model.compile(optimizer='adam',loss=tf.keras.losses.CategoricalCrossentropy(),metrics=['accuracy'])
folder=SEM_PATH.split('/')[-1]

try:
    os.makedirs(f'models/{folder}')
except:
    pass
#Defining Callbacks
callbacks=[tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=5,min_lr=0.001),
           tf.keras.callbacks.ModelCheckpoint(filepath=f'Models/{folder}/Normal_Model.hdf5', monitor='val_loss',verbose=1, save_best_only=True,)                 
          ]
print('Training Normal Model ...')
normal_history= normal_model.fit(train,epochs=normal_epochs,validation_data=val,callbacks=callbacks,verbose=1)
#plotting Loss and Accuracy
plot_history(normal_history,folder,"Normal")

#load saved normal model
normal_model= tf.keras.models.load_model(f'Models/{folder}/Normal_model.hdf5')

train,val = data_loader(LBP_PATH,batch_size)
lbp_model = create_lbp_model(normal_model,categories)
# LBP Model Compile + Train
lbp_model.compile(optimizer='adam',loss=tf.keras.losses.CategoricalCrossentropy(),metrics=['accuracy'])
callbacks=[tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=5,min_lr=0.001),
           tf.keras.callbacks.ModelCheckpoint(filepath=f'Models/{folder}/LBP_Model.hdf5', monitor='val_loss',verbose=1, save_best_only=True,)                 
          ]
print('Training LBP Model...')
lbp_history= lbp_model.fit(train,epochs=lbp_epochs,validation_data=val,callbacks=callbacks,verbose=1)

#plotting LBP Loss and Accuracy
plot_history(lbp_history,folder,"LBP")

#Loading Both Models For Training Multi Model(Proposed Model)
normal_model= tf.keras.models.load_model(f'Models/{folder}/Normal_model.hdf5')
lbp_model  = tf.keras.models.load_model(f'Models/{folder}/LBP_Model.hdf5')

# loading data from custom data generator for multi model training
train,val=data_loader_multiple(SEM_PATH,LBP_PATH,batch_size)
multi_model=create_multi_model(normal_model,lbp_model,categories)
steps = train_len / batch_size
val_steps = val_len / batch_size
# compiling multi model
multi_model.compile(loss="categorical_crossentropy",optimizer='adam',metrics=['accuracy'])

callbacks=[tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=5,min_lr=0.001),
           tf.keras.callbacks.ModelCheckpoint(filepath=f'Models/{folder}/Multi_Model.hdf5', monitor='val_loss',verbose=1, save_best_only=True,)                 
          ]
#Training Multi Model
print('Training Multi Model...')
multi_history =multi_model.fit_generator(train,epochs =multi_epochs,validation_data = val,steps_per_epoch=steps,validation_steps=val_steps,use_multiprocessing=False,shuffle=False,callbacks=callbacks)

#plotting Multi Model Loss and Accuracy
plot_history(multi_history,folder,"Multi")




