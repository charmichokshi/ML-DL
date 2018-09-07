from __future__ import print_function
import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf
from skimage.io import imsave,imread
from keras.utils import plot_model, to_categorical
import numpy as np
from keras.models import Model
from keras.layers import *
from keras.optimizers import SGD,Adam
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint,TensorBoard, Callback,History
from keras.layers.advanced_activations import LeakyReLU
from keras import backend as K
from keras.utils import plot_model
import keras
from keras import metrics
import sys
import pandas as pd
import json
import tifffile as tiff
from keras.layers import LeakyReLU, Softmax

saved_tones = []


class UniformNoise(Layer):
    """Apply additive uniform noise
    Only active at training time since it is a regularization layer.
    # Arguments
        minval: Minimum value of the uniform distribution
        maxval: Maximum value of the uniform distribution
    # Input shape
        Arbitrary.
    # Output shape
        Same as the input shape.
    """

    def __init__(self, std=0.5,**kwargs,):
        super(UniformNoise, self).__init__(**kwargs)
        self.supports_masking = True
        self.std = std

    def call(self, inputs, training=None):
        def noised():
            rtvl = (inputs * K.random_normal(shape=(1,1,1,int(inputs.shape[-1])), mean =1,stddev=self.std)) 
            return rtvl
        return K.in_train_phase(noised, inputs, training=training)

    def get_config(self):
        config = {'std': self.std}
        base_config = super(UniformNoise, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def tile_image(image,tile_size):
    tiles = []
    w,h,_ = image.shape
    for wsi in range(0,w-tile_size+1,int(tile_size)):
        for hsi in range(0,h-tile_size+1,int(tile_size)):
            img = image[wsi:wsi+tile_size,hsi:hsi+tile_size] 
            tiles.append(img)
    
    if h%tile_size != 0 :
        for wsi in range(0,w-tile_size+1,int(tile_size)):    
            img = image[wsi:wsi+tile_size,h-tile_size:]
            
            tiles.append(img)

    if w%tile_size !=0 :
        for hsi in range(0,h-tile_size+1,int(tile_size)):
            img = image[w-tile_size:,hsi:hsi+tile_size]
            tiles.append(img)
    
    if w%tile_size !=0 and h%tile_size != 0:
        img = image[w-tile_size:,h-tile_size:]
        tiles.append(img)

    return tiles

def rotate(inputs,rotation):
    assert (rotation==90 or rotation==180 or rotation==270),"rotation value not supported!"
    rot90=tf.transpose(inputs[:,::-1,:,:],perm=[0,2,1,3])
    rot180=tf.transpose(rot90[:,::-1,:,:],perm=[0,2,1,3])
    rot270=tf.transpose(rot180[:,::-1,:,:],perm=[0,2,1,3])
    return rot90 if rotation==90 else (rot180 if rotation==180 else rot270)

    
def SepRotConv2D(inputs,suppressRotations=False,**kwargs):
    sharedConv = SeparableConv2D(**kwargs)
    if suppressRotations:
        return sharedConv(inputs)
    inputs_90,inputs_180,inputs_270 = Lambda(rotate,arguments={'rotation':90})(inputs),Lambda(rotate,arguments={'rotation':180})(inputs),Lambda(rotate,arguments={'rotation':270})(inputs)
    shared_conv_0, shared_conv_90, shared_conv_180, shared_conv_270 = sharedConv(inputs), sharedConv(inputs_90), sharedConv(inputs_180), sharedConv(inputs_270)
    shared_conv_90, shared_conv_180, shared_conv_270 = Lambda(rotate,arguments={'rotation':270})(shared_conv_90),Lambda(rotate,arguments={'rotation':180})(shared_conv_180),Lambda(rotate,arguments={'rotation':90})(shared_conv_270)
    shared_conv_max=Maximum()([shared_conv_0, shared_conv_90, shared_conv_180, shared_conv_270])
    return shared_conv_max




def encoder(input,num_channels,dropout=0.8,pool_size=2):
    conv1 = Conv2D(num_channels, (3, 3), padding='same',activation='elu')(input)
    conv1 = BatchNormalization()(conv1)
    conv1 = SpatialDropout2D(dropout)(conv1)
    conv1 = Conv2D(num_channels, (3, 3), padding='same',activation='elu')(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = SpatialDropout2D(dropout)(conv1)
    conv1 = Conv2D(num_channels, (3, 3), padding='same',activation='elu')(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = UniformNoise(std=dropout*2)(conv1)
    conv1 = SpatialDropout2D(dropout)(conv1)
    if pool_size>1:
        conv1 = MaxPooling2D(2)(conv1)
    
    return conv1

def get_model(input_matrix):
    conv1m  = encoder(input_matrix,64,pool_size=1,dropout=0.2)
    conv2m  = encoder(conv1m,128,dropout=0.3)
    conv3m  = encoder(conv2m,128,dropout=0.4)
    conv4m  = encoder(conv3m,128,dropout=0.5)
    conv5m  = encoder(conv4m,128,dropout=0.5)
    conv5u = UpSampling2D(16)(conv5m)
    conv4u = UpSampling2D(8)(conv4m)
    conv3u = UpSampling2D(4)(conv3m)
    conv2u = UpSampling2D(2)(conv2m)
    conv_concatenate = concatenate([UpSampling2D(256)((Lambda(lambda x: K.stop_gradient(x))(AveragePooling2D(256)(conv1m)))),conv5u, conv2u, conv3u, conv4u,conv1m], axis = -1)
    full = Conv2D(128*5, (1, 1), padding='same',activation='elu')(conv_concatenate)
    full = BatchNormalization()(full)
    full = Dropout(0.5)(full)

    #full = Dropout(0.5)(full)
    full = Conv2D(256, (1, 1), padding='same',activation='elu')(full)
    full = BatchNormalization()(full)
    full = Dropout(0.5)(full)
    
    full = Conv2D(2, (1, 1), padding='same',activation='elu')(full)
    return full

	# conv1  = encoder(input_matrix,128,dropout=0.2)
	# conv1m = MaxPooling2D(2)(conv1)

	# conv2  = encoder(conv1m,64,dropout=0.2)
	# conv2m = MaxPooling2D(2)(conv1)

    # conv2 = Conv2D(256, (3, 3), padding='same')(conv1)
    # conv2 = LeakyReLU(alpha=0.3)(conv2)
    # conv2 = BatchNormalization()(conv2)
    # conv2 = Dropout(0.5)(conv2)
    # conv2 = Conv2D(128, (3, 3), padding='same')(conv2)
    # conv2 = LeakyReLU(alpha=0.3)(conv2)
    # conv2 = BatchNormalization()(conv2)
    # conv2 = Dropout(0.5)(conv2)
    
    # conv3 = Conv2D(2, (3, 3), padding='same')(conv2)
    # conv3 = LeakyReLU(alpha=0.3)(conv3)
    # conv3 = BatchNormalization()(conv3)

    


  
def load_image(image_folder_path,tile_size,isLabel=False):
    
    images = [tiff.imread(image_folder_path+f) for f in sorted(os.listdir(image_folder_path))]
    print(images[0].shape)

    tiles = []

    for image in images:
        tiles.extend(tile_image(image,tile_size))
        
    tiles = np.array(tiles)
    print(tiles.shape,tiles.dtype)
    return tiles

def tone_to_integer(images,tones):
    cls_idxs = np.zeros(images.shape[:-1])
    for i in range(1,len(tones)+1):
        print("Encoding tone",i,tones[i-1])
        cls_idxs[np.all(images==tones[i-1],axis=-1)]=i
    return cls_idxs.astype(np.uint8)

def integer_to_onehot(integer_maps,saved_tones_length):

    return np.stack([integer_maps==integer for 
    integer in range(1,saved_tones_length+1)],axis=-1).astype(np.uint8)

def onehot_preds_to_integer(one_hot_preds):
    return np.argmax(one_hot_preds,axis=-1)+1

def integer_to_tone(integer_maps,tones):
    images_shape = list(integer_maps.shape)
    images_shape.append(len(tones[0]))
    cls_idxs = np.zeros(tuple(images_shape))
    for i in range(1,len(tones)+1):
        cls_idxs[integer_maps==i]=tones[i-1]
    return cls_idxs.astype(np.uint8)

def one_hot_preds_to_color(one_hot_preds):
    return integer_to_tone(onehot_preds_to_integer(one_hot_preds),saved_tones)

def get_tones(images):
    print("Preparing tone list")
    if(os.path.exists('tones.json')):
        with open('tones.json','r') as fp:
            rettones=json.load(fp)
    else:
        with open('tones.json','w') as fp:
            del(rettones[1])
            json.dump(saved_tones,fp)

        rettones =  np.unique(images.reshape(-1,images.shape[-1]),axis=0).tolist()
    
    print(rettones)
    return rettones

def color_to_onehot(images):
    global saved_tones 
    if len(saved_tones)==0:
        saved_tones = get_tones(images)

    print("Number of tones calculated",len(saved_tones))
    return integer_to_onehot(tone_to_integer(images,saved_tones),len(saved_tones))


def get_training_data(tile_size=256):
    return load_image('input/',tile_size),load_image('label/',tile_size,True)





class OutputObserver(Callback):

    def __init__(self,x_val,v_labels_u):
        self.x_val=x_val
        self.v_labels_u = v_labels_u
        self.json_log = open('test_perf/loss_log.json', mode='a+', buffering=1)
            
    def on_epoch_end(self,epoch,logs={}):
        predictions =  self.model.predict(self.x_val,batch_size=1,verbose=True)
        os.mkdir('test_perf/val_images/'+str(epoch+1))
        color_images = one_hot_preds_to_color(predictions)
        color_image_array = list(color_images)
        label_image_array = list(self.v_labels_u)
        feature_image_array = list(self.x_val)


        for j in range(len(color_image_array)):
            imsave('test_perf/val_images/'+str(epoch+1)+'/'+str(j)+'_label.png',color_image_array[j])
        
        for j in range(len(color_image_array)):
            imsave('test_perf/val_images/'+str(epoch+1)+'/'+str(j)+'_actual.png',label_image_array[j].astype(np.uint8))
        
        for j in range(len(color_image_array)):
            imsave('test_perf/val_images/'+str(epoch+1)+'/'+str(j)+'_input.png',feature_image_array[j][...,0:3].astype(np.uint8))
        
        self.json_log = open('test_perf/loss_log.json', mode='a+')
        logs['epoch'] = epoch
        self.json_log.write(json.dumps(logs)+ '\n')
        self.json_log.flush()
    
    def on_train_end(self,logs={}):
        self.json_log.close() 




def dice_coef(y_true,y_pred,smooth=1):
    y_true_f = K.flatten(y_true[...,1])
    y_pred_f = K.flatten(y_pred[...,1])
    intersection = K.sum(y_true_f*y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true,y_pred):
    return -dice_coef(y_true,y_pred)


def train():

    print("here")
    img_rows = img_cols = 256
    val_split=0.2
    K.set_image_data_format('channels_last')

    train_features,train_labels = get_training_data(tile_size=img_rows)
    split=int((1-val_split)*train_features.shape[0])
    v_labels_u = train_labels[split:,...].copy()
    print("Training data loaded.") 
    print("Preprocessing started.")
    train_labels = color_to_onehot(train_labels)
    print("Preprocessing completed.")
    t_features = train_features[:split,...]
    t_labels = train_labels[:split,...]
    v_features = train_features[split:,...]
    v_labels = train_labels[split:,...]




    # print("here")
    # img_rows = img_cols = 256
    # val_split=0.2
    # K.set_image_data_format('channels_last')

    # train_features,train_labels = get_training_data(tile_size=img_rows)
    # split=int((1-val_split)*train_features.shape[0])
    # """
    # v_labels_u = train_labels[split:,...].copy()
    # print("Training data loaded.") 
    # print("Preprocessing started.")
    # train_labels = color_to_onehot(train_labels)
    # print("Preprocessing completed.")
    # t_features = train_features[:split,...]
    # t_labels = train_labels[:split,...]
    # v_features = train_features[split:,...]
    # v_labels = train_labels[split:,...]
    # """

    # v_labels_u = train_labels[::,...].copy()
    # print("Training data loaded.") 
    # print("Preprocessing started.")
    # train_labels = color_to_onehot(train_labels)
    # print("Preprocessing completed.")
    # t_features = train_features[:split,...]
    # t_labels = train_labels[:split,...]
    # v_features = train_features[split:,...]
    # v_labels = train_labels[split:,...]

    inputs = Input((256,256,3))

    inputsn = UniformNoise(std=0.4)(inputs)
    inputsn = inputs
    #src1 = SepRotConv2D(inputs, filters=32, kernel_size=(3,3), padding='same')
    #src2 = SepRotConv2D(inputs, filters=32, kernel_size=(3,3), padding='same')
    #src3 = SepRotConv2D(inputs, filters=32, kernel_size=(3,3), padding='same')
    #src4 = SepRotConv2D(inputs, filters=32, kernel_size=(3,3), padding='same')

    model1 = get_model(inputsn)
    # print("summary of model 1")
    #model2 = get_model(src2)
    #model3 = get_model(src3)
    #model4 = get_model(src4)

    add_model = model1 
    #add([model1, model2])
    final_model = Softmax(axis=-1)(add_model)
    
    initial_epoch=0

    model = Model(inputs, final_model)
    print(model.summary())
    #plot_model(model, to_file = 'model/model.png')
    # old optimizer SGD(lr = 1e-3, momentum = 0.9, nesterov = True)
    
    model.compile(optimizer = SGD(lr = 1e-3, momentum = 0.9, nesterov = True), loss = 'categorical_crossentropy', metrics = ['categorical_accuracy']) 
    model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_categorical_accuracy',verbose=True, save_best_only=True)
    
    #OutputObserver(x_val=v_features,v_labels_u=v_labels_u),
    #model.fit(t_features, t_labels,batch_size=1, epochs=200, verbose=1, shuffle=True,
    #          callbacks=[model_checkpoint,keras.callbacks.ReduceLROnPlateau(monitor='val_categorical_accuracy',factor=0.1,patience=5,verbose=1)],validation_data=(v_features,v_labels),initial_epoch=initial_epoch)
    '''
    model.compile(optimizer = Adam(lr=1e-3), loss = dice_coef_loss, metrics = [dice_coef]) 
    model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_dice_coef',verbose=True, save_best_only=True)
   '''
    # model.compile(optimizer = SGD(lr = 1e-3, momentum = 0.8, nesterov = True), loss = 'categorical_crossentropy', metrics = ['categorical_accuracy']) 
    # model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_categorical_accuracy',verbose=True, save_best_only=True)




    model.fit(t_features, t_labels,batch_size=1, epochs=200, verbose=1, shuffle=True,
               callbacks=[model_checkpoint,OutputObserver(x_val=v_features,v_labels_u=v_labels_u)],validation_data=(v_features,v_labels),initial_epoch=initial_epoch)


train()