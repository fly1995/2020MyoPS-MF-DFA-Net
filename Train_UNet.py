from __future__ import division, print_function
from keras.layers import Add,Multiply,pooling,Conv2D, Activation, Concatenate, concatenate, MaxPooling2D, ZeroPadding2D,Conv2DTranspose, Cropping2D, average, Input,normalization
from keras.optimizers import Adam,SGD
from keras import Model
from keras.layers import Dropout, Lambda
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, ReduceLROnPlateau
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
import numpy as np
from keras import backend as K
from  Loss_function import *
from keras.utils import plot_model
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'gpu': 0})))
K.set_image_data_format('channels_last')
flt = 44

def dial_multi_conv(flt, input):
    conv1 = Conv2D(flt, (3, 3),  dilation_rate=1, padding='same')(input)
    conv1 = normalization.BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv2 = Conv2D(flt, (3, 3),  dilation_rate=2, padding='same')(input)
    conv2 = normalization.BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    conv3 = Conv2D(flt, (3, 3),  dilation_rate=4, padding='same')(input)
    conv3 = normalization.BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    concat = concatenate([conv1,conv2,conv3],axis=3)
    concat = Conv2D(flt, (3,3),padding='same')(concat)
    concat = normalization.BatchNormalization()(concat)
    concat = Activation('relu')(concat)
    return concat

def IB(input,flt):
    conv1 = Conv2D(flt, (1, 1), activation='relu', padding='same')(input)
    conv2 = Conv2D(flt*2, (3, 3), activation='relu', padding='same')(input)
    conv2 = Conv2D(flt, (1,1), activation='relu', padding='same')(conv2)
    conv3 = Conv2D(flt*2, (5, 5), activation='relu', padding='same')(input)
    conv3 = Conv2D(flt, (1, 1), activation='relu', padding='same')(conv3)
    concate = concatenate([conv1, conv2, conv3], axis=3)
    conv = Conv2D(flt, (1, 1), activation='relu')(concate)
    output = conv
    return output

def conv_bn_relu(flt, input):
    kwargs = dict(kernel_size=(3, 3), strides=1, padding='same', kernel_initializer='he_normal')
    conv1 = Conv2D(flt, **kwargs)(input)
    conv1 = normalization.BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    return conv1

def path1(x = Input(shape=(256, 256, 1)), features=16, depth=4):
    inputs = x
    maps = [inputs]
    ib = IB(inputs, features)
    x = Conv2D(features, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(ib)
    for n in range(depth):
        x = Conv2D(features, (3, 3),  strides=1, dilation_rate=2, padding='same',kernel_initializer='he_normal')(x)
        maps.append(x)
        x = Concatenate(axis=3)(maps)
        x = normalization.BatchNormalization()(x)
        x = Activation('relu')(x)
    x1 = Conv2D(features, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(x)
    return x1

def Cardiac_Seg1(x = Input(shape=(256, 256, 1))):
    inputs = x
    global conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8, conv9, conv10, conv11, conv12, conv13, conv14, conv15, pool1, pool2, pool3, pool4
    conv1 = conv_bn_relu(flt, inputs)
    conv2 = conv_bn_relu(flt, conv1)
    pool1 = MaxPooling2D(pool_size=2, strides=2, padding='same')(conv2)  # 128
    conv4 = conv_bn_relu(flt * 2, pool1)
    conv5 = conv_bn_relu(flt * 2, conv4)
    pool2 = MaxPooling2D(pool_size=2, strides=2, padding='same')(conv5)  # 64
    conv7 = conv_bn_relu(flt * 4, pool2)
    conv8 = conv_bn_relu(flt * 4, conv7)
    pool3 = MaxPooling2D(pool_size=2, strides=2, padding='same')(conv8)  # 32
    conv10 = conv_bn_relu(flt * 8, pool3)
    conv11 = conv_bn_relu(flt * 8, conv10)
    pool4 = MaxPooling2D(pool_size=2, strides=2, padding='same')(conv11)  # 16
    conv13 = conv_bn_relu(flt * 16, pool4)
    conv14 = conv_bn_relu(flt * 16, conv13)
    conv14 = Dropout(0.5)(conv14)
    return conv14

def Cardiac_Seg2(x = Input(shape=(256, 256, 1))):
    inputs = x
    global co12, co22, co32, co42, co52, co62, co72, co82, co92, co102, co112, co122, co132, co142, co152, po12, po22, po32, po42
    co12 = conv_bn_relu(flt, inputs)
    co22 = conv_bn_relu(flt, co12)
    po12 = MaxPooling2D(pool_size=2, strides=2, padding='same')(co22)  # 128
    co42 = conv_bn_relu(flt * 2, po12)
    co52 = conv_bn_relu(flt * 2, co42)
    po22 = MaxPooling2D(pool_size=2, strides=2, padding='same')(co52)  # 64
    co72 = conv_bn_relu(flt * 4, po22)
    co82 = conv_bn_relu(flt * 4, co72)
    po32 = MaxPooling2D(pool_size=2, strides=2, padding='same')(co82)  # 32
    co102 = conv_bn_relu(flt * 8, po32)
    co112 = conv_bn_relu(flt * 8, co102)
    po42 = MaxPooling2D(pool_size=2, strides=2, padding='same')(co112)  # 16
    co132 = conv_bn_relu(flt * 16, po42)
    co142 = conv_bn_relu(flt * 16, co132)
    co142 = Dropout(0.5)(co142)
    return co142

def Cardiac_Seg3(x = Input(shape=(256, 256, 1))):
    inputs = x
    global con13, con23, con33, con43, con53, con63, con73, con83, con93, con103, con113, con123, con133, con143, con153, poo13, poo23, poo33, poo43
    con13 = conv_bn_relu(flt, inputs)
    con23 = conv_bn_relu(flt, con13)
    poo13 = MaxPooling2D(pool_size=2, strides=2, padding='same')(con23)  # 128
    con43 = conv_bn_relu(flt * 2, poo13)
    con53 = conv_bn_relu(flt * 2, con43)
    poo23 = MaxPooling2D(pool_size=2, strides=2, padding='same')(con53)  # 64
    con73 = conv_bn_relu(flt * 4, poo23)
    con83 = conv_bn_relu(flt * 4, con73)
    poo33 = MaxPooling2D(pool_size=2, strides=2, padding='same')(con83)  # 32
    con103 = conv_bn_relu(flt * 8, poo33)
    con113 = conv_bn_relu(flt * 8, con103)
    poo43 = MaxPooling2D(pool_size=2, strides=2, padding='same')(con113)  # 16
    con133 = conv_bn_relu(flt * 16, poo43)
    con143 = conv_bn_relu(flt * 16, con133)
    con143 = Dropout(0.5)(con143)
    return con143


def MLM(a = Input(shape=(256, 256, 1)), b = Input(shape=(256, 256, 1)), c = Input(shape=(256, 256, 1)),num_classes=6):
    kwargs = dict(filters=num_classes, kernel_size=(3, 3), strides=2, padding='same', kernel_initializer='he_normal')
    model1 = Cardiac_Seg1(a)
    model2 = Cardiac_Seg2(b)
    model3 = Cardiac_Seg3(c)
    merge1 = concatenate([model1, model2, model3], axis=-1)  # (?, 16, 16, 512*3)
    merge1 = Conv2D(flt, (1, 1), padding='same', activation='relu')(merge1)

    upsample1 = Conv2DTranspose(**kwargs)(merge1)
    fuse1 = concatenate([upsample1, conv11, co112, con113], axis=-1)
    conv16 = conv_bn_relu(flt * 16, fuse1)
    conv17 = conv_bn_relu(flt * 16, conv16)

    upsample2 = Conv2DTranspose(**kwargs)(conv17)
    fuse2 = concatenate([upsample2, conv8, co82, con83], axis=-1)
    conv19 = conv_bn_relu(flt * 8, fuse2)
    conv20 = conv_bn_relu(flt * 8, conv19)

    upsample3 = Conv2DTranspose(**kwargs)(conv20)
    fuse3 = concatenate([upsample3, conv5, co52, con53], axis=-1)
    conv22 = conv_bn_relu(flt * 4, fuse3)
    conv23 = conv_bn_relu(flt * 4, conv22)

    upsample4 = Conv2DTranspose(**kwargs)(conv23)
    fuse4 = concatenate([upsample4, conv2, co22, con23], axis=-1)
    conv25 = conv_bn_relu(flt * 2, fuse4)
    conv26 = conv_bn_relu(flt * 2, conv25)

    '''
    model4 = path1(a)
    model5 = path1(b)
    model6 = path1(c)
    merge2 = concatenate([conv26, model4, model5, model6], axis=-1)
    merge2 = conv_bn_relu(flt * 2, merge2)
    '''

    output = Conv2D(6, kernel_size=(1, 1), activation='softmax')(conv26)
    model = Model(inputs=[a, b, c], outputs=output)
    model.compile(optimizer=Adam(lr=0.0001), loss=[forward],
                  metrics=[class_mertics2, class_Edema, class_Scar])
    #model.summary()
    #plot_model(model, 'F:\\2020MICCAI_Cardiac_Segmentation\\Cardiac_Seg_MLM.png', show_shapes=True)
    return model

def normolize(input):
    epsilon = 1e-6
    mean = np.mean(input)
    std = np.std(input)
    return (input-mean)/(std+epsilon)

def label_smoothing(inputs, epsilon=0.01):
    return ((1-epsilon) * inputs) + (epsilon / 6)

def train():
    train_C0 = np.load('F:\\train_data\\c0.npy')
    train_DE = np.load('F:\\train_data\\lge.npy')
    train_T2 = np.load('F:\\train_data\\t2.npy')
    train_mask = np.load('F:\\train_data\\label.npy')
    train_C0 = normolize(train_C0)
    train_DE = normolize(train_DE)
    train_T2 = normolize(train_T2)
    train_mask = label_smoothing(train_mask)

    val1 = np.load('F:\\test_data\\c0.npy')  # (11, 256, 256, 1)
    val2 = np.load('F:\\test_data\\lge.npy')  # (11, 256, 256, 1)
    val3 = np.load('F:\\test_data\\t2.npy')  # (11, 256, 256, 1)
    val_label = np.load('F:\\test_data\\label.npy')  # (11, 256, 256, 1)
    val1 = normolize(val1)
    val2 = normolize(val2)
    val3 = normolize(val3)
    val_label = label_smoothing(val_label)

    earlystop = EarlyStopping(monitor='class_mertics2', patience=20, verbose=1, mode='max')
    reduce_lr = ReduceLROnPlateau(monitor='class_mertics2', factor=0.1, patience=20, mode='max')
    model = MLM()
    model.load_weights('F:\\2020MICCAI_Cardiac_Segmentation\\Cardiac_Seg_MLM.hdf5')
    csv_logger = CSVLogger('Cardiac_Seg_MLM.csv')
    model_checkpoint = ModelCheckpoint(filepath='Cardiac_Seg_MLM.hdf5', monitor='loss', verbose=1, save_best_only=True, mode= 'min')
    model.fit([train_C0, train_DE, train_T2], train_mask, batch_size=6, validation_data=([val1, val2, val3], val_label), epochs=100, verbose=1, shuffle=True,
              callbacks=[model_checkpoint, csv_logger, earlystop, reduce_lr])


if __name__ == '__main__':
    train()





