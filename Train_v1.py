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
from keras.utils import plot_model
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'gpu': 0})))
K.set_image_data_format('channels_last')
flt = 16
def dice_coef(y_true, y_pred):
    sum1 = 2*tf.reduce_sum(y_true*y_pred, axis=(0, 1, 2))
    sum2 = tf.reduce_sum(y_true**2+y_pred**2, axis=(0, 1, 2))
    dice = (sum1+0.00001)/(sum2+0.00001)
    dice = tf.reduce_mean(dice)
    return dice

def dice_coef_loss(y_true, y_pred):
    return 1.-dice_coef(y_true, y_pred)

def compute_softmax_weighted_loss(gt, y_pred):
    n_dims=y_pred.shape[-1]
    loss = 0.
    for i in range(n_dims):
        gti = gt[:,:,:,i]
        predi = y_pred[:,:,:,i]
        weighted = 1-(tf.reduce_sum(gti)/tf.reduce_sum(gt))
        focal_loss=1
        loss = loss + -tf.reduce_mean(weighted * gti * focal_loss * tf.log(tf.clip_by_value(predi, 0.005, 1 )))
    return loss


def IB(input,flt):
    conv1 = Conv2D(flt, (1,1), activation='relu', padding='same')(input)
    conv2 = Conv2D(flt, (1, 1), activation='relu', padding='same')(input)
    conv21 = Conv2D(flt, (3, 3), activation='relu', padding='same')(conv2)
    conv3 = Conv2D(flt, (1,1), activation='relu', padding='same')(input)
    conv31 = Conv2D(flt, (3, 3), activation='relu', padding='same')(conv3)
    conv32 = Conv2D(flt, (3, 3), activation='relu', padding='same')(conv31)
    concate = concatenate([conv1, conv21, conv32], axis=3)
    conv = Conv2D(flt, (1, 1), activation='relu')(concate)
    output = conv
    return output


def dial_multi_conv(flt, input):
    conv1 = Conv2D(flt, (3, 3),  dilation_rate=1, activation='relu', padding='same')(input)
    conv2 = Conv2D(flt, (3, 3),  dilation_rate=2, activation='relu', padding='same')(input)
    conv3 = Conv2D(flt, (3, 3),  dilation_rate=4, activation='relu', padding='same')(input)
    concat = concatenate([conv1,conv2,conv3],axis=3)
    return concat


def path1(x = Input(shape=(256, 256, 1)), features=16, depth=4):
    inputs = x
    maps = [inputs]
    ib = dial_multi_conv(16,inputs)
    x = Conv2D(features, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(ib)
    for n in range(depth):
        x = Conv2D(features, (3, 3),  strides=1, dilation_rate=2, padding='same',kernel_initializer='he_normal')(x)
        maps.append(x)
        x = Concatenate(axis=3)(maps)
        x = normalization.BatchNormalization()(x)
        x = Activation('relu')(x)
    x1 = Conv2D(features, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(x)
    return x1


def conv_bn_relu(flt, input):
    kwargs = dict(kernel_size=(3, 3), strides=1, padding='same', kernel_initializer='he_normal')
    conv1 = Conv2D(flt, **kwargs)(input)
    conv1 = normalization.BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    return conv1



def Cardiac_Seg(x = Input(shape=(256, 256, 1))):
    inputs = x
    global conv1, conv2, conv3, conv4,conv5, conv6, conv7, conv8,conv9, conv10, conv11, conv12,conv13, conv14, conv15, pool1, pool2, pool3, pool4
    conv1 = conv_bn_relu(flt, inputs)
    conv2 = conv_bn_relu(flt, conv1)
    conv3 = conv_bn_relu(flt, conv2)
    conv3 = concatenate([inputs, conv3],axis=-1)
    conv3 = dial_multi_conv(flt, conv3)

    pool1 = MaxPooling2D(pool_size=2, strides=2, padding='same')(conv3)#128
    conv4 = conv_bn_relu(flt*2, pool1)
    conv5 = conv_bn_relu(flt*2, conv4)
    conv6 = conv_bn_relu(flt*2, conv5)
    conv6 = concatenate([pool1, conv6], axis=-1)
    conv6 = dial_multi_conv(flt*2, conv6)

    pool2 = MaxPooling2D(pool_size=2, strides=2, padding='same')(conv6)#64
    conv7 = conv_bn_relu(flt*4, pool2)
    conv8 = conv_bn_relu(flt*4, conv7)
    conv9 = conv_bn_relu(flt*4, conv8)
    conv9 = concatenate([pool2, conv9], axis=-1)
    conv9 = dial_multi_conv(flt*4, conv9)

    pool3 = MaxPooling2D(pool_size=2, strides=2, padding='same')(conv9)#32
    conv10 = conv_bn_relu(flt*8, pool3)
    conv11 = conv_bn_relu(flt*8, conv10)
    conv12 = conv_bn_relu(flt*8, conv11)
    conv12 = concatenate([pool3, conv12], axis=-1)
    conv12 = dial_multi_conv(flt*8, conv12)

    pool4 = MaxPooling2D(pool_size=2, strides=2, padding='same')(conv12)#16
    conv13 = conv_bn_relu(flt*16, pool4)
    conv14 = conv_bn_relu(flt*16, conv13)
    conv15 = conv_bn_relu(flt*16, conv14)
    conv15 = concatenate([pool4, conv15], axis=-1)
    conv15 = dial_multi_conv(flt*16, conv15)
    return conv15


def MLM(a = Input(shape=(256, 256, 1)), b = Input(shape=(256, 256, 1)),c = Input(shape=(256, 256, 1)),num_classes=6):
    kwargs = dict(filters=num_classes, kernel_size=(3, 3), strides=2, padding='same', kernel_initializer='he_normal')
    model1 = Cardiac_Seg(a)
    model2 = Cardiac_Seg(b)
    model3 = Cardiac_Seg(c)
    merge1 = concatenate([model1, model2, model3], axis=-1)#(?, 16, 16, 512*3)
    merge1 = dial_multi_conv(flt*16, merge1)

    upsample1 = Conv2DTranspose(**kwargs)(merge1)
    fuse1 = concatenate([upsample1, conv12], axis=-1)
    conv16 = conv_bn_relu(flt * 16, fuse1)
    conv17 = conv_bn_relu(flt * 16, conv16)
    conv18 = conv_bn_relu(flt * 16, conv17)
    conv18 = concatenate([upsample1, conv18], axis=-1)
    conv18 = dial_multi_conv(flt * 16, conv18)

    upsample2 = Conv2DTranspose(**kwargs)(conv18)
    fuse2 = concatenate([upsample2, conv9], axis=-1)
    conv19 = conv_bn_relu(flt * 8, fuse2)
    conv20 = conv_bn_relu(flt * 8, conv19)
    conv21 = conv_bn_relu(flt * 8, conv20)
    conv21 = concatenate([upsample2, conv21], axis=-1)
    conv21 = dial_multi_conv(flt * 8, conv21)

    upsample3 = Conv2DTranspose(**kwargs)(conv21)
    fuse3 = concatenate([upsample3, conv6], axis=-1)
    conv22 = conv_bn_relu(flt * 4, fuse3)
    conv23 = conv_bn_relu(flt * 4, conv22)
    conv24 = conv_bn_relu(flt * 4, conv23)
    conv24 = concatenate([upsample3, conv24], axis=-1)
    conv24 = dial_multi_conv(flt * 4, conv24)

    upsample4 = Conv2DTranspose(**kwargs)(conv24)
    fuse4 = concatenate([upsample4, conv3], axis=-1)
    conv25 = conv_bn_relu(flt * 2, fuse4)
    conv26 = conv_bn_relu(flt * 2, conv25)
    conv27 = conv_bn_relu(flt * 2, conv26)
    conv27 = concatenate([upsample4, conv27], axis=-1)
    conv27 = dial_multi_conv(flt * 2, conv27)

    model4 = path1(a)
    model5 = path1(b)
    model6 = path1(c)
    merge2 = concatenate([conv27, model4, model5, model6], axis=-1)
    merge2 = conv_bn_relu(flt*2, merge2)

    output = Conv2D(6, kernel_size=(1, 1), activation='softmax')(merge2)
    model = Model(inputs=[a, b, c], outputs=output)

    model.compile(optimizer=Adam(lr=0.0001), loss=[compute_softmax_weighted_loss], metrics=['categorical_accuracy'])

    #model.summary()
    plot_model(model, 'F:\\2020MICCAI_Cardiac_Segmentation\\Cardiac_Seg_MLM.png', show_shapes=True)
    return model

def normolize(input):
    epsilon = 1e-6
    mean = np.mean(input)
    std = np.std(input)
    return (input-mean)/(std+epsilon)

def train():
    train_C0 = np.load('F:\\2020MICCAI_Cardiac_Segmentation\\myops2020\\C0_train.npy')
    train_DE = np.load('F:\\2020MICCAI_Cardiac_Segmentation\\myops2020\\DE_train.npy')
    train_T2 = np.load('F:\\2020MICCAI_Cardiac_Segmentation\\myops2020\\T2_train.npy')
    train_mask = np.load('F:\\2020MICCAI_Cardiac_Segmentation\\myops2020\\train_label.npy')
    train_C0 = normolize(train_C0)
    train_DE = normolize(train_DE)
    train_T2 = normolize(train_T2)

    earlystop = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, mode='auto')
    model = MLM()
    #model.load_weights('F:\\2020MICCAI_Cardiac_Segmentation\\Cardiac_Seg_MLM.hdf5')
    csv_logger = CSVLogger('Cardiac_Seg_MLM.csv')
    model_checkpoint = ModelCheckpoint(filepath='Cardiac_Seg_MLM.hdf5', monitor='loss', verbose=1, save_best_only=True, mode= 'min')
    model.fit([train_C0, train_DE, train_T2], train_mask, batch_size=4, validation_split=0.2, epochs=100, verbose=1, shuffle=True,
              callbacks=[model_checkpoint, csv_logger, earlystop, reduce_lr])


if __name__ == '__main__':
    train()



