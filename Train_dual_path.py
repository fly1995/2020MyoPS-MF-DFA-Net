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


def class5_mertics(y_true,y_pred,):
    class_dice = []
    for i in range(1, 6):
        class_dice.append(diceCoeff(y_true[:,:,:,i:i +1 ],y_pred[:,:,:,i:i + 1]))
    mean_dice = sum(class_dice) / len(class_dice)
    return mean_dice


def diceCoeff(gt, pred, smooth=1e-5):
    pred_flat = tf.layers.flatten(pred)
    gt_flat = tf.layers.flatten(gt)
    intersection = K.sum((pred_flat * gt_flat))
    unionset = K.sum(pred_flat) + K.sum(gt_flat)
    score = (2 * intersection + smooth) / (unionset + smooth)
    return score

def forward(y_true,y_pred,):
    y_true = tf.convert_to_tensor(y_true)
    y_pred = tf.convert_to_tensor(y_pred)
    class_dice = []
    for i in range(1, 6):
        class_dice.append(diceCoeff(y_true[:,:,:,i:i +1], y_pred[:,:,:,i:i + 1]))
    mean_dice = sum(class_dice) / len(class_dice)
    return 1 - mean_dice


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
    conv3 = Conv2D(flt, (3,3), activation='relu', padding='same')(input)
    conv5 = Conv2D(flt, (5,5), activation='relu', padding='same')(input)
    concate = concatenate([conv1, conv3, conv5], axis=3)
    conv = Conv2D(flt, (1, 1), activation='relu')(concate)
    output = conv
    return output


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


def path1(x = Input(shape=(256, 256, 1)), features=16, depth=4, padding='same', kernel_size = (3, 3)):
    inputs = x
    maps = [inputs]
    ib = IB(inputs, features)
    x = Conv2D(features, kernel_size=(3, 3), activation='relu', padding=padding, kernel_initializer='glorot_uniform')(ib)
    for n in range(depth):
        x = Conv2D(features, kernel_size, padding=padding, kernel_initializer='glorot_uniform')(x)
        x = Conv2D(features, kernel_size, padding=padding, kernel_initializer='glorot_uniform')(x)
        maps.append(x)
        x = Concatenate(axis=3)(maps)
        x = Activation('relu')(x)
    x1 = Conv2D(features, kernel_size=(3, 3), activation='relu', padding=padding,kernel_initializer='glorot_uniform')(x)
    return x1

def mvn(tensor):
    epsilon = 1e-6
    mean = K.mean(tensor, axis=(1, 2), keepdims=True)
    std = K.std(tensor, axis=(1, 2), keepdims=True)
    mvn = (tensor - mean) / (std + epsilon)
    return mvn


def crop(tensors):
    h_dims, w_dims = [], []
    for t in tensors:
        b, h, w, d = K.get_variable_shape(t)
        h_dims.append(h)
        w_dims.append(w)
    crop_h, crop_w = (h_dims[1] - h_dims[0]), (w_dims[1] - w_dims[0])
    rem_h = crop_h % 2
    rem_w = crop_w % 2
    crop_h_dims = (int(crop_h / 2), int(crop_h / 2) + rem_h)
    crop_w_dims = (int(crop_w / 2), int(crop_w / 2) + rem_w)
    cropped = (Cropping2D(cropping=(crop_h_dims, crop_w_dims))(tensors[1]))
    return cropped

def conv_bn_relu(flt, input):
    kwargs = dict(kernel_size=(3, 3), strides=1, padding='same', kernel_initializer='he_normal')
    conv1 = Conv2D(flt, **kwargs)(input)
    conv1 = normalization.BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    return conv1



def Cardiac_Seg1(x = Input(shape=(256, 256, 1))):
    global  mvn11,mvn7,inputs1
    inputs1 = x
    kwargs = dict(kernel_size=3, strides=1, activation='relu', padding='same', use_bias=True,
                  kernel_initializer='glorot_uniform', bias_initializer='zeros', bias_regularizer=None,
                  activity_regularizer=None, kernel_constraint=None, bias_constraint=None, trainable=True, )
    mvn0 = Lambda(mvn, name='mvn01')(inputs1)
    pad = ZeroPadding2D(padding=5, name='pad1')(mvn0)
    conv1 = Conv2D(filters=32, name='conv11', **kwargs)(pad)
    mvn1 = Lambda(mvn, name='mvn11')(conv1)
    conv2 = Conv2D(filters=32, name='conv21', **kwargs)(mvn1)
    mvn2 = Lambda(mvn, name='mvn21')(conv2)
    conv3 = Conv2D(filters=32, name='conv31', **kwargs)(mvn2)
    mvn3 = Lambda(mvn, name='mvn31')(conv3)
    pool1 = MaxPooling2D(pool_size=3, strides=2, padding='valid', name='pool11')(mvn3)
    conv4 = Conv2D(filters=64, name='conv41', **kwargs)(pool1)
    mvn4 = Lambda(mvn, name='mvn41')(conv4)
    conv5 = Conv2D(filters=64, name='conv51', **kwargs)(mvn4)
    mvn5 = Lambda(mvn, name='mvn51')(conv5)
    conv6 = Conv2D(filters=64, name='conv61', **kwargs)(mvn5)
    mvn6 = Lambda(mvn, name='mvn61')(conv6)
    conv7 = Conv2D(filters=64, name='conv71', **kwargs)(mvn6)
    mvn7 = Lambda(mvn, name='mvn71')(conv7)
    pool2 = MaxPooling2D(pool_size=3, strides=2, padding='valid', name='pool21')(mvn7)
    conv8 = Conv2D(filters=128, name='conv81', **kwargs)(pool2)
    mvn8 = Lambda(mvn, name='mvn81')(conv8)
    conv9 = Conv2D(filters=128, name='conv91', **kwargs)(mvn8)
    mvn9 = Lambda(mvn, name='mvn91')(conv9)
    conv10 = Conv2D(filters=128, name='conv101', **kwargs)(mvn9)
    mvn10 = Lambda(mvn, name='mvn101')(conv10)
    conv11 = Conv2D(filters=128, name='conv111', **kwargs)(mvn10)
    mvn11 = Lambda(mvn, name='mvn111')(conv11)
    pool3 = MaxPooling2D(pool_size=3, strides=2, padding='valid', name='pool31')(mvn11)
    drop1 = Dropout(rate=0.5, name='drop1')(pool3)
    conv12 = Conv2D(filters=256, name='conv121', **kwargs)(drop1)
    mvn12 = Lambda(mvn, name='mvn121')(conv12)
    conv13 = Conv2D(filters=256, name='conv131', **kwargs)(mvn12)
    mvn13 = Lambda(mvn, name='mvn131')(conv13)
    conv14 = Conv2D(filters=256, name='conv141', **kwargs)(mvn13)
    mvn14 = Lambda(mvn, name='mvn141')(conv14)
    conv15 = Conv2D(filters=256, name='conv151', **kwargs)(mvn14)
    mvn15 = Lambda(mvn, name='mvn151')(conv15)
    drop2 = Dropout(rate=0.5, name='drop21')(mvn15)
    return  drop2

def Cardiac_Seg2(x = Input(shape=(256, 256, 1))):
    global  mvn112,mvn72,inputs2
    inputs2 = x
    kwargs = dict(kernel_size=3, strides=1, activation='relu', padding='same', use_bias=True,
                  kernel_initializer='glorot_uniform', bias_initializer='zeros', bias_regularizer=None,
                  activity_regularizer=None, kernel_constraint=None, bias_constraint=None, trainable=True, )

    mvn02 = Lambda(mvn, name='mvn02')(inputs2)
    pad2 = ZeroPadding2D(padding=5, name='pad2')(mvn02)
    conv12 = Conv2D(filters=32, name='conv12', **kwargs)(pad2)
    mvn12 = Lambda(mvn, name='mvn12')(conv12)
    conv22 = Conv2D(filters=32, name='conv22', **kwargs)(mvn12)
    mvn22 = Lambda(mvn, name='mvn22')(conv22)
    conv32 = Conv2D(filters=32, name='conv32', **kwargs)(mvn22)
    mvn32 = Lambda(mvn, name='mvn32')(conv32)
    pool12 = MaxPooling2D(pool_size=3, strides=2, padding='valid', name='pool12')(mvn32)
    conv42 = Conv2D(filters=64, name='conv42', **kwargs)(pool12)
    mvn42 = Lambda(mvn, name='mvn42')(conv42)
    conv52 = Conv2D(filters=64, name='conv52', **kwargs)(mvn42)
    mvn52 = Lambda(mvn, name='mvn52')(conv52)
    conv62 = Conv2D(filters=64, name='conv62', **kwargs)(mvn52)
    mvn62 = Lambda(mvn, name='mvn62')(conv62)
    conv72 = Conv2D(filters=64, name='conv72', **kwargs)(mvn62)
    mvn72 = Lambda(mvn, name='mvn72')(conv72)
    pool22 = MaxPooling2D(pool_size=3, strides=2, padding='valid', name='pool22')(mvn72)
    conv82 = Conv2D(filters=128, name='conv82', **kwargs)(pool22)
    mvn82 = Lambda(mvn, name='mvn82')(conv82)
    conv92 = Conv2D(filters=128, name='conv92', **kwargs)(mvn82)
    mvn92 = Lambda(mvn, name='mvn92')(conv92)
    conv102 = Conv2D(filters=128, name='conv102', **kwargs)(mvn92)
    mvn102 = Lambda(mvn, name='mvn102')(conv102)
    conv112 = Conv2D(filters=128, name='conv112', **kwargs)(mvn102)
    mvn112 = Lambda(mvn, name='mvn112')(conv112)
    pool32 = MaxPooling2D(pool_size=3, strides=2, padding='valid', name='pool32')(mvn112)
    drop12 = Dropout(rate=0.5, name='drop12')(pool32)
    conv122 = Conv2D(filters=256, name='conv122', **kwargs)(drop12)
    mvn122 = Lambda(mvn, name='mvn122')(conv122)
    conv132 = Conv2D(filters=256, name='conv132', **kwargs)(mvn122)
    mvn132 = Lambda(mvn, name='mvn132')(conv132)
    conv142 = Conv2D(filters=256, name='conv142', **kwargs)(mvn132)
    mvn142 = Lambda(mvn, name='mvn142')(conv142)
    conv152 = Conv2D(filters=256, name='conv152', **kwargs)(mvn142)
    mvn152 = Lambda(mvn, name='mvn152')(conv152)
    drop22 = Dropout(rate=0.5, name='drop22')(mvn152)
    return  drop22


def Cardiac_Seg3(x = Input(shape=(256, 256, 1))):
    global  mvn113,mvn73,inputs3
    inputs3 = x
    kwargs = dict(kernel_size=3, strides=1, activation='relu', padding='same', use_bias=True,
                  kernel_initializer='glorot_uniform', bias_initializer='zeros', bias_regularizer=None,
                  activity_regularizer=None, kernel_constraint=None, bias_constraint=None, trainable=True, )

    mvn03 = Lambda(mvn, name='mvn03')(inputs3)
    pad3 = ZeroPadding2D(padding=5, name='pad')(mvn03)
    conv13 = Conv2D(filters=32, name='conv13', **kwargs)(pad3)
    mvn13 = Lambda(mvn, name='mvn13')(conv13)
    conv23 = Conv2D(filters=32, name='conv23', **kwargs)(mvn13)
    mvn23 = Lambda(mvn, name='mvn23')(conv23)
    conv33 = Conv2D(filters=32, name='conv33', **kwargs)(mvn23)
    mvn33 = Lambda(mvn, name='mvn33')(conv33)
    pool13 = MaxPooling2D(pool_size=3, strides=2, padding='valid', name='pool13')(mvn33)
    conv43 = Conv2D(filters=64, name='conv43', **kwargs)(pool13)
    mvn43 = Lambda(mvn, name='mvn43')(conv43)
    conv53 = Conv2D(filters=64, name='conv53', **kwargs)(mvn43)
    mvn53 = Lambda(mvn, name='mvn53')(conv53)
    conv63 = Conv2D(filters=64, name='conv63', **kwargs)(mvn53)
    mvn63 = Lambda(mvn, name='mvn63')(conv63)
    conv73 = Conv2D(filters=64, name='conv73', **kwargs)(mvn63)
    mvn73 = Lambda(mvn, name='mvn73')(conv73)
    pool23 = MaxPooling2D(pool_size=3, strides=2, padding='valid', name='pool23')(mvn73)
    conv83 = Conv2D(filters=128, name='conv83', **kwargs)(pool23)
    mvn83 = Lambda(mvn, name='mvn83')(conv83)
    conv93 = Conv2D(filters=128, name='conv93', **kwargs)(mvn83)
    mvn93 = Lambda(mvn, name='mvn93')(conv93)
    conv103 = Conv2D(filters=128, name='conv103', **kwargs)(mvn93)
    mvn103 = Lambda(mvn, name='mvn103')(conv103)
    conv113 = Conv2D(filters=128, name='conv113', **kwargs)(mvn103)
    mvn113 = Lambda(mvn, name='mvn113')(conv113)
    pool33 = MaxPooling2D(pool_size=3, strides=2, padding='valid', name='pool33')(mvn113)
    drop13 = Dropout(rate=0.5, name='drop13')(pool33)
    conv123 = Conv2D(filters=256, name='conv123', **kwargs)(drop13)
    mvn123 = Lambda(mvn, name='mvn123')(conv123)
    conv133 = Conv2D(filters=256, name='conv133', **kwargs)(mvn123)
    mvn133 = Lambda(mvn, name='mvn133')(conv133)
    conv143 = Conv2D(filters=256, name='conv143', **kwargs)(mvn133)
    mvn143 = Lambda(mvn, name='mvn143')(conv143)
    conv153 = Conv2D(filters=256, name='conv153', **kwargs)(mvn143)
    mvn153 = Lambda(mvn, name='mvn153')(conv153)
    drop23 = Dropout(rate=0.5, name='drop23')(mvn153)
    return  drop23

def MLM(a = Input(shape=(256, 256, 1)), b = Input(shape=(256, 256, 1)),c = Input(shape=(256, 256, 1)),num_classes=6):
    model1 = Cardiac_Seg1(a)
    model2 = Cardiac_Seg2(b)
    model3 = Cardiac_Seg3(c)
    merge1 = concatenate([model1, model2, model3], axis=-1)#(?, 16, 16, 512*3)


    score_conv15 = Conv2D(filters=num_classes, kernel_size=1, strides=1, activation=None, padding='valid',
                          kernel_initializer='glorot_uniform', use_bias=True, name='score_conv15')(merge1)
    upsample1 = Conv2DTranspose(filters=num_classes, kernel_size=3, strides=2, activation=None, padding='valid',
                                kernel_initializer='glorot_uniform', use_bias=False, name='upsample1')(score_conv15)
    score_conv11 = Conv2D(filters=num_classes, kernel_size=1, strides=1, activation=None, padding='valid',
                          kernel_initializer='glorot_uniform', use_bias=True, name='score_conv11')(mvn11)
    crop1 = Lambda(crop, name='crop1')([upsample1, score_conv11])
    fuse_scores1 = average([crop1, upsample1], name='fuse_scores1')

    upsample2 = Conv2DTranspose(filters=num_classes, kernel_size=3, strides=2, activation=None, padding='valid',
                                kernel_initializer='glorot_uniform', use_bias=False, name='upsample2')(fuse_scores1)
    score_conv7 = Conv2D(filters=num_classes, kernel_size=1, strides=1, activation=None, padding='valid',
                         kernel_initializer='glorot_uniform', use_bias=True, name='score_conv7')(mvn7)
    crop2 = Lambda(crop, name='crop2')([upsample2, score_conv7])
    fuse_scores2 = average([crop2, upsample2], name='fuse_scores2')

    upsample3 = Conv2DTranspose(filters=num_classes, kernel_size=3, strides=2, activation=None, padding='valid',
                                kernel_initializer='glorot_uniform', use_bias=False, name='upsample3')(fuse_scores2)
    inputs = concatenate([inputs1,inputs2,inputs3])
    crop3 = Lambda(crop, name='crop3')([inputs, upsample3])
    model4 = path1(a)
    model5 = path1(b)
    model6 = path1(c)
    merge2 = concatenate([crop3, model4, model5, model6], axis=-1)

    output = Conv2D(6, kernel_size=(1, 1), activation='softmax')(merge2)
    model = Model(inputs=[a, b, c], outputs=output)
    model.compile(optimizer=Adam(lr=0.0001), loss=[dice_coef_loss], metrics=['categorical_accuracy'])
    #model.summary()
    #plot_model(model, 'F:\\2020MICCAI_Cardiac_Segmentation\\Cardiac_Seg_MLM.png', show_shapes=True)
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



