#!/usr/bin/python
#coding:utf-8
#from Train_v1 import *
#from Train_v2 import *
#from Train_dual_path import *
#from Train_UNet import *
from  Train_ablation import  *
import numpy as np
from keras.preprocessing.image import array_to_img
import tensorflow as tf
session = tf.Session()


def multi_dice(y_true, y_pred):
    sum1 = 2*tf.reduce_sum(y_true*y_pred, axis=(0, 1, 2))
    sum2 = tf.reduce_sum(y_true+y_pred, axis=(0, 1, 2))
    dice = (sum1+0.00001)/(sum2+0.00001)
    dice = tf.reduce_mean(dice)
    return dice

def dice(y_true, y_pred):
    smooth = 1.0
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) /(K.sum(y_true_f) + K.sum(y_pred_f) + smooth)



def test_predict(file_dir, weight_dir, save_dir):
        test_list = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        model = MLM()
        model.load_weights(weight_dir)
        val1 = np.load('F:\\test_data\\c0.npy')  # (11, 256, 256, 1)
        val2 = np.load('F:\\test_data\\lge.npy')  # (11, 256, 256, 1)
        val3 = np.load('F:\\test_data\\t2.npy')  # (11, 256, 256, 1)
        label = np.load('F:\\test_data\\label.npy')  # (11, 256, 256, 1)
        val1 = normolize(val1)
        val2 = normolize(val2)
        val3 = normolize(val3)
        label = label_smoothing(label)
        preds = model.predict([val1,val2,val3])#(11, 256, 256, 6)

        preds[preds >= 0.5] = 1
        preds[preds < 0.5] = 0

        result1 = multi_dice(label, preds)
        print("fly 6 class Dice is :" + str(session.run(result1)))
        p1 = dice(label[...,0:1],preds[...,0:1])
        print("value 0 dice is :" + str(session.run(p1)))
        p2 = dice(label[...,1:2],preds[...,1:2])
        print("value 200 dice is :" + str(session.run(p2)))
        p3 = dice(label[...,2:3],preds[...,2:3])
        print("value 500 dice is :" + str(session.run(p3)))
        p4 = dice(label[...,3:4],preds[...,3:4])
        print("value 600 dice is :" + str(session.run(p4)))
        p5 = dice(label[...,4:5],preds[...,4:5])
        print("value 1220 dice is :" + str(session.run(p5)))
        p6 = dice(label[...,5:6],preds[...,5:6])
        print("value 2221 dice is :" + str(session.run(p6)))
        ave = (p2+p3+p4+p5+p6)/5.0
        print("5 class ave dice is :" + str(session.run(ave)))

        preds = np.argmax(preds, axis=-1) #(11, 256, 256)
        preds = preds.transpose(1, 2, 0)#(256, 256, 11)
        for i in range(21):
            pred = array_to_img(preds[:, :, i:i+1])
            pred.save(save_dir+'patient%d.jpg' % i)

file_dir = 'F:\\2020MICCAI_Cardiac_Segmentation\\myops2020\\'
weight_dir ='F:\\2020MICCAI_Cardiac_Segmentation\\Cardiac_Seg_MLM.hdf5'
save_dir = 'F:\\2020MICCAI_Cardiac_Segmentation\\myops2020\\test_results\\mlm\\'
test_predict(file_dir, weight_dir, save_dir)

