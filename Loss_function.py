from __future__ import division, print_function
import tensorflow as tf
from keras import backend as K

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
    for i in range(1,6):
        gti = gt[:,:,:,i]
        predi = y_pred[:,:,:,i]
        weighted = 1-(tf.reduce_sum(gti)/tf.reduce_sum(gt))
        focal_loss=1
        loss = loss + -tf.reduce_mean(weighted * gti * focal_loss * tf.log(tf.clip_by_value(predi, 0.005, 1 )))
    return loss

def class_mertics5(y_true,y_pred,):
    class_dice = []
    for i in range(1,6):
        class_dice.append(diceCoeff(y_true[:,:,:,i:i +1 ],y_pred[:,:,:,i:i + 1]))
    mean_dice = sum(class_dice) / len(class_dice)
    return mean_dice
def class_Edema(y_true,y_pred,):
    class_dice = []
    for i in range(4,5):
        class_dice.append(diceCoeff(y_true[:,:,:,i:i +1 ],y_pred[:,:,:,i:i + 1]))
    mean_dice = sum(class_dice) / len(class_dice)
    return mean_dice
def class_Scar(y_true,y_pred,):
    class_dice = []
    for i in range(5,6):
        class_dice.append(diceCoeff(y_true[:,:,:,i:i +1 ],y_pred[:,:,:,i:i + 1]))
    mean_dice = sum(class_dice) / len(class_dice)
    return mean_dice
'''
def class_Edema(y_true,y_pred,):
    smooth = 1.0
    y_true_f = K.flatten(y_true[:,:,:,4:5])
    y_pred_f = K.flatten(y_pred[:,:,:,4:5])
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def class_Scar(y_true,y_pred,):
    smooth = 1.0
    y_true_f = K.flatten(y_true[:,:,:,5:6])
    y_pred_f = K.flatten(y_pred[:,:,:,5:6])
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
'''
def class_mertics2(y_true,y_pred,):
    r1=class_Edema(y_true,y_pred)
    r2=class_Scar(y_true,y_pred)
    return (r1+r2)/2.0

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
    for i in range(1,6):
        class_dice.append(diceCoeff(y_true[:,:,:,i:i +1], y_pred[:,:,:,i:i + 1]))
    mean_dice = sum(class_dice) / len(class_dice)
    return 1 - mean_dice


def sum_loss(y_true, y_pred):
    l1=forward(y_true,y_pred)
    l2=compute_softmax_weighted_loss(y_true,y_pred)
    sum = l1+l2
    return  sum