#!/usr/bin/python
#coding:utf-8
#from Train_v1 import *
#from Train_v2 import *
#from Train_dual_path import *
#from Train_UNet import *
from  Train_ablation import  *
session = tf.Session()
import nibabel as nb
from Data_preprocess import *
np.set_printoptions(threshold=np.inf)
import SimpleITK as sitk

def Test(test_dir, weight_dir):
    imgname = glob.glob(test_dir+'\\*'+'C0.nii.gz')
    for file_name in imgname:
        midname = file_name[file_name.rindex("\\") + 1:]

        test_img1 = nb.load(test_dir + midname).get_data()#(478, 482, 5)
        img_crop1 = test_img1[int((test_img1.shape[0]-256)/2):int((test_img1.shape[0]+256)/2), int((test_img1.shape[1]-256)/2):int((test_img1.shape[1]+256)/2)]
        img_crop1 = img_crop1[:, :, :, np.newaxis].transpose(2, 1, 0, 3)#(5, 256, 256, 1)
        img_crop1 = normolize(img_crop1)

        test_img2 = nb.load(test_dir + midname.replace('C0', 'DE')).get_data()#(478, 482, 5)
        img_crop2 = test_img2[int((test_img2.shape[0]-256)/2):int((test_img2.shape[0]+256)/2), int((test_img2.shape[1]-256)/2):int((test_img2.shape[1]+256)/2)]
        img_crop2 = img_crop2[:, :, :, np.newaxis].transpose(2, 1, 0, 3)#(5, 256, 256, 1)
        img_crop2 = normolize(img_crop2)

        test_img3 = nb.load(test_dir + midname.replace('C0', 'T2')).get_data()#(478, 482, 5)
        img_crop3 = test_img3[int((test_img3.shape[0]-256)/2):int((test_img3.shape[0]+256)/2), int((test_img3.shape[1]-256)/2):int((test_img3.shape[1]+256)/2)]
        img_crop3 = img_crop3[:, :, :, np.newaxis].transpose(2, 1, 0, 3)#(5, 256, 256, 1)
        img_crop3 = normolize(img_crop3)

        model = MLM()
        model.load_weights(weight_dir)
        preds = model.predict([img_crop1,img_crop2,img_crop3]) #(5, 256, 256, 6)

        preds[preds >= 0.5] = 1
        preds[preds < 0.5] = 0

        preds = np.argmax(preds, axis=-1)#(5, 256, 256)
        preds[preds == 0] = 0
        preds[preds == 1] = 200
        preds[preds == 2] = 500
        preds[preds == 3] = 600
        preds[preds == 4] = 1220
        preds[preds == 5] = 2221

        x = np.zeros((int(test_img1.shape[2]), int(test_img1.shape[1]), int(test_img1.shape[0])))#(5,482,478)
        x[:, int((x.shape[1] - 256) / 2): int((x.shape[1] + 256) / 2), int((x.shape[2] - 256) / 2): int((x.shape[2] + 256) / 2)] = preds
        x=x.astype('int16')
        label_nii = sitk.GetImageFromArray(x)
        sitk.WriteImage(label_nii, 'F:\\2020MICCAI_Cardiac_Segmentation\myops2020\\test_results\\Compete_Test\\MLM\\'+midname[0:-9]+'seg.nii.gz')
'''
test_dir = 'F:\\2020MICCAI_Cardiac_Segmentation\\myops2020\\test20\\'
weight_dir ='F:\\2020MICCAI_Cardiac_Segmentation\\Cardiac_Seg_MLM.hdf5'
Test(test_dir, weight_dir)
'''


#test_results
def nii2jpg(file_dir):
    imgname = glob.glob(file_dir+'\\*'+'_seg.nii.gz')
    for file_name in imgname:
        midname = file_name[file_name.rindex("\\") + 1:]
        img = nb.load(file_dir+midname).get_data()#(478, 482, 5)
        img = reset_value(img)
        img = np.argmax(img, axis=-1)
        print(img.shape)
        for i in range(img.shape[2]):
            img0 = array_to_img(img[:, :, i:i + 1])
            img0.save('F:\\2020MICCAI_Cardiac_Segmentation\\myops2020\\test_results\\Compete_Test\\MLM\\'+midname[0:-10]+'%d.jpg' % i)
'''
file_dir = 'F:\\2020MICCAI_Cardiac_Segmentation\\myops2020\\test_results\\Compete_Test\\MLM\\'
nii2jpg(file_dir)
'''


img = nb.load('F:\\7 医学图像数据集\\2020MICCAI心肌病分割挑战赛数据\\train25\\myops_training_101_C0.nii.gz').get_data()
img0 = array_to_img(img[:, :, 1:2])
img0.save('F:\\7 医学图像数据集\\2020MICCAI心肌病分割挑战赛数据\\train25\\x.jpg')