import glob
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import os
import nibabel
import scipy.io as io
import SimpleITK as sitk


def jpg2npy(file_dir,save_dir):
    i=0
    imgs = glob.glob(file_dir + '/*.jpg')
    #imgdatas = np.ndarray((15,512,512,1))
    imgdatas = np.ndarray((1232,128,128,1))
    #imgdatas = np.ndarray((15,400,400,1))
    for imgname in imgs:
        midname = imgname[imgname.rindex("/") + 1:]
        print(midname)
        img = load_img(file_dir + "/" + midname, grayscale=True)
        img = img_to_array(img)
        print(str(img.shape))
        for x in range(128):
            for y in range(128):
                if img[:,:,0][x][y]>0:
                    imgdatas[i,:,:,:][x][y]=1
        #imgdatas[i,:,:,:] = img
        i += 1
    print(i)
    np.save(save_dir+'train_mask2.npy', imgdatas)


def nii2npy(path1, path2):
    N = 45
    if not os.path.exists(path2):
        os.makedirs(path2)
    for n in range(N):
        print('Processing File ' + str(n + 1))
        filename1 = 'patient' + str(n + 1) + '_CO' + '.nii.gz'
        directory1 = os.path.join(path1, filename1)
        filename2 = 'patient' + str(n + 1) + '_LCO' + '.npy'
        file1 = os.path.join(path1, filename1)
        data = nibabel.load(file1).get_data()
        print('  Data shape is ' + str(data.shape) + ' .')
        file2 = os.path.join(path2, filename2)
        np.save(file2, data)
        print('File ' + 'patient' + str(n + 1) + '_LGE.' + ' is saved in ' + file2 + ' .')


def npy2jpg():
    for j in range(1):
        imgs = np.load('../c0gt_npy/test/patient36_C0.npy'%(j+1))

        for i in range(imgs.shape[2]):
            img = np.expand_dims(imgs[:, :, i], axis=2)
            img1 = array_to_img(img)
            img1.save("../c0gt_npy/patient%d_C0_%d.jpg"%(j+1,i+1))
#npy2jpg()


def tojpg():
    img= np.load('C:/Users/fly/Downloads/MDFA_Net_c0200f_result.npy')
    #img = img[0:10,:,:,:]
    for i in range(img.shape[0]):
        x = img[i,:,:,:]
        print(x.shape)
        y =array_to_img(x)
        y.save('C:/Users/fly/Downloads/test/patient%d.jpg'%i)
#tojpg()


def npy2mat(file_dir,save_dir):
    i=0
    imgs = glob.glob(file_dir + '*.npy')
    for imgname in imgs:
        midname = imgname[imgname.rindex("/") + 1:]
        print(midname)
        img = np.load(file_dir + midname)
        io.savemat(save_dir+midname[:-4]+'.mat', {'data': img})


def jpg2nii(file_dir,save_dir):
    img = glob.glob(file_dir + '*.jpg')
    for imgname in img:
        midname = imgname[imgname.rindex("/") + 1:]
        img = load_img(file_dir + midname)
        img_npy = img_to_array(img)
        img_npy = sitk.GetImageFromArray(img_npy)
        sitk.WriteImage(img_npy, save_dir + midname[0:-4] + '.nii')