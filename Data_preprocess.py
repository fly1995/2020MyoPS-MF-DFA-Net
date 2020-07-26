import glob
import nibabel
import cv2
from keras.preprocessing.image import array_to_img, img_to_array, load_img
import numpy as np
from keras.utils import to_categorical
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'gpu': 0})))


#读取原始nii数据的格式：每个病例中元素的最大值、最小值以及病例的形状,并裁剪中心区域160*160保存为npy
def read_data_format(file_dir,save_dir):
    imgname = glob.glob(file_dir + '\\*' + '.nii.gz')
    for file_name in imgname:
        midname = file_name[file_name.rindex("\\") + 1:]
        img = nibabel.load(file_dir + midname).get_data()
        img_max = np.amax(img)
        img_min = np.amin(img)
        print('%s: max is %d, min is %d' %(midname, img_max, img_min))
        print(img.shape)
        img = img[int((img.shape[0]-256)/2):int((img.shape[0]+256)/2), int((img.shape[1]-256)/2):int((img.shape[1]+256)/2)]
        print(img.shape)
        np.save(save_dir+midname[0:-7], img)
#read_data_format('F:\\7 医学图像数据集\\2020MICCAI心肌病分割挑战赛数据\\train25_myops_gd\\')


#将有标签的切片筛选出来(初步筛选)2020心肌分割挑战赛给的数据都是有效标签，其实不用筛选
def select_slice(file_dir, label_dir, save_file_dir, save_label_dir):
    imglist = []
    labellist = []
    imgname = glob.glob(label_dir+'\\*'+'.npy')
    for file_name in imgname:
        j = 0
        midname = file_name[file_name.rindex("\\") + 1:]
        print(midname)
        img = np.load(file_dir + midname[0:-6]+'DE.npy')
        label = np.load(label_dir + midname)
        print(img.shape)
        newimg = np.zeros((img.shape[0], img.shape[1], img.shape[2]),dtype='float32')
        newlabel = np.zeros((img.shape[0], img.shape[1], label.shape[2]),dtype='float32')
        for i in range(label.shape[2]):#筛选当前切片总和大于0的
            if np.sum(label[:, :, i:i+1]) > 0:
                newimg[:, :, j:j+1] = img[:, :, i:i+1]
                newlabel[:, :, j:j+1] = label[:, :, i:i+1]
                j = j + 1
        finalimg = newimg[:, :, 0: j]
        finallabel = newlabel[:, :, 0: j]
        print(finalimg.shape)
        imglist.append(finalimg)
        labellist.append(finallabel)
    img_save = np.concatenate(imglist, axis=-1)
    print(img_save.shape)
    label_save = np.concatenate(labellist, axis=-1)
    print(label_save.shape)
    np.save(save_file_dir + 'DE_img.npy', img_save)
    np.save(save_label_dir + 'label.npy', label_save)
'''
select_slice("F:\\2020MICCAI_Cardiac_Segmentation\\myops2020\\train25\\DE\\",
                 'F:\\2020MICCAI_Cardiac_Segmentation\\myops2020\\train25_myops_gd\\',
                'F:\\2020MICCAI_Cardiac_Segmentation\\myops2020\\',
                 'F:\\2020MICCAI_Cardiac_Segmentation\\myops2020\\')
'''

#重置标签值并one-hot
def reset_value(label):
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            for k in range(label.shape[2]):
                if label[i][j][k] == 200:
                    label[i][j][k] = 1
                elif label[i][j][k] == 500:
                    label[i][j][k] = 2
                elif label[i][j][k] == 600:
                    label[i][j][k] = 3
                elif label[i][j][k] == 1220:
                    label[i][j][k] = 4
                elif label[i][j][k] == 2221:
                    label[i][j][k] = 5
                else:
                    label[i][j][k] = 0
    newlabel = to_categorical(label, num_classes=6)
    return newlabel
    #np.save(label_dir+'newlabel.npy', newlabel)
#reset_value('F:\\2020MICCAI_Cardiac_Segmentation\\myops2020\\label.npy')


#将生成的numpy转换成jpg并保存下来
def test_npy(file_dir,save_dir):
    npy = np.load(file_dir)
    #npy = np.argmax(npy, axis=-1)
    for i in range(npy.shape[2]):
      img = npy[:,:,i:i+1]
      img = array_to_img(img)
      img.save(save_dir+'patient%d.jpg'%i)
'''
file_dir='F:\\2020MICCAI_Cardiac_Segmentation\\myops2020\\T2_img.npy'
save_dir='F:\\2020MICCAI_Cardiac_Segmentation\\myops2020\\T2\\'
test_npy(file_dir, save_dir)
'''


def gamma_aug(data):
    value = [0.5, 0.6, 0.7, 0.8, 0.9, 1]
    gamma_list = []
    for i in range(len(value)):
        data_gamma = tf.image.adjust_gamma(data, value[i])
        session = tf.Session()
        data_gamma = session.run(data_gamma)
        gamma_list.append(data_gamma)
    gamma_list_save = np.concatenate(gamma_list, axis=0)
    return gamma_list_save


def flip_rot_aug(data):
    data_rot90 = tf.image.rot90(data)
    data_rot180 = tf.image.rot90(data_rot90)
    data_rot270 = tf.image.rot90(data_rot180)
    data_u2d = tf.image.flip_up_down(data)
    data_l2r = tf.image.flip_left_right(data)
    data_trans = tf.image.transpose_image(data)
    session = tf.Session()
    data_rot90 = session.run(data_rot90)
    data_rot180 = session.run(data_rot180)
    data_rot270 = session.run(data_rot270)
    data_u2d = session.run(data_u2d)
    data_l2r = session.run(data_l2r)
    data_trans = session.run(data_trans)
    data_flip_rot_save = np.concatenate([data, data_rot90, data_rot180, data_rot270, data_u2d, data_l2r, data_trans], axis=0)
    print(data_flip_rot_save.shape)
    return data_flip_rot_save


def data_aug(data):
    data1 = gamma_aug(data)
    data2 = flip_rot_aug(data1)
    print(data2.shape)
    return data2

'''
data0 = np.load('F:\\2020MICCAI_Cardiac_Segmentation\\myops2020\\T2_img.npy')
data = data0[:, :, :, np.newaxis].transpose(2, 0, 1, 3)#(102, 256, 256, 1) img
#data = data0.transpose(2, 0, 1, 3)#label
test_list = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
train_data = []
test_data = []
for i in range(data.shape[0]):
    if i not in test_list:
        train_data.append(data[i:i+1, :, :, :])
train = np.concatenate(train_data, axis=0)
for i in test_list:
    test_data.append(data[i:i+1, :, :, :])
test = np.concatenate(test_data, axis=0)

train_aug = data_aug(train)
print(train_aug.shape)
np.save('F:\\2020MICCAI_Cardiac_Segmentation\\myops2020\\T2_train.npy', train_aug)
np.save('F:\\2020MICCAI_Cardiac_Segmentation\\myops2020\\T2_test.npy', test)

print(train_aug.shape)
print(test.shape)
'''

def append_slice(file_dir):
    imglist = []
    imgname = glob.glob(file_dir+'\\*'+'.npy')
    for file_name in imgname:
        midname = file_name[file_name.rindex("\\") + 1:]
        img = np.load(file_dir + midname)
        img = img[int((img.shape[0] - 256) / 2):int((img.shape[0] + 256) / 2),
              int((img.shape[1] - 256) / 2):int((img.shape[1] + 256) / 2)]
        img = np.expand_dims(img, 0)
        label = img
        for i in range(label.shape[0]):
            for j in range(label.shape[1]):
                for k in range(label.shape[2]):
                    if label[i][j][k] == 200:
                        label[i][j][k] = 1
                    elif label[i][j][k] == 500:
                        label[i][j][k] = 2
                    elif label[i][j][k] == 600:
                        label[i][j][k] = 3
                    elif label[i][j][k] == 1220:
                        label[i][j][k] = 4
                    elif label[i][j][k] == 2221:
                        label[i][j][k] = 5
                    else:
                        label[i][j][k] = 0
        img = to_categorical(label, num_classes=6)
        print(img.shape)
        imglist.append(img)
    img_save = np.concatenate(imglist, axis=0)
    print(img_save.shape)
    np.save(file_dir + 'train_label.npy', img_save)

#append_slice('F:\media\media\LIBRARY\Datasets\MyoPS2020\Augdata\\train1\\npy\\Labels\\')

#标签像素值统计
def read_data_format(file_dir):
    imgname = glob.glob(file_dir + '\\*' + '.nii.gz')
    for file_name in imgname:
        r=0
        midname = file_name[file_name.rindex("\\") + 1:]
        img = nibabel.load(file_dir + midname).get_data()
        #print(img.shape[2])

        for i in range(img.shape[2]):
            #print(img.shape[0] * img.shape[1])
            for j in range(img.shape[0]):
                for k in range(img.shape[1]):
                    if img[j][k].any()>0:
                        r=r+1
            print(r)
#read_data_format('F:\\7 医学图像数据集\\2020MICCAI心肌病分割挑战赛数据\\train25_myops_gd\\')