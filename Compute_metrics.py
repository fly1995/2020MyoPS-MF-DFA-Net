import numpy as np
from Metrics import *
import SimpleITK as sitk

gt = np.load("E:/JBHI/processed_data/c0gt200/c0gt200.npy")
pred = np.load("E:/JBHI/processed_data/c0gt200/c0gt200.npy")
for i in range(pred.shape[0]):
    for j in range(160):
        for k in range(160):
            if pred[i][j][k] >= 0.5:
                pred[i][j][k] = 1
            else:
                pred[i][j][k] = 0

result1 = []
result2 = []
result3 = []

for i in range(3):
    x1 = dice_coef(gt[i:i+1], pred[i:i+1])
    result1.append(x1)
    x2 = jaccard(gt[i:i+1], pred[i:i+1])
    result2.append(x2)

    a = sitk.GetImageFromArray(gt[i], isVector=False)
    b = sitk.GetImageFromArray(pred[i], isVector=False)
    hausdorff = sitk.HausdorffDistanceImageFilter()
    hausdorff.Execute(a,b)
    x3 = hausdorff.GetHausdorffDistance()
    result3.append(x3)


with tf.Session() as sess:
    y1 = sess.run(result1)
    y2 = sess.run(result2)
y3 = result3


np.savetxt('E:/JBHI/model_and_csv/dice.csv',y1)
np.savetxt('E:/JBHI/model_and_csv/jaccard.csv',y2)
np.savetxt('E:/JBHI/model_and_csv/hausdorff.csv',y3)

print("Dice:mean std min max",end='\t')
mean = sum(y1)/len(y1)
print(str(mean),end='\t')
std = np.std(y1)
print(str(std),end='\t')
print(str(min(y1)),end='\t')
print(str(max(y1)),end='\n')


print("Jaccard:mean std min max",end='\t')
mean = sum(y2)/len(y2)
print(str(mean),end='\t')
std = np.std(y2)
print(str(std),end='\t')
print(str(min(y2)),end='\t')
print(str(max(y2)),end='\n')

print("Hausdorff:mean std min max",end='\t')
mean = sum(y3)/len(y3)
print(str(mean),end='\t')
std = np.std(y3)
print(str(std),end='\t')
print(str(min(y3)),end='\t')
print(str(max(y3)),end='\n')

gt = np.load('E:/JBHI/processed_data/t2gt200/t2gt200.npy')  # (324, 160, 160, 1)
gt = gt[:,:,:,0]
pred = np.load('C:/Users/fly/Downloads/t2200_result/MSCMR_t2200_result.npy')  # (324, 160, 160, 1)
pred=pred[:,:,:,0]
# cv2.imshow('show', gt[0:1, :, :, 0])
# cv2.waitKey(0)
surface = Surface(gt, pred, connectivity=2)
assd = surface.get_average_symmetric_surface_distance()
print(assd)