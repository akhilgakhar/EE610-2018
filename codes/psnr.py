#### This paart of code fragment computes the PSNR and SSIM of Restored Image in comparison to that of available Ground truth ####


import numpy as np 
import math
import cv2
import skimage
from skimage.measure import compare_ssim


def psnr(img1, img2):

    ### Calculating mean-squared error 
    mse_calc_mean=(np.sum( (img1 - img2) ** 2 ))/(img1.shape[0]*img1.shape[1])
    print('mse_calc_mean',mse_calc_mean)
    if mse_calc_mean == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse_calc_mean))

def ssim_1(img1,img2):
    ## using inbuilt function to calculate SSIM
    ss=compare_ssim(img1,img2,multichannel=True)
    return ss

#b=[1,10e-5,100]
#a=['gamma_1','gamma_10e-5','gamma_100']
    
#b=[1e5,5e3,5e4,9e4]
#a=['L_1e5','L_5e3','L_5e4','L_9e4']

#b=[0.5,0.8]
#a=['wn_0.5','wn_0.8']
        
# for x in range(0,len(a)):


#     file = open('/home/akhil/Desktop/SEM3/Study/subjects/Image_processing/Image_processing_EE/assignments/assign_2/restored_images/truncated/'+a[x]+'/PSNR_SSIM.txt','w') 
#     file.write('truncated Filtering\n')
#     file.write('Ground truth Image       Deblurred Image                     PSNR                         SSIM\n')

#     for i in range(1,5):
#         for k in range(1,8):
#             original = cv2.imread('/home/akhil/Desktop/SEM3/Study/subjects/Image_processing/Image_processing_EE/assignments/assign_2/Original_Images/gt_'+str(i)+'.jpg')
#             blurr = cv2.imread('/home/akhil/Desktop/SEM3/Study/subjects/Image_processing/Image_processing_EE/assignments/assign_2/restored_images/truncated/'+a[x]+'/gt_'+str(i)+'_k_'+str(k)+'_wn_'+str(b[x])+'.png')
#             file.write('gt_'+str(i)+'.jpg                gt_'+str(i)+'_k_'+str(k)+'_wn_'+str(b[x])+'.png           '+str(psnr(original,blurr))+'           '+str(ssim_1(original,blurr))+'\n' )
#             print(str(i)+str(k))
#     file.close() 




file = open('/home/akhil/Desktop/SEM3/Study/subjects/Image_processing/Image_processing_EE/assignments/assign_2/173079027/restored_images/inverse/PSNR_SSIM.txt','w') 
file.write('Full Inverse Filtering\n')
file.write('Ground truth Image       Deblurred Image                     PSNR                         SSIM\n')

for i in range(1,5):
    for k in range(1,8):
        original = cv2.imread('/home/akhil/Desktop/SEM3/Study/subjects/Image_processing/Image_processing_EE/assignments/assign_2/173079027/Original_Images/gt_'+str(i)+'.jpg')
        blurr = cv2.imread('/home/akhil/Desktop/SEM3/Study/subjects/Image_processing/Image_processing_EE/assignments/assign_2/173079027/restored_images/inverse/gt_'+str(i)+'_k_'+str(k)+'_inv.png')
        file.write('gt_'+str(i)+'.jpg                gt_'+str(i)+'_k_'+str(k)+'_inv.png           '+str(psnr(original,blurr))+'           '+str(ssim_1(original,blurr))+'\n' )
        #print(str(i)+str(k))
file.close() 

