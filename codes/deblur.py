######## Image Restoration Techniques #############

# This Part of code fragment is developed by Akhil Gakhar (173079027) , M.Tech RA , IIT Bombay as a part of course assignment
# of Image Processing Course EE610. Every one is encouraged to use/modify the code segment and find out more possibilities of 
# Image Restoration.

# Dataset of Original Images/Blurred Images/Restored Images/Blurr kernels is provided along.
# 7 different types of kernels have been used to blurr 4 different Original Images/Ground Truth
# Several approaches have been done to restore back the original Image from Blurred Image including:
# 1) Full Inverse Filtering
# 2) Truncated Inverse Filtering
# 3) Weiner Filtering
# 4) Constrained Least Squared Filtering
# 
# Each of which Requires a user Input to provide necessary arguments
# User Defined FFT and IFFT function have be made
# 
# Since we have the ground truth of these Images we have computed PSNR and SSIM of all Deblurred Images
# Moreover an attempt has been to recover the kernel apart from the ground truth from the blurred Images
# 
# In order to save the respective Images automatically, looping has been done at the bottom of this code
#
# As idealy expected both Image and kernel should be size A+B-1 , where A being array size along one of the dimenshions of Image
# and B be size of kernel along the same dimenshion,
# in this program the Image has not been paded with zeros aroung it, only the kernel size has been made equal to that of Image
# as a result one might see some wrap-around artifacts in the blurred Image, it is of little consequence if one is 
# carefull about the kernel been chosen to blurr that very Image
#
# Every function need must be applied to blue,green and red channels of an Image separately 
# and lateron recombined to yeild the desired output

import numpy as np   
import cv2
import cmath
import scipy
from scipy import signal 
import matplotlib
from matplotlib import pyplot as plt



# Given the dimension of desired Image , this function resizes the given image by padding zeros to right and bottom
def resize_image(image,height_img,width_img):
    
    width = image.shape[1]
    height = image.shape[0]
    image=np.vstack((image,np.zeros([height_img-height,width])))  # adding as many zeros to bottom equal to difference in dimesion
    image=np.hstack((image,np.zeros([height_img,width_img-width]))) # adding as many zeros to right equal to difference in dimesion
    
    return image

# Given an Image, this function computes the DFT of the Image
# Since the formula can be given the interpretation of a matrix multiplication
# Specail attention has been given to vectorize the FFT formula for fast computation
def func_fft(func):

    iota = 0+1j
    N = func.shape[1]
    M = func.shape[0]
    # we are dividing the formula into three matrices :
    	# One which computes e^(ux/M)
    	# since both u as well as x can vary from (0,M-1)
    	# therefore this matrix is MxM in shape/size
    exponent_ux_M = np.exp(-1*iota*2*np.pi*np.array(np.fromfunction(lambda x, u: x*u/M, (M, M), dtype=complex)))

    	# One which computes e^(vy/N)
    	# since both v and y can vary from (0,N-1)
    	# the matrix size if NxN
    exponent_vy_N = np.exp(-1*iota*2*np.pi*np.array(np.fromfunction(lambda y, v: y*v/N, (N, N), dtype=complex)))

    	# Third One is Image Matrix iteslf of shape MxN

    # The result of matrix multplication of all these three matrices yeilds FFT of the Image 
    # of same shape as the Original Image
    F_u_v =  np.array(np.matrix(exponent_ux_M)*np.matrix(func)*np.matrix(exponent_vy_N))
    return F_u_v
    
# Given an Image, this function computes the IDFT of the Image
def func_ifft(func):

    iota = 0+1j
    N = func.shape[1]
    M = func.shape[0]
    exponent_ux_M = np.exp(iota*2*np.pi*np.array(np.fromfunction(lambda x, u: x*u/M, (M, M), dtype=complex)))
    exponent_vy_N = np.exp(iota*2*np.pi*np.array(np.fromfunction(lambda y, v: y*v/N, (N, N), dtype=complex)))
    f_x_y =  np.array(np.matrix(exponent_ux_M)*np.matrix(func)*np.matrix(exponent_vy_N))/(M*N)
    return f_x_y
    

# This Function accepts two arguments which can be:
# (Blurred Image, Blurr Kernel) ---> Estimates Ground Truth
# (Blurred Image, Original Image) ---> Estimates Kernel
# The working Principle is division of repective fourier transforms and computing IFFT
def de_blurr(image,img2):
    
    im_fft_image = func_fft(image)
    im_fft_kernel = func_fft(img2) 
    im_ifft3_image = im_fft_image/im_fft_kernel
    restored_image = func_ifft(im_ifft3_image).real
    # since some values can go negative in the process, and hence map to 0
    # this line takes care of this issue and maps every pixel value to (0->255)
    restored_image = (restored_image - restored_image.min())*255/(restored_image.max()-restored_image.min())
    
    return restored_image

# Given the Ground truth adn The blurr Kernel, one can use this funtion to see the effect of Blurring 
# the Image with respective kernel
# underlying priciple is multiplication of two FFt's
# One is encouraged to try both blurring and Deblurring in spatial domains using the same blurr kernels
# since DFT assumes input Image to be  periodic one might see wrap-around error if adequated zero-padding has not been done
def blurr(image,img2):
    
    im_fft_image = func_fft(image)
    im_fft_kernel = func_fft(img2) 
    im_ifft3_image = im_fft_image*im_fft_kernel
    blurred_image = func_ifft(im_ifft3_image).real
    blurred_image = (blurred_image - blurred_image.min())*255/(blurred_image.max()-blurred_image.min())
    
    return blurred_image


# Implementation of weiner filter 
# F_estimate(u,v) = G(u,v)*|H(u,v)|^2/{(|H(u,v)|^2 + L)*H(u,v)}
# where:
# 	H(u,v) -> FFT of the blurr kernel
# 	G(u,v) -> FFT of the blurred Image
# 	L -> Weiner coeff decided by the user 
def weiner_filter(image,L,img2):

    fft_kernel=func_fft(img2)
    fft_image=func_fft(image)
    numerator=np.conjugate(fft_kernel)*fft_kernel
 
    denomenator=fft_kernel*(numerator+L)
    
    restored_image=(numerator/denomenator)*fft_image
    restored_image=func_ifft(restored_image)
    restored_image=restored_image.real
    restored_image = (restored_image - restored_image.min())*255/(restored_image.max()-restored_image.min())
    
    return restored_image 

# Implementation of the constrained Least Square Filter
# F_estimate(u,v) = H*(u,v)*G(u,v)/{(|H(u,v)|^2 + y*|P(u,v)|^2}
# where:
#	P(u,v) is the fourier transform of the Laplacian Kernel (double derivative)
#	Important note is that the kernel,Laplacian,Image should be of same size
# The value of Gamma is decided by user, typical values being 10e-5
def LS_filter(image,kernal,gamma):
       
	h= image.shape[0]
	w= image.shape[1]

	kernal_fft = func_fft(kernal)
	image_fft = func_fft(image)
	 
	laplace = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
	laplace = resize_image(laplace,h,w)
	laplace_fft = func_fft(laplace)

	mag_laplace_fft_squared = np.conjugate(laplace_fft)*laplace_fft
	mag_kernal_fft_squared = np.conjugate(kernal_fft)*kernal_fft

	image_estimate_fft = (np.conjugate(kernal_fft)*image_fft)/(mag_kernal_fft_squared+gamma*mag_laplace_fft_squared)
	restored_image = func_ifft(image_estimate_fft).real    
	restored_image = (restored_image - restored_image.min())*255/(restored_image.max()-restored_image.min())

	return restored_image
    
# Implementation of Truncated Inverse 
# we define a 10th order butterworth low-pass filter 
# The filter is applied along x-dimension and y-dimension separately to the blurred channels of the Image
def trunc_inv(blue,green,red,wn,img2):

    b,a = scipy.signal.butter(10, wn, btype='low')
    blue=de_blurr(blue,img2)
    green=de_blurr(green,img2)
    red=de_blurr(red,img2)
    
    blue=scipy.signal.filtfilt(b, a, blue)
    blue=scipy.signal.filtfilt(b, a, blue.T)
    
    green=scipy.signal.filtfilt(b, a, green)
    green=scipy.signal.filtfilt(b, a, green.T)
    
    red=scipy.signal.filtfilt(b, a, red)
    red=scipy.signal.filtfilt(b, a, red.T)
    blue = (blue - blue.min())*255/(blue.max()-blue.min())
    red = (red - red.min())*255/(red.max()-red.min())
    green = (green - green.min())*255/(green.max()-green.min())
    restored_image = np.dstack((blue.T,green.T,red.T)).astype(np.uint8)
    
    return restored_image

###### Looping done to automatically generate/save Images ####

#for k in range(1,8):       # K -> number of blurr Kernels
#    for i in range(1,5):	# I -> number of Input Images Available (original)
#
#        img2=cv2.imread('/home/akhil/Desktop/SEM3/Study/subjects/Image_processing/Image_processing_EE/assignments/assign_2/kernels/Kernel'+str(k)+'G_c.png',0)	
#        img1=cv2.imread('/home/akhil/Desktop/SEM3/Study/subjects/Image_processing/Image_processing_EE/assignments/assign_2/blurry_images/gt_'+str(i)+'_k_'+str(k)+'_blurred.png')	
#        
#        img2 = img2/np.sum(img2)    # Normalizing the kernel 
#        
#        [blue,green,red] = cv2.split(img1)
#        height1= blue.shape[0]
#        width1= blue.shape[1]
#        N=width1
#        M=height1
#        iota = 0+1j
#        img2 = resize_image(img2,height1,width1)	# resizing the kernel to Image size
#        
#        
#        			### for Truncated Inverse Filter
#        wn =float(input("Enter wn:  "))
#        wn=0.5
#        restored_image = trunc_inv(blue,green,red,wn,img2)
#        
#       			### for Constrained Least-Square Filter
#        gamma =float(input("Enter ls_coeff:\t"))
#        blue = LS_filter(blue,img2,gamma)
#        green = LS_filter(green,img2,gamma)
#        red = LS_filter(red,img2,gamma)
#        restored_image = np.dstack((blue,green,red)).astype(np.uint8)
#        
#					### for weiner filter
#        L=float(input("Enter weiner_filter_coeff"))
#        restored_image = np.dstack((weiner_filter(blue,L,img2),weiner_filter(green,L,img2),weiner_filter(red,L,img2))).astype(np.uint8)
#
#					### for Full Inverse Filtering
#        restored_image = np.dstack((de_blurr(blue,img2),de_blurr(green,img2),de_blurr(red,img2))).astype(np.uint8)
#  
#
#        cv2.imwrite('/home/akhil/Desktop/SEM3/Study/subjects/Image_processing/Image_processing_EE/assignments/assign_2/restored_images/gt_1_k_6.png',restored_image)
#        
#        restored_image = np.dstack((blurr(blue,img2),blurr(green,img2),blurr(red,img2))).astype(np.uint8)
#
#
#cv2.imshow('restored_image',restored_image)
#cv2.waitKey()










##### An attempt to recover the blurr kernel on the same lines as estimating the Ground truth
##### from the blurred Image


# img_Original=cv2.imread('/home/akhil/Desktop/SEM3/Study/subjects/Image_processing/Image_processing_EE/assignments/assign_2/Original_Images/gt_3.jpg')
# img_blurr=cv2.imread('/home/akhil/Desktop/SEM3/Study/subjects/Image_processing/Image_processing_EE/assignments/assign_2/blurry_images/gt_3_k_6.png')	

# [blue_Original,green_Original,red_Original] = cv2.split(img_Original)
# [blue_blurr,green_blurr,red_blurr] = cv2.split(img_blurr)


# gamma =float(input("Enter ls_coeff:\t"))

# kernel_blue = LS_filter(blue_blurr,blue_Original,gamma)
# kernel_green = LS_filter(green_blurr,green_Original,gamma)
# kernel_red = LS_filter(red_blurr,red_Original,gamma)


# #kernel_blue = de_blurr(blue_blurr,blue_Original)
# #kernel_green = de_blurr(green_blurr,green_Original)
# #kernel_red = de_blurr(red_blurr,red_Original)

# plt.subplot(221),plt.imshow(kernel_blue,cmap='gray')
# plt.subplot(222),plt.imshow(kernel_green,cmap='gray')
# plt.subplot(223),plt.imshow(kernel_red,cmap='gray')

# plt.show()