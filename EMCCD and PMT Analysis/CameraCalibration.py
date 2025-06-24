from audioop import avg
from gettext import find
from re import T
from turtle import left
import PIL as PIL
from PIL import Image
import numpy as np
import os
from matplotlib import pyplot as plt
from scipy.fftpack import diff 
from skimage.morphology import h_maxima
from scipy.ndimage import gaussian_filter
from numpy import average, unravel_index


print("Camera Calibration Running")

#os.chdir("/home/frank/spyder/imageProcessing_project/pictures/camera_calibration_images") #calibration image folder


#im = Image.open('raw_img_Mon Mar 21 2022_15.49.35_1080_exp800.tif')
#im = Image.open('raw_img_Fri Feb 18 2022_1024.tif')





#IMPORT CALIBRATION IMAGE HERE
im = Image.open('raw_img_Tue Apr 12 2022_13.10.11_1179.tif')


def Normalize(list):
    min=np.amin(list)
    max=np.amax(list)
    lenAll=len(list)
    lenIndividual=len(list[0])
    normal = np.array([[(list[i][j]-min)/(max-min) for j in range(lenIndividual)] for i in range(lenAll)])
    return normal

#create an array of array of each image
images = []
for i, page in enumerate(PIL.ImageSequence.Iterator(im)):
    images.append(np.array(page))
    
    
numImages= len(images) #number of images
images=gaussian_filter(images,.70) #smooth out the images... might look into getting rid of this

#split the images into bright and dark
darkImages = np.array(images[0:(numImages//2-1)]) #creates array of dark images
brightImages = np.array(images[numImages//2:(numImages-1)]) #creates array of bright images

#FIND ROI'S

#dont think we need this for the dark images 
sumIonImageDark=np.sum(darkImages,axis=0) #sum of all dark images
avgIonImageDark=np.true_divide(sumIonImageDark,len(darkImages)) #average of all the dark images

sumIonImageBright=np.sum(brightImages,axis=0) #sum of all bright images
avgIonImageBright=np.true_divide(sumIonImageBright,len(brightImages)) #average of all the dark images

#sumIonImage=np.sum(images,axis=0) #sum of all images, this was for other file 
#avgIonImage=np.true_divide(sumIonImage,len(images)) #average of all the images, this was for other file

normalizedAvgBright=Normalize(avgIonImageBright) #rescale the image to 0...1, this will ensure h_maxima doesn't need to be changed
findIonPos = h_maxima(normalizedAvgBright,0.2) #average image will be converted to 0s and 1s. For some reason .05 seems to be best for camera calibration purposes
ionPos=(np.array(np.nonzero(findIonPos))).T #array of (x,y) positions of ions [y, x]
numIons = len(ionPos) #number of ions 


#FIND THE CORNERS OF ROI 

m=2 #rectangle of 2*m+1
atomPosCorners=[[[ionPos[i][0]-m, ionPos[i][0]+m],[ionPos[i][1]-m ,ionPos[i][1]+m]] for i in range (numIons)] # [y min, y max],[x min, x max]
atomPosCorners=np.array(atomPosCorners)

#returns array of ROIs for the image/list of images
#for each image there are numIon number of matrices that correspond to the ROI's[image1,image2,...]->[[ionCluster1(4x4),ionCluster2(4x4)]],[ionCluster1,ionCluster2],...]
def clusters(coordinates,img):
    return np.array([[arr[coordinates[i][0,0]:coordinates[i][0,1],coordinates[i][1,0]:coordinates[i][1,1]] for i in range(numIons)] for arr in img])

#sums cluster elements

 #sums cluster elements
#returns array [ion1,ion2]->[[image1Count,image2Count,...,imageNCount],image1Count,image2Count,...,imageNCount]]
def summed(cls,numIm):  
    x = np.array([[np.sum(cls[i][j]) for j in range (numIons)] for i in range(numIm)])
    return np.array([x[:,i] for i in range(numIons)])

clusteredROI=clusters(atomPosCorners,images)
darkClusterkROI=clusters(atomPosCorners,darkImages)
brightClusterROI=clusters(atomPosCorners,brightImages)


#sum of the clusters... this contains the information of all of the ions for bright and dark counts
sumDarkClusters=summed(darkClusterkROI,(numImages//2)-1)
sumBrightClusters=summed(brightClusterROI,(numImages//2)-1)


def findThreshold(darkClusters,brightClusters):
    #create empty 2x2 array: [ion1,ion2,...,ionN]->[[thresh1,avgFid1],[thresh2,avgFid2],...[threshN,avgFid2]]
    threshFid=np.zeros((numIons,2))
    for i in range(numIons):

        DarkHistogramData = darkClusters[i].astype(int) #the zero is the ion number
        BrightHistogramData = brightClusters[i].astype(int) #the zero is the ion number

        DarkHistogramData=np.sort(DarkHistogramData)
        BrightHistogramData=np.sort(BrightHistogramData)


        #get histogram data for dark and bright counts
        brightData=plt.hist(BrightHistogramData,bins = np.max(BrightHistogramData),density=True,alpha=.5)
        darkData=plt.hist(DarkHistogramData,bins=np.max(DarkHistogramData), color= 'g',density=True,alpha=.5)
        #plt.xlabel("Photon Counts")
        #plt.ylabel("Probabilties")
        #plt.title("Exposure: " + str(exposures[k]) + " us filter used")
        #plt.yscale('log')
        #plt.show()
        nBright, binsBright, patchesBright = brightData #this will get the data from histogram
        nDark, binsDark, patchesDark = darkData #this will get the data from histogram

        #want the number of bins of each to be the same... need to add zeros to nDark 
        lenBright=len(nBright)
        lenDark=len(nDark)
        nDark = np.concatenate((nDark,np.zeros(abs(lenBright-lenDark))))

        #print(np.sum(nDark),np.sum(nBright)) #probability conservation test
        

        #values to consider for the threshold...idea: threshold should be between the two means
        darkMean = int(np.ceil(np.mean(DarkHistogramData)))
        brightMean = int(np.ceil(np.mean(BrightHistogramData)))

        thresholds = np.array(range(0,brightMean+1)) #should we start from 0 photon counts or the mean of dark data?...need to think about this
        #error convention... example(thresh=1) anything with 1 ion and below will be dark, anything with 2 ions and above will be bright
        
        def Fidelity_ReturnThreshold(thrsh):
            fidsDark = np.zeros(len(thrsh))
            fidsBright = np.zeros(len(thrsh))
            fidsAvg = np.zeros(len(thrsh))
            diffIter=thrsh[0]
            for n, i in enumerate(thrsh):
                errBrightData=nBright[0:i+1]
                errDarkData=nDark[i+1:]

                errBright=np.sum(errBrightData)
                errDark=np.sum(errDarkData)

                fDark = 1-errDark
                fBright=1-errBright
                fvg=(fDark+fBright)/2

                fidsDark[n] = fDark
                fidsBright[n] = fBright
                fidsAvg[n] = fvg 

            num = np.argmax(fidsAvg) #index of max average fidelity
            actualFidelity = np.max(fidsAvg)
            threshToReturn = diffIter + num #finds the actual threshold 

            return np.array([threshToReturn,actualFidelity])

        threshFid[i]=Fidelity_ReturnThreshold(thresholds)
    return threshFid


data = np.array(findThreshold(sumDarkClusters,sumBrightClusters))

thresholds_final = np.array([i[0] for i in data]) # list of all thresholds 
fidelities_final = np.array([i[1] for i in data]) #list of all fidelities

print(thresholds_final)
print(atomPosCorners)
