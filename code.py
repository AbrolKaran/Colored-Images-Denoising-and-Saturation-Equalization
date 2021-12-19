import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

from numpy.__config__ import show

img = cv2.imread('camera.jpg', 0)
#img = cv2.imread('noiseIm.jpg',0)  #Q4 only
img2 = cv2.imread('noiseIm.jpg',0) 
# img = np.zeros((512,512))
# for i in range(512):
#     img[256,i] = 1
ip_xlen = len(img)
ip_ylen = len(img[0])
ip_size = ip_xlen * ip_ylen


def element_product(img1,img2):
    xlen = len(img1)
    ylen = len(img1[0])
    prod = np.zeros((xlen,ylen),dtype=complex)
    for i in range(xlen):
        for j in range(ylen):
            prod[i][j] = img1[i][j]*img2[i][j]
    return prod   

def getFilter(conj_h_dft,laplac_dft,val):
    denom = element_product(np.abs(conj_h_dft),np.abs(conj_h_dft)) + val*(element_product(np.abs(laplac_dft),np.abs(laplac_dft)))
    filter = conj_h_dft/denom
    return filter

def zero_pad(img,xlenAdd,ylenAdd):
    xlen = len(img)
    ylen = len(img[0])
    padded = np.zeros((xlen+xlenAdd, ylen+ylenAdd))
    for i in range(xlen):
        for j in range(ylen):
            padded[i][j] = img[i][j]
    return padded


def remove_pad(img,xlenAdd,ylenAdd):
    xlen = len(img)
    ylen = len(img[0])
    padded = np.zeros((xlen-xlenAdd, ylen-ylenAdd))
    for i in range(xlen-xlenAdd):
        for j in range(ylen-ylenAdd):
            padded[i][j] = img[i][j]
    return padded





def getConjugate(img):
    xlen = len(img)
    ylen = len(img[0])
    return np.conjugate(img)


def getDft(img):
    dft = np.fft.fft2(img)
    return dft



def getIdft(dft):
    ift = np.fft.ifft2(dft)
    return ift


def getMSE(in_img,out_img):
    xlen = len(in_img)
    ylen = len(in_img[0])
    ip_size = ip_xlen * ip_ylen
    sum = 0
    for i in range(xlen):
        for j in range(ylen):
            sum += (int(out_img[i][j])- int(in_img[i][j]))**2
    mse = sum/ip_size
    return mse


def getPSNR(in_img,out_img):
    mse = getMSE(in_img,out_img)
    psnr = 10*math.log10((255**2)/mse)
    return psnr

def getHue(p):
    p = np.array(p).astype('float32')
    if (p[0] == 0 and p[1] == 0 and p[2] == 0):
        return 0
    num = (1/2)*(2*p[0] - p[1] - p[2])
    denom = ((p[0]-p[1])**2 + (p[0]-p[2])*(p[1]-p[2]))**(1/2)

    theta = math.degrees(math.acos(num/denom))

    if p[2]>p[1]:
        theta = 360-theta
    return theta

def getSaturation(p):
    p = np.array(p).astype('float32')
    if (p[0] == 0 and p[1] == 0 and p[2] == 0):
        return 0
    s = 1 - (3/(p[0]+p[1]+p[2]))*min(min(p[0],p[1]),p[2])
    return s

def getIntensity(p):
    p = np.array(p).astype('float32')
    return (p[0] + p[1] + p[2])/3

def getRed(p):
    if p[0]<120:
        red = p[2]
        red *= (1 + (p[1]*math.cos(math.radians(p[0]))/math.cos(math.radians(60-p[0]))))
    elif p[0]<240:
        p[0] = p[0] - 120
        red = p[2]*(1-p[1])
    return red



def getGreen(p):
    if p[0]>=120 and p[0]<240:
        p[0] = p[0] - 120
        green = p[2]
        green *= (1 + (p[1]*math.cos(math.radians(p[0]))/math.cos(math.radians(60-p[0]))))
    else:
        p[0] = p[0] - 240
        green = p[2]*(1-p[1])
    return green



def getBlue(p):
    if p[0]<120:
        blue = p[2]*(1-p[1])
    else:
        p[0] = p[0] - 240
        blue = p[2]
        blue *= (1 + (p[1]*math.cos(math.radians(p[0]))/math.cos(math.radians(60-p[0]))))
    return blue

def RGBtoHSI(img):
    xlen = len(img)
    ylen = len(img[0])
    hsi = np.zeros((xlen,ylen,3),dtype='float32')
    for i in range(xlen):
        for j in range(ylen):
            hue = getHue(img[i][j])
            saturation = getSaturation(img[i][j])
            intensity = getIntensity(img[i][j])
            hsi[i][j] = [hue,saturation,intensity]
    print(hsi[:,:,0])
    print(hsi[:,:,1])
    print(hsi[:,:,2])
    return hsi

def HSItoRGB(hsi):
    xlen = len(hsi)
    ylen = len(hsi[0])
    new_img = np.zeros((xlen,ylen,3),dtype='float32')
    for i in range(xlen):
        for j in range(ylen):
            p = hsi[i][j].astype('float32')
            if p[0] <120:
                red = getRed(hsi[i][j])
                blue = getBlue(hsi[i][j])
                green = 3*p[2] - (red + blue)
            elif p[0] < 240:
                red = getRed(p)
                green = getGreen(p)
                blue = 3*p[2] - (red+green)
            else:
                blue = getBlue(p)
                green = getGreen(p)
                red = 3*p[2] - (blue+green)

            new_img[i][j] = [red,green,blue]
    return new_img


def getIntensityFromHSI(hsi):
    xlen = len(img)
    ylen = len(img[0])
    intensity = np.zeros((xlen,ylen))
    for i in range(xlen):
        for j in range(ylen):
            intensity[i][j] = hsi[i][j][2]
    return intensity

def getHist(img):
    xlen = len(img)
    ylen = len(img[0])
    hist = np.zeros(256)
    for i in range(xlen):
        for j in range(ylen):
            hist[int(img[i][j])] += 1
    hist /= xlen*ylen
    return hist


def getCdf(hist):
    H = np.zeros(256)
    add = 0
    for i in range(256):
        H[i] = add + hist[i]
        add += hist[i]
    return H

def equalize(img, H):
    xlen = len(img)
    ylen = len(img[0])
    mapping = np.zeros(256, int)
    out_img = np.zeros([xlen, ylen], int)
    for i in range(256):
        mapping[i] = int(round(255 * H[i]))
    for i in range(xlen):
        for j in range(ylen):
            out_img[i][j] = mapping[int(img[i][j])]
    return out_img

def getHSIfromIntensity(hsi,intensity):
    xlen = len(img)
    ylen = len(img[0])
    new_hsi = np.zeros((xlen,ylen,3),dtype='float32')
    for i in range(xlen):
        for j in range(ylen):
            new_hsi[i][j] = [hsi[i][j][0],hsi[i][j][1],intensity[i][j]]
    return new_hsi

def changeRnB(img):
    xlen = len(img)
    ylen = len(img[0])
    new_img = np.zeros((xlen,ylen,3),dtype='float32')
    for i in range(xlen):
        for j in range(ylen):
            new_img[i][j] = [img[i][j][2],img[i][j][1],img[i][j][0]]
    return new_img

def showDftLog(dft,name):
    mat = abs(dft)
    fin = np.log(1+mat)/np.log(1+np.max(mat))
    cv2.imshow(name,fin/np.max(fin))

def showDftRaw(dft,name):
    mat = abs(dft)
    mat = 100*(mat - np.min(mat))/(np.max(mat)-(np.min(mat)))  #for q4 multiply by 255
    cv2.imshow(name,mat)
    if name == "noisyCentDft_Raw":
        cv2.imwrite("Centered_DFT_noiseIm.jpg",255*mat) 

# Q1

noisy_psnr = getPSNR(img,img2)

h = (1/121)*np.ones((11,11))
laplac = [[1,1,1],[1,-8,1],[1,1,1]]

h_pad = zero_pad(h,ip_xlen-1,ip_ylen-1)
img2_pad = zero_pad(img2,10,10)
laplac_pad = zero_pad(laplac,263,263)

h_dft = getDft(h_pad)
laplac_dft = getDft(laplac_pad)
img2_dft = getDft(img2_pad)

conj_h_dft = getConjugate(h_dft)

filters = [getFilter(conj_h_dft,laplac_dft,val) for val in [0.25,0.5,0.75,1]]


for i in range(4):

    showDftLog(filters[i],"filter"+str(i))

    fin_img_dft = element_product(filters[i],img2_dft)

    fin_img = remove_pad(np.real(getIdft(fin_img_dft)),10,10)

    cv2.imshow("denoised"+str(i),fin_img/255)

    fin_psnr = getPSNR(img,fin_img) 
    print(fin_psnr)


# # Q3 
# img = cv2.imread('Fig0646(a)(lenna_original_RGB).tif')
# img = np.asarray(img).astype('float32')



# img = changeRnB(img)

# hsi_img = RGBtoHSI(img)

# intensity = getIntensityFromHSI(hsi_img)

# ip_hist = getHist(intensity)


# plt.subplot(1,2,1)
# plt.plot(ip_hist)


# ip_H = getCdf(ip_hist)


# new_intensity = np.array(equalize(intensity, ip_H)).astype('float32')

# print(new_intensity)




# op_hist = getHist(new_intensity)

# plt.subplot(1,2,2)
# plt.plot(op_hist)
# plt.show()

# new_intensity/=255

# new_hsi = getHSIfromIntensity(hsi_img,new_intensity)

# print(new_hsi)

# new_img = HSItoRGB(new_hsi)

# new_img = changeRnB(new_img)

# print(new_img)


# cv2.imshow("new image", new_img)


cv2.waitKey(0)
cv2.destroyAllWindows()





