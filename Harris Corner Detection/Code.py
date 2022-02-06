import Utilities as utl
import cv2
import numpy as np
import matplotlib.pyplot as plt




def NonMaximalSuppression(img, radius):

    MyIndices = []

    for i in range(0,img.shape[0]):
        for j in range(0,img.shape[1]):
            if i+1 <img.shape[0] and j+1 <img.shape[1]:
                MyIndices.append([[i, j], [i, j + 1], [i + 1, j], [i + 1, j + 1]])



    for imglen in range(len(MyIndices)):
        MaxVal = 0
        MaxIndexi = 0
        MaxIndexj = 0
        for r in range(radius*radius):
            i = MyIndices[imglen][r][0]
            j = MyIndices[imglen][r][1]
            if img[i,j] > MaxVal:
                MaxVal = img[i,j]
                MaxIndexi = i
                MaxIndexj = j

        #Setting Values
        for rad in range(radius*radius):
            i = MyIndices[imglen][rad][0]
            j = MyIndices[imglen][rad][1]
            if i == MaxIndexi and j == MaxIndexj:
                img[MaxIndexi,MaxIndexj] = 255
            else:
                img[i, j] = 0
        # print('Max I value = '+str(MaxIndexi))
        # print('Max J Value = '+str(MaxIndexj))
        # print('Max Value = '+str(MaxVal))
    # print(img)






    # print(img)




    """
    consider only the max value "Let it 255"
    within window of size(radious x radious)
    around each pixel and assume all other value with 0
    """

    return img

"""
Sample Input: Input/chessboard.jpg
Test Input: Input/*
Steps:
1- Gradients in both the X and Y directions.
2- Smooth the derivative a little using gaussian (can be any other smoothing)
3- Calculate R:
    3.1 Loop on each pixel:
    3.2 Calculate M for each pixel:
        3.2.1 calculate a11=Gx^2, a12=GxGy, a21=GxGy, a22=Gy^2 
    3.3 Calculate Det_M = np.linalg.det(a) or Det_M = a11*a22 - a12*a21; and trace=a11+a22
    3.4 Calculate Response at this pixel = det-k*trace^2
    3.5 Display the result, but make sure to re-scale the data in the range 0 to 255 
4- Threshold and Non-Maximal Suppression 

"""
def harris(img, verbose=True):
    # 1- gradients in both the X and Y directions.
    Gx, Gy = utl.get_gradients_xy(img, 5)

    if verbose:
        cv2.imshow("Gradients", np.hstack([Gx, Gy]))

    Gx = cv2.GaussianBlur(Gx,(5,5),3)
    Gy = cv2.GaussianBlur(Gy,(5,5),3)

    #End of Student Code

    cv2.imshow("Blured", np.hstack([Gx, Gy]))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 3- Calculate R:
    R = np.zeros(img.shape)
    k = 0.04

    # 	3.1 Loop on each pixel:
    for i in range(len(Gx)):
        for j in range(len(Gx[i])):
            a11 = Gx[i,j] * Gx[i,j]
            a12 = Gx[i,j] * Gy[i,j]
            a21 = Gx[i,j] * Gy[i,j]
            a22 = Gy[i,j] * Gy[i,j]
            # 3.2 Calculate M for each pixel:
            #     M = [[a11, a12],
            #          [a21, a22]]
            # where a11=Gx^2, a12=GxGy, a21=GxGy, a22=Gy^2

            #Student Code ~ 1 line of code
            M = np.array([[a11 , a12],
                          [a21 , a22]])

            #End of Student Code


            # 3.3 Calculate Det_M
            # Hint: use np.linalg.det(a) or Det(a) = a11*a22 - a12*a21;
            # Student Code ~ 1 line of code
            Det_M = np.linalg.det(M)

            # End of Student Code

            # 3.4 Calculate Response at this pixel = det-k*trace^2
            # where trace=a11+a22
            trace = a11 + a22

            #Student Code ~ 1 line of code
            R[i,j] = Det_M - k*(trace*trace)


            #End of Student Code

    # 4 Display the result, but make sure to re-scale the data in the range 0 to 255
    R = utl.rescale(R, 0, 255)
    #print(R)
    plt.imshow(R, cmap="gray")


    # 5- Threshold and Non-Maximal Suppression
    # If We Have Threshold  = 255
    Threshold = 180
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            if R[i,j] > Threshold:
                continue
            else:
                R[i,j] = 0

    # End of Student Code

    R = NonMaximalSuppression(R, 2)
    return R

img_pairs = [['check.bmp', 'check_rot.bmp']]


dir = 'input/'
i = 0

for img1,img2 in img_pairs:
    i += 1
    image1 = cv2.imread(dir+img1, 0)
    image2 = cv2.imread(dir+img2 , 0)
    r1 = harris(image1)
    r2 = harris(image2)

    plt.figure(i)
    plt.subplot(221), plt.imshow(image1, cmap='gray')
    plt.subplot(222), plt.imshow(image2, cmap='gray')
    plt.subplot(223), plt.imshow(r1, cmap='gray')
    plt.subplot(224), plt.imshow(r2, cmap='gray')
    plt.show()