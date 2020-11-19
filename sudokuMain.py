print('Setting UP')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np
import os
from utlis import *   
import sudokuSolver

###################################
pathImage = 'images/7-medium.png'
heightImg = 450
widthImg = 450
model = initializePredictionModel() #load cnn model
####################################

#1. IMAGE PREPARATION
img = cv2.imread(pathImage)
img = cv2.resize(img, (widthImg,heightImg))
imgBlank = np.zeros((heightImg, widthImg,3),np.uint8) # blank image for debugging
imgThreshold = preProcess(img)

#2. FIND ALL CONTOURS
imgContours = img.copy() # copy image to display all contours
imgBigContour = img.copy() #copy image to display the biggest contour
contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #find contour
cv2.drawContours(imgContours, contours, -1, (0,255,0),3) #draw all contour


#3. FIND THE BIGGEST CONTOURS
biggest, max_area = biggestContour(contours)
# print(biggest) #4 points of biggest contour is not in proper arrangement. need to sort.
if biggest.size != 0:
    biggest = reorder(biggest) #function call reorder check utils
    # print(biggest)
    cv2.drawContours(imgBigContour, biggest,-1,(0,0,255),20) #draw biggest contours
    pts1 = np.float32(biggest) #prepare points for warp
    pts2 = np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]]) # PREPARE POINTS FOR WARP
    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
    imgDetectedDigits = imgBlank.copy()
    imgWarpColored = cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY)

    #4. SPLIT THE IMAGE AND FIND NUMBERS
    imgSolvedDigits = imgBlank.copy() #copy for display purpose
    boxes = splitBoxes(imgWarpColored)
    # print(len(boxes)) #should get 81
    #cv2.imshow(boxes[0]) #will show box no 1
    numbers = getPrediction(boxes, model)
    print(numbers)
    #display detected number/digits
    imgDetectedDigits = displayNumbers(imgDetectedDigits, numbers, color=(255,0,255))
    numbers = np.asarray(numbers)
    #find the position where number > 0, put 0 overthere otherwise 1
    posArray = np.where(numbers > 0, 0 ,1)
    print(posArray)

    # 5. FIND THE SOLUTIONS
    # we need to rearrange our number array[] to the format board as per sudokuSolver.py. So we need to split the array
    board = np.array_split(numbers,9)
    print(board)
    #use try accept for error handling
    try:
        sudokuSolver.solve(board)
    except:
        pass
    print('------------After Solve----------------')
    print(board)

    #change the result to flat array again and draw
    flatList = []
    for sublist in board:
        for item in sublist:
            flatList.append(item)
    solvedNumbers =flatList*posArray
    imgSolvedDigits= displayNumbers(imgSolvedDigits,solvedNumbers)

    # #### 6. OVERLAY SOLUTION
    #unwarp the warp image
    pts2 = np.float32(biggest) # PREPARE POINTS FOR WARP
    pts1 =  np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]]) # PREPARE POINTS FOR WARP
    matrix = cv2.getPerspectiveTransform(pts1, pts2)  # inverse matrix
    imgInvWarpColored = img.copy()
    imgInvWarpColored = cv2.warpPerspective(imgSolvedDigits, matrix, (widthImg, heightImg))
    inv_perspective = cv2.addWeighted(imgInvWarpColored, 1, img, 0.5, 1)
    imgDetectedDigits = drawGrid(imgDetectedDigits)
    imgSolvedDigits = drawGrid(imgSolvedDigits)


    imageArray = ([img,imgThreshold,imgContours, imgBigContour],
                    [imgDetectedDigits, imgSolvedDigits,imgInvWarpColored,inv_perspective])

    stackedImage = stackImages(imageArray, 1)
    cv2.imshow('Stacked Images', stackedImage)

else:
    print("No Sudoku Found")

cv2.waitKey(0)