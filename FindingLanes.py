#!/usr/bin/env python
import numpy as np
import cv2
import os
import glob


def findLaneLines(image):
    
    
    #Define Parameters
    #------------------------------------
    #Region mask Parameters
    lowery=1
    upppery=0.6
    dx=0.04
    gap=0.07
    #Color mask parameters
    white_threshold=np.array([180],dtype="uint8")
    yellow_threshold=np.array([50, 0, 150],dtype="uint8")
    upperb_white=np.array([255],dtype="uint8")
    upperb_yellow=np.array([255,150,255],dtype="uint8")
    #Gaussian smoothing kernel
    kernel_size = 5
    #Canny edge detection thresholds
    low_threshold = 15
    high_threshold = 180
    #Hough lines parameters
    rho = 1 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 5     # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 15 #minimum number of pixels making up a line
    max_line_gap = 15    # maximum gap in pixels between connectable line segments
    #threshold to determine if line is yellow
    minYellow=200
    #-------------------------------------
    
    #Convert BGR image to lab color space and gray scale
    lab = cv2.cvtColor(image,cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    #Calculate yellow and white color masks
    white_mask=cv2.inRange(gray,white_threshold,upperb_white)
    yellow_mask=cv2.inRange(lab,yellow_threshold,upperb_yellow)
    #Smoothe gray scale image
    blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)
    #Extract edges from gray scale LAB color space images and merge them
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
    edges=cv2.bitwise_or(cv2.Canny(lab[:,:,2], low_threshold, high_threshold),edges)
    #Filter edges with color masks
    white_edges=cv2.bitwise_and(edges,edges,mask=white_mask)
    yellow_edges=cv2.bitwise_and(edges,edges,mask=yellow_mask)
    #Create regional masks, for areas whare lane lines are expected
    left_mask = np.zeros_like(edges)
    right_mask = np.zeros_like(edges)
    ignore_mask_color = 255   
    #define corners
    ysize = image.shape[0]
    xsize = image.shape[1]
    
    left_left_top=[xsize*(0.5-dx),ysize*upppery]
    left_right_top=[xsize*(0.50-gap),ysize*upppery]
    left_left_bottom=[xsize*dx,ysize*lowery]
    left_right_bottom=[xsize*(0.5-2*gap),ysize*lowery]
    
    right_left_top=[xsize*(0.5+gap),ysize*upppery]
    right_right_top=[xsize*(0.5+dx),ysize*upppery]
    right_left_bottom=[xsize*(0.5+2*gap),ysize*lowery]
    right_right_bottom=[xsize*(1-dx),ysize*lowery]
    #Create polygons from corners
    left_vertices = np.array([[left_left_top,left_right_top, left_right_bottom, left_left_bottom]], dtype=np.int32)
    right_vertices = np.array([[right_left_top,right_right_top, right_right_bottom, right_left_bottom]], dtype=np.int32)
    #Set marks as areas within the polygons
    cv2.fillPoly(left_mask, left_vertices, ignore_mask_color)
    cv2.fillPoly(right_mask, right_vertices, ignore_mask_color)
    #check if lane lines are white or yellow
    countYellowLeft=cv2.countNonZero(cv2.bitwise_and(yellow_mask, left_mask))
    countYellowRight=cv2.countNonZero(cv2.bitwise_and(yellow_mask, right_mask))
    left_yellow=(countYellowLeft>minYellow)
    print("Left: Yellow Count: ", countYellowLeft)
    right_yellow=(countYellowRight>minYellow)
    print("Right Yellow Count: ", countYellowRight)
    if(left_yellow):
        left_masked_edges = cv2.bitwise_and(yellow_edges, left_mask)
    else:
        left_masked_edges = cv2.bitwise_and(white_edges, left_mask)
    if(right_yellow):
        right_masked_edges = cv2.bitwise_and(yellow_edges, right_mask)
    else:
        right_masked_edges = cv2.bitwise_and(white_edges, right_mask)
    
    line_image = np.copy(image)*0 # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    left_lines = cv2.HoughLinesP(left_masked_edges, rho, theta, threshold, np.array([]),
                                    min_line_length, max_line_gap)
    right_lines = cv2.HoughLinesP(right_masked_edges, rho, theta, threshold, np.array([]),
                                    min_line_length, max_line_gap)
    # Iterate over the output "lines" and draw lines on a blank image
    x=[]
    y=[]
    if(left_lines is None):
        print("No left_lines detected! Yellow: "+str(left_yellow))
    else:
        #Extract X and Y values from hough lines
        for line in left_lines:
            for x1,y1,x2,y2 in line:
                x.append(x1)
                y.append(y1)
                x.append(x2)
                y.append(y2)
        #try fitting a 2nd degree polynomial
        poly=np.polyfit(y,x,2)
        #if rank is low fit a 1st gegree polynomial
        if(np.ndim(poly)<2):
            poly=np.polyfit(y,x,1)
            p = np.poly1d(poly)
        #calculate points along the poly in 10px steps
        y = np.arange(ysize*upppery, ysize*(lowery), 10)
        y = np.arange(ysize*upppery, ysize*(lowery), 10)
        x=p(y)
        i=0
        #draw lines
        for xi in x:
            cv2.line(line_image,(int(round(xi)),int(round(y[i]))),(int(round(x[i+1])),int(round(y[i+1]))),(255,0,0),10)
            i=i+1
            if i+1==len(x):
                break
    #Do the same for the right side
    x=[]
    y=[]
    if(right_lines is None):
        print("No right_lines detected! Yellow: "+str(right_yellow))
    else:
        
        for line in right_lines:
            for x1,y1,x2,y2 in line:
                #if((abs(x1-x2))<(100*abs(y1-y2))):
                x.append(x1)
                y.append(y1)
                x.append(x2)
                y.append(y2)
        poly=np.polyfit(y,x,2)
        if(np.ndim(poly)<2):
            poly=np.polyfit(y,x,1)
            p = np.poly1d(poly)
        y = np.arange(ysize*upppery, ysize*(lowery), 10)
        y = np.arange(ysize*upppery, ysize*(lowery), 10)
        x=p(y)
        i=0
        for xi in x:
            cv2.line(line_image,(int(round(xi)),int(round(y[i]))),(int(round(x[i+1])),int(round(y[i+1]))),(0,0,255),10)
            i=i+1
            if i+1==len(x):
                break

    # Draw the lines on the original image

    lines_edges = cv2.addWeighted(image, 0.8, line_image, 1, 0) 
    return lines_edges;


# Read videos
#Define input Path
path = "/resources/test_videos"
directory = os.path.dirname(__file__)
directory = directory+path
#Outer loop iterate over videos
for vidName in glob.glob(directory+"/*.mp4"):

    fps=24
    fourcc = cv2.VideoWriter_fourcc('S', 'V', 'Q', '3') 
    init=bool(1)
    print("opening video: "+vidName)
    vidcap = cv2.VideoCapture(vidName)
    success,image = vidcap.read()
    count = 0
    success = True
    while success:
        success,image = vidcap.read()
        if (init):
            writer = cv2.VideoWriter(vidName.replace(".mp4","_processed.mp4"),fourcc, fps, (image.shape[1],image.shape[0]))
            init=bool(0)
        if (success):
            writer.write(findLaneLines(image))
        else:
            writer.release()