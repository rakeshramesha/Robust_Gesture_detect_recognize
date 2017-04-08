'''
@author: Rakesh M R
2016
'''
import numpy as np
import cv2
import math
from cv2 import imshow


# intensities to be considered 'skin'
lower = np.array([0, 48, 80], dtype = np.uint8)
upper = np.array([25, 255, 255], dtype = np.uint8)

#input camera taken as default webcam
camera = cv2.VideoCapture(0)
_,frame=camera.read()
fcount=0
count_defects = 0
avg2 = np.float32(frame)
while True:
    # grabing the current frame
    (grabbed, frame) = camera.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (11, 11), 0)
    temframe=frame
    check=0
    if (fcount<500):
        cv2.accumulateWeighted(temframe,avg2,0.005)
        backmask = cv2.convertScaleAbs(avg2)
        grayback = cv2.cvtColor(backmask, cv2.COLOR_BGR2GRAY)
        grayback = cv2.GaussianBlur(grayback, (7, 7), 0)
        
        grayback=cv2.threshold(grayback, 125,255 , cv2.THRESH_BINARY)[1]
        grayback=cv2.bitwise_not(grayback)
    gray=cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY)[1]
    gray=cv2.bitwise_not(gray)    
    
    backsub=cv2.bitwise_xor(gray, grayback)     
    
    backsub = cv2.dilate(backsub, None, iterations=2)
    
    cv2.imshow("mask", backsub)
    temframe = cv2.bitwise_and(temframe, temframe, mask = backsub)  
    imshow('background removed', temframe)       
    
    converted = cv2.cvtColor(temframe, cv2.COLOR_BGR2HSV)
    skinMask = cv2.inRange(converted, lower, upper)
        
 
    # apply a series of erosions and dilations to the mask
    # using an elliptical kernel 
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    # patching seperated fingers
    skinMask = cv2.morphologyEx(skinMask, cv2.MORPH_CLOSE, kernel) 
    # bluring the mask to help remove noise, then apply the
    # mask to the frame
    skinMask = cv2.GaussianBlur(skinMask, (7, 7), 0)
    
    cv2.waitKey(1)    
    areaArray = []    
       
    fcount=fcount+1
    
    contours, _ = cv2.findContours(skinMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        M = cv2.moments(c)
        cX = int(M["m10"]/M["m00"])
        cY = int(M["m01"]/M["m00"])
        if area>1000:
        #print area and validate minimum size of target contour
            areaArray.append(area)
        else:
            areaArray.append(0)

    #sorting the array by area
    sorteddata = sorted(zip(areaArray, contours), key=lambda x: x[0], reverse=True)

    #find the second or first largest contour 
    try:
        secondlargestcontour = sorteddata[0][1]
        
        #print secondlargestcontour
    except IndexError:  
        try:
            firstlargestcontour= sorteddata[0][0] 
            secondlargestcontour=firstlargestcontour
            #print secondlargestcontour
        except IndexError:
            check=1

    if (fcount>1000 or check==1):
        fcount=0
        avg2 = np.float32(frame)
        print "restarting background removal"
    hull = cv2.convexHull(secondlargestcontour,returnPoints = False)
    defects = cv2.convexityDefects(secondlargestcontour,hull)
    prevcountdefects=count_defects  
    count_defects = 0
    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]
        start = tuple(secondlargestcontour[s][0])
        end = tuple(secondlargestcontour[e][0])
        far = tuple(secondlargestcontour[f][0])
        #depth=tuple(secondlargestcontour[d][0])
        #cv2.line(frame,start,end,[0,255,0],2)
        #cv2.circle(frame,far,5,[0,0,255],-1)

        a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
        c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
        angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57
        #valid defect and dist from centroid 
        if angle <= 90:
            if (((cX-far[0])**2 + (cY-far[0])**2 )<100000):
                count_defects += 1
                cv2.circle(frame,far,5,[0,0,255],-1)
            cv2.line(frame,start,end,[0,255,0],2)
    #if(fcount%3==0): 
    #averaging to reduce errors   
    count_defects =((prevcountdefects+count_defects )/2)
    
    if count_defects == 1:
        cv2.putText(frame,"1", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
    elif count_defects == 2:       
        cv2.putText(frame, "2", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
    elif count_defects == 3:
        cv2.putText(frame,"3", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
    elif count_defects == 4:
        cv2.putText(frame,"4", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
    elif count_defects == 5:
        cv2.putText(frame,"5", (50,50),\
                    cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
    else:
        cv2.putText(frame,"0", (50,50),\
                        cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
    
    #drawing contours
    if(check!=1 and cv2.contourArea(secondlargestcontour)>1000):
        x, y, w, h = cv2.boundingRect(secondlargestcontour)
        cv2.drawContours(frame, secondlargestcontour, -1, (255, 255, 255), 3)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,255), 2)
        cv2.circle(frame, (cX, cY), 5, (0, 255, 255), -1)
            
        #cv2.imshow('finger detect',frame)
        #cv2.waitKey(1)
    
    skin = cv2.bitwise_and(temframe, temframe, mask = skinMask)
    
    # showing the skin in the image along with the mask
    cv2.imshow("skin detect", np.hstack([frame, skin]))   
        
    
    
    # press 'q' key to stop the loop
    if cv2.waitKey(10) & 0xFF == ord("q"):
        break
 
camera.release()
cv2.destroyAllWindows()
