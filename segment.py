import cv2
import imutils
import numpy as np

bg = None

#Find the runnig naverage over the background

def run_avg(image, aWeight) :
    global bg
    #Initiliaze the background
    if bg is None:
            bg = image.copy().astype("float")
            return
    #Compute the weighted average
    cv2.accumulateWeighted(image, bg, aWeight)

#Segment the region of hand in the image
def segment(image, threshold=25) :
    global bg
    #Find the absolute difference between background and current frame
    diff = cv2.absdiff(bg.astype("uint8"), image)
    #Threshold the different image so that we get the foreground
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]
    #get the contours in the thresholded image
    (cnts, _) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #Return None if no contours detected
    if len(cnts) == 0:
        return
    else:
        #Get the max contour which is the hand
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)

#Main function
if __name__ =="__main__" :
    #Initialize weight for running average
    aWeight = 0.1
    #Get the reference to the webcam
    camera = cv2.VideoCapture(0)
    #Region of interest coordinates
    top, right, bottom, left = 0, 100, 400, 790
    #Number of frames
    num_frames = 0
    #Keep looping until interrupted
    while(True):
        #Get current frame
        (grabbed, frame) = camera.read()
        #Resize the frame
        frame = imutils.resize(frame, width=700)
        #Flip the frame to prevent mirrior view
        frame = cv2.flip(frame, 1)
        #Clone the frame
        clone = frame.copy()

        # get the height and width of the frame
        (height, width) = frame.shape[:2]

        # get the ROI
        roi = frame[top:bottom, right:left]

        # convert the roi to grayscale and blur it
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # to get the background, keep looking till a threshold is reached
        # so that our running average model gets calibrated
        if num_frames < 30:
            run_avg(gray, aWeight)
        else:
            # segment the hand region
            hand = segment(gray)
            # check whether hand region is segmented
            if hand is not None:
                # if yes, unpack the thresholded image and
                # segmented region
                (thresholded, segmented) = hand

                # draw the segmented region and display the frame
                cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
                cv2.imshow("Thesholded", thresholded)

        # draw the segmented hand
        cv2.rectangle(clone, (left, top), (right, bottom), (0, 255, 0), 2)

        # increment the number of frames
        num_frames += 1

        # display the frame with segmented hand
        cv2.imshow("Video Feed", clone)

        # observe the keypress by the user
        keypress = cv2.waitKey(1) & 0xFF

        # if the user pressed "q", then stop looping
        if keypress == ord("q"):
            break

    # free up memory
camera.release()
cv2.destroyAllWindows()