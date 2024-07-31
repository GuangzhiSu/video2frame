# -----------------------------
# Cut black margin for surgical video
# Copyright (c) CUHK 2021. 
# IEEE TMI 'Temporal Relation Network for Workflow Recognition from Surgical Video'
# -----------------------------


import cv2
import os
import numpy as np
import PIL
from PIL import Image


source_path = "./"  # original path
save_path = "./frames2/"  # save path


'''
ensure that the resulting image is focused on the main content, making it easier for subsequent processing steps
lead to faster and more efficient algorithms.
'''
def change_size(image):
    # converts the input image from BGR color space to grayscale。
    binary_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #applies a threshold to the grayscale image. Pixels with values less than 15 are set to 0 (black), and those with values equal to or greater than 15 are set to 255 (white).
    _, binary_image2 = cv2.threshold(binary_image, 15, 255, cv2.THRESH_BINARY)
    #applies a median blur with a kernel size of 19 to remove noise from the binarized image.
    #The median blur smooths the binary image, reducing noise and small artifacts. This makes the image cleaner and the boundaries of objects more distinct.
    binary_image2 = cv2.medianBlur(binary_image2, 19)  # filter the noise, need to adjust the parameter based on the dataset
    #iterates over the binarized image, recording the coordinates of non-zero pixels (i.e., the white pixels) to determine the boundaries of the content.
    x = binary_image2.shape[0]
    y = binary_image2.shape[1]

    edges_x = []
    edges_y = []
    for i in range(x):
        for j in range(10,y-10):
            #non-zero pixles
            if binary_image2.item(i, j) != 0:
                edges_x.append(i)
                edges_y.append(j)
    
    if not edges_x:
        return image

    left = min(edges_x)  # left border
    right = max(edges_x)  # right
    width = right - left  
    bottom = min(edges_y)  # bottom
    top = max(edges_y)  # top
    height = top - bottom  

    pre1_picture = image[left:left + width, bottom:bottom + height]  

    #print(pre1_picture.shape) 
    
    return pre1_picture  

#initializes the video number and creates a directory to save the processed images if it doesn't already exist.
Video_num = 0

if not os.path.exists(save_path):
    os.mkdir(save_path) 

#opens the video file and retrieves its frames per second
cap = cv2.VideoCapture(source_path + "cut1.mp4")
#cap.get(cv2.CAP_PROP_FPS) retrieves the FPS (frames per second) of the video.
fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = int(fps / 1)  # calculate the interval to downsample to 1fps

frame_num = 0
saved_frame_num = 0

'''
This loop reads frames from the video. 
If a frame is read successfully, it processes the frame every frame_interval frames (downsampling to 1 FPS). Each frame is resized, cropped, and resized again to 250x250 pixels. The processed frame is then converted to RGB and saved as an image file.
'''

while cap.isOpened():
    #ret is a boolean indicating if the frame was read successfully
    #frame is a NumPy array representing the image.
    ret, frame = cap.read()

    if not ret:
        break
    
    if frame_num % frame_interval == 0:
        img_save_path = os.path.join(save_path, f"{saved_frame_num}.jpg")
        
        '''
        it typically returns (height, width, channels)
        maintain the original proportions of the image when resizing
        The desired height for the resized image is set to 300 pixels.
        '''
        dim = (int(frame.shape[1]/frame.shape[0]*300), 300)
        
        #dim: The desired size for the output image. It is given as a tuple (width, height)
        frame = cv2.resize(frame,dim)
        frame = change_size(frame)
        img_result = cv2.resize(frame,(250,250))

        img_result = cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB)
        #This converts the NumPy array (used by OpenCV) to a PIL Image object.
        img_result = PIL.Image.fromarray(img_result)

        img_result.save(img_save_path)
        saved_frame_num += 1

    frame_num += 1
    cv2.waitKey(1)

cap.release()

cv2.destroyAllWindows()
print("Cut Done")

