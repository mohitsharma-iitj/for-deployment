import numpy as np
import cv2
import tensorflow as tf


def sort_contours(cnts, method="left-to-right"):
	# initialize the reverse flag and sort index
	reverse = False
	i = 0
	# handle if we need to sort in reverse
	if method == "right-to-left" or method == "bottom-to-top":
		reverse = True
	# handle if we are sorting against the y-coordinate rather than
	# the x-coordinate of the bounding box
	if method == "top-to-bottom" or method == "bottom-to-top":
		i = 1
	# construct the list of bounding boxes and sort them from top to
	# bottom
	boundingBoxes = [cv2.boundingRect(c) for c in cnts]
	(cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
		key=lambda b:b[1][i], reverse=reverse))
	# return the list of sorted contours and bounding boxes
	return (cnts, boundingBoxes)


def cut(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    thresh, img_bin = cv2.threshold(img, 128, 255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# Invert the image
    img_bin = 255-img_bin 

    # Defining a kernel length
    kernel_length = np.array(img).shape[1]//80
    
    # A verticle kernel of (1 X kernel_length), which will detect all the verticle lines from the image.
    verticle_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
    # A horizontal kernel of (kernel_length X 1), which will help to detect all the horizontal line from the image.
    hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
    # A kernel of (3 X 3) ones.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # Morphological operation to detect vertical lines from an image
    img_temp1 = cv2.erode(img_bin, verticle_kernel, iterations=3)
    verticle_lines_img = cv2.dilate(img_temp1, verticle_kernel, iterations=3)
    # Morphological operation to detect horizontal lines from an image
    img_temp2 = cv2.erode(img_bin, hori_kernel, iterations=3)
    horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=3)

    # Weighting parameters, this will decide the quantity of an image to be added to make a new image.
    alpha = 0.5
    beta = 1.0 - alpha
    # This function helps to add two image with specific weight parameter to get a third image as summation of two image.
    img_final_bin = cv2.addWeighted(verticle_lines_img, alpha, horizontal_lines_img, beta, 0.0)
    img_final_bin = cv2.erode(~img_final_bin, kernel, iterations=2)
    (thresh, img_final_bin) = cv2.threshold(img_final_bin, 128,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # Find contours for image, which will detect all the boxes
    contours, hierarchy = cv2.findContours(img_final_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Sort all the contours by top to bottom.
    (contours, boundingBoxes) = sort_contours(contours, method="top-to-bottom")
    return contours, boundingBoxes
    
def checkval(img):
    result = img.copy()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20,2))
    remove_horizontal = cv2.morphologyEx(thresh , cv2.RETR_EXTERNAL,horizontal_kernel, iterations =2)
    cnts = cv2.findContours(remove_horizontal,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if  len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(result,[c],-1,(255,255,255),5)

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,20))
    remove_horizontal = cv2.morphologyEx(thresh , cv2.RETR_EXTERNAL,horizontal_kernel, iterations =2)
    cnts = cv2.findContours(remove_horizontal,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if  len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(result,[c],-1,(255,255,255),5)


    hsv1 = cv2.cvtColor(result,cv2.COLOR_BGR2HSV)
    lower_black = np.array([0,0,0])
    upper_black = np.array([130,130,130])
    mask = cv2.inRange(hsv1,lower_black,upper_black)
    resultf = cv2.bitwise_and(img,img,mask=mask)
    re = np.count_nonzero(mask)
    return re,img


def fragment_generator(path1,path2):
    
    img1 = []
    img = cv2.imread(path1)

    print(img.shape)
    x,y,_=img.shape
    img= cv2.resize(img,(int((y/x)*224),224),fx=0.3,fy=0.3,interpolation = cv2.INTER_LINEAR)         
    contours, boundingBoxes = cut(img)
    for c in contours:
        # Returns the location and width,height for every contour
        x, y, w, h = cv2.boundingRect(c)
        if (w > 20 and h > 10) and w > 3*h:

            new_img = img[y:y+h, x:x+w]
            xl,yl,_=new_img.shape
            imgcut= cv2.resize(new_img,(int((yl/xl)*224),224),fx=0.3,fy=0.3,interpolation = cv2.INTER_LINEAR)
            for m in range(0,int(yl/xl)):
                valtochek, imageto = checkval(imgcut[:,m*224:(m+1)*224,:])
                if (1100 < valtochek < 10000 ):
                    gray = cv2.cvtColor(imageto,cv2.COLOR_BGR2GRAY)
                    img1.append(np.array(gray))
            
    img2 = []
    img = cv2.imread(path2)
    print(type(img))
    x,y,_=img.shape
    img= cv2.resize(img,(int((y/x)*224),224),fx=0.3,fy=0.3,interpolation = cv2.INTER_LINEAR)

    contours, boundingBoxes = cut(img)
    for c in contours:
        # Returns the location and width,height for every contour
        x, y, w, h = cv2.boundingRect(c)
        if (w > 20 and h > 10) and w > 3*h:

            new_img = img[y:y+h, x:x+w]
            xl,yl,_=new_img.shape
            imgcut= cv2.resize(new_img,(int((yl/xl)*224),224),fx=0.3,fy=0.3,interpolation = cv2.INTER_LINEAR)
            for m in range(0,int(yl/xl)):
                valtochek, imageto = checkval(imgcut[:,m*224:(m+1)*224,:])
                if (1100 < valtochek < 10000 ):
                    gray = cv2.cvtColor(imageto,cv2.COLOR_BGR2GRAY)
                    img2.append(np.array(gray))  
    imges = []
    for i in img1:
        for j in img2:
            imges.append([i,j])
    return np.array(imges)


def prediction2(model,path1,path2):
    proba = []
    y_preds = []
    ids = []
    notx = 0
    img1 = []
    index = []
    
        
    #     plt.imshow(image1)
    #     plt.show()
    #     plt.imshow(image2)
    #     plt.show()
    generated_frag = fragment_generator(path1,path2)
   

    y_pred =  model.predict([generated_frag[:,0],generated_frag[:,1]])
# #     print(y_pred.T)
#     print(y_pred.shape)
#     for i in index:
#         if (i[0] == i[1]):
#             y_preds.append(0)
#             notx += 1
#             proba.append(0)
#         else:
#             if(np.mean(y_pred[i[0]:i[1],0])>.5):
#                 y_preds.append(1)
#             else:
#                 y_preds.append(0)
#             proba.append(np.mean(y_pred[i[0]:i[1],0]))
        
        
                             
        
#         if img1.shape[0] == 0:
#             proba.append(0)
#             y_pred = [0,0]
#             notx += 1
#         else:
#             y_pred = model.predict([img1[:,0],img1[:,1]])
#             proba.append(np.mean(y_pred))

#         if(np.mean(y_pred) > .5):
#             y_preds.append(1)
#         else:
#             y_preds.append(0)
    return np.mean(y_pred)
# model2 = tf.keras.models.load_model("D:/Writer_handwriting_classifier/vgg16_model/archive/vgg16_nn_data.h5")
# model2.summary()
# print(prediction2(model2,"D:/Writer_handwriting_classifier/dataset/dataset/train/P172/A1.jpg","D:/Writer_handwriting_classifier/dataset/dataset/train/P172/B0.jpg"))



