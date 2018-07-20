from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras import backend as K
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os
import time

mode = 1 # 0 for single image, 1 for batch
testset_path = "test"
model_path = "smallVGG.model"
labelbin_path = "mlb.pickle"

total_counter = 0
correct_counter = 0
total_time = 0

def predict(img_path):
    
	# load the image
    image = cv2.imread(img_path)
    output = imutils.resize(image, width=400)
 
    # pre-process the image for classification
    image = cv2.resize(image, (96, 96))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    # load the trained convolutional neural network and the multi-label binarizer
    # print("[INFO] loading network...")
    model = load_model(model_path)
    mlb = pickle.loads(open(labelbin_path, "rb").read())

    # classify the input image then find the indexes of the two class
    # labels with the *largest* probability
    # print("[INFO] classifying image...")
    time_start = time.time()
    proba = model.predict(image)[0]
    K.clear_session()

    time_end = time.time()
    time_interval = time_end - time_start

    idxs = np.argsort(proba)[::-1][:2]
    
    is_correct = True
    # loop over the indexes of the high confidence class labels
    for (i, j) in enumerate(idxs):
	    # build the label and draw the label on the image
        label = "{}: {:.2f}%".format(mlb.classes_[j], proba[j] * 100)
        print("  " + label)
        if mlb.classes_[j] not in img_path:
            is_correct = False
        cv2.putText(output, label, (10, (i * 30) + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    global correct_counter
    if is_correct:
        result = "Correct"
        correct_counter += 1
    else:
        result = "Wrong"
    
    print("  Result: {}; Predict Time: {:.2f}s".format(result, time_interval))
    print("  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    global total_time
    total_time += time_interval

    if mode == 0:
        # show the output image
        cv2.imshow("Output", output)
        cv2.waitKey(0)
'''
    # show the probabilities for each of the individual labels
    for (label, p) in zip(mlb.classes_, proba):
        print("{}: {:.2f}%".format(label, p * 100))
'''

if mode == 0:
    image_path = "test/pink_left/w1.jpg"
    predict(image_path)
elif mode == 1:
    for folder in os.listdir(testset_path):
        print("----------------------------------------")
        print("  Testing in " + folder + " folder...")
        current_folder_path = testset_path + "/" + folder
        for file in os.listdir(current_folder_path):
            total_counter += 1
            print("  Testing " + file + " in " + folder + " folder...")
            current_file_path = current_folder_path + "/" + file
            predict(current_file_path)
    print("Accuracy: " + "{0:.0%}, ".format(correct_counter/total_counter) + str(correct_counter) + "/" + str(total_counter) + "; Average time: {:.0f}ms".format(total_time/total_counter*1000))
 