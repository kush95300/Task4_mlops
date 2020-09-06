#!/usr/bin/env python
# coding: utf-8

# In[1]:


### Testing Our Trained Model


# In[2]:


from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

img_rows, img_cols = 224, 224

classifier = load_model('celebrity_face_recognition_model_weight.h5')

validation_data_dir = 'Dataset/test/'

# images converted as input images 
validation_datagen = ImageDataGenerator(rescale=1./255)
 
# set our batch size (typically on most mid tier systems we'll use 16-32)
batch_size = 32
 
validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode='categorical')


# ### Testing our classifer on some test images

# In[4]:


import os
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

celebrity_dict = {"[0]": "Aamir_khan",
                  "[1]": "Abhay_Deol",
                  "[2]": "Abhishek_Bachchan",
                  "[3]": "Aftab_Shivdasani",
                  "[4]": "Aishwarya_Rai ",
                  "[5]": "Ajay_Devgn",
                  "[6]": "Akshay_Kumar",
                  "[7]": "Akshaye_Khanna",
                  "[8]": "Alia_Bhatt",
                  "[9]": "Ameesha_Patel",
                 "[10]": "Amitabh_Bachchan",
                  "[11]": "Amrita_Rao",
                  "[12]": "Amy_Jackson",
                  "[13]": "Anil_Kapoor",
                  "[14]": "Anushka_Sharma ",
                  "[15]": "Anushka_Shetty",
                  "[16]": "Arjun_Kapoor",
                  "[17]": "Arjun_Rampal",
                  "[18]": "Arshad_Warsi",
                  "[19]": "Bhumi_Pednekar",
                 "[20]": "Bipasha_Basu",
                  "[21]": "Bobby_Deol",
                  "[22]": "Deepika_Padukone",
                  "[23]": "Disha_Patani",
                  "[24]": "Emraan_Hashmi ",
                  "[25]": "Esha_Gupta",
                  "[26]": "Farhan_Akhtar",
                  "[27]": "Govinda",
                  "[28]": "Hrithik_Roshan",
                  "[29]": "Huma_Qureshi",
                  "[30]": "Ileana"}

       

def draw_test(name, pred, im):
    celebrity = celebrity_dict[str(pred)]
    BLACK = [0,0,0]
    expanded_image = cv2.copyMakeBorder(im, 80, 0, 0, 100 ,cv2.BORDER_CONSTANT,value=BLACK)
    cv2.putText(expanded_image, celebrity, (20, 60) , cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,255), 2)
    cv2.imshow(name, expanded_image)

def getRandomImage(path):
    """function loads a random images from a random folder in our test path """
    folders = list(filter(lambda x: os.path.isdir(os.path.join(path, x)), os.listdir(path)))
    random_directory = np.random.randint(0,len(folders))
    path_class = folders[random_directory]
    print("Actual : "+str(path_class))
    #print("Class - " + celebrity_dict_n[str(path_class)])
    file_path = path + path_class
    #print(file_path)
    file_names = [f for f in listdir(file_path) if isfile(join(file_path, f))]
    random_file_index = np.random.randint(0,len(file_names))
    image_name = file_names[random_file_index]
    return cv2.imread(file_path+"/"+image_name)    

for i in range(0,10):
    input_im = getRandomImage("Dataset/test/")
    input_original = input_im.copy()
    input_original = cv2.resize(input_original, None, fx=0.5, fy=0.5, interpolation = cv2.INTER_LINEAR)
    
    input_im = cv2.resize(input_im, (224, 224), interpolation = cv2.INTER_LINEAR)
    input_im = input_im / 255.
    input_im = input_im.reshape(1,224,224,3) 
    
    # Get Prediction
    res = np.argmax(classifier.predict(input_im, 1, verbose = 0), axis=1)
    # Show image with predicted class
    draw_test("Prediction", res, input_original) 
    print("Predicted : " + celebrity_dict[str(res)] + "\n")
    cv2.waitKey(0)

cv2.destroyAllWindows()


# In[ ]:




