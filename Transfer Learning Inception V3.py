#!/usr/bin/env python
# coding: utf-8

# ## Transfer Learning Inception V3 using Keras

# Please download the dataset from the below url

# In[1]:


from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


# In[81]:


# import the libraries as shown below

from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
#from keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.models import Sequential
import numpy as np
from glob import glob
#import matplotlib.pyplot as plt


# In[82]:


# re-size all the images to this
IMAGE_SIZE = [262, 255]

train_path = 'img3/train'
valid_path = 'img3/validate'


# In[83]:


# Import the Vgg 16 library as shown below and add preprocessing layer to the front of VGG
# Here we will be using imagenet weights

inception = InceptionV3(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)



# In[84]:


# don't train existing weights
for layer in inception.layers:
    layer.trainable = False


# In[85]:


# useful for getting number of output classes
folders = glob('img3/train/*')


# In[86]:


# our layers - you can add more if you want
x = Flatten()(inception.output)


# In[87]:


prediction = Dense(len(folders), activation='softmax')(x)

# create a model object
model = Model(inputs=inception.input, outputs=prediction)


# In[88]:



# view the structure of the model
model.summary()


# In[89]:


# tell the model what cost and optimization method to use
model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)


# In[90]:


# Use the Image Data Generator to import the images from the dataset
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)


# In[108]:


# Make sure you provide the same target size as initialied for the image size
training_set = train_datagen.flow_from_directory('img3/train',
                                                 target_size = (262, 255),
                                                 batch_size = 10,
                                                 class_mode = 'categorical')


# In[109]:


test_set = test_datagen.flow_from_directory('img3/validate',
                                            target_size = (262, 255),
                                            batch_size = 10,
                                            class_mode = 'categorical')


# In[110]:


# fit the model
# Run the cell. It will take some time to execute
r = model.fit_generator(
  training_set,
  validation_data=test_set,
  epochs=20,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set)
)


# In[94]:


import matplotlib.pyplot as plt


# In[111]:


# plot the loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# plot the accuracy
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')


# In[115]:


# save it as a h5 file


from tensorflow.keras.models import load_model

model.save('model_inception.h5')


# In[104]:


#test_batches = "img3/test"


# In[113]:



y_pred = model.predict(test_set)


# In[114]:


y_pred


# In[36]:


import numpy as np
y_pred = np.argmax(y_pred, axis=1)


# In[37]:


y_pred


# In[ ]:





# In[ ]:





# In[38]:


from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


# In[ ]:


model=load_model('model_resnet50.h5')
img_data


# In[11]:


img=image.load_img('Datasets/Test/Coffee/download (2).jpg',target_size=(224,224))


# In[12]:


x=image.img_to_array(img)
x


# In[13]:


x.shape


# In[14]:


x=x/255


# In[15]:


import numpy as np
x=np.expand_dims(x,axis=0)
img_data=preprocess_input(x)
img_data.shape


# In[16]:


model.predict(img_data)


# In[17]:


a=np.argmax(model.predict(img_data), axis=1)


# In[102]:


a==1


# In[18]:


import tensorflow as tf


# In[19]:


tf.__version__


# In[ ]:




