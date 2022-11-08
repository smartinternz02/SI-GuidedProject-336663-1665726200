#!/usr/bin/env python
# coding: utf-8

# ## Import The ImageDataGenerator Library

# In[1]:


from keras.preprocessing.image import ImageDataGenerator


# # Image Data Agumentation

# In[6]:


train_datagen = ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)

test_datgen=ImageDataGenerator(rescale=1./255)


# # Loading our data and performing data agumentation

# In[7]:


x_train = train_datagen.flow_from_directory(
    r'D:\project files\Dataset rock classification\Rock_Classification_Dataset\Rock_Classification_Dataset\train_set',
    target_size=(64, 64),batch_size=5,color_mode='rgb',class_mode='categorical')
x_test = test_datgen.flow_from_directory(
    r'D:\project files\Dataset rock classification\Rock_Classification_Dataset\Rock_Classification_Dataset\test_set',
    target_size=(64, 64),batch_size=5,color_mode='rgb',class_mode='categorical')


# # Importing Necessary Libraries

# In[8]:


import numpy as np
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense,Flatten
from tensorflow.keras.layers import Conv2D,MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator


# # Initializing the model

# In[9]:


# Example for initialization
# model=Sequential()


# # Adding CNN Layers

# In[10]:


classifier = Sequential()

classifier.add(Conv2D(32,(3,3),input_shape=(64,64,3),activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Conv2D(32,(3,3),activation='relu'))

classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Flatten())


# In[11]:


classifier.add(Dense(units=128,activation='relu'))
classifier.add(Dense(units=5,activation='softmax'))

classifier.summary()


# # Compiling the model

# In[12]:


classifier.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])


# # Fitting the model

# In[13]:


classifier.fit_generator(
    generator=x_train,steps_per_epoch = len(x_train),
    epochs=30,validation_data=x_test,validation_steps=len(x_test)
)


# # Save the Model

# In[14]:


classifier.save('rock.h5')

model_json=classifier.to_json()

with open("model-bw.json","w") as json_file:
    json_file.write(model_json)


# # Test the model

# In[20]:


from tensorflow.keras.models import load_model
from keras.preprocessing import image
model=load_model("rock.h5")
13
img=tensorflow.keras.utils.load_img(r'D:\project\uploads\lime6.jpg',grayscale=False,target_size=(64,64))

x=tensorflow.keras.utils.img_to_array(img)
x=np.expand_dims(x,axis=0)
pred=model.predict(x)
pred


# # predicting the output

# In[21]:


index=['blue calcite','limestone','marble','olivine','red crystal']
result=str(index[int(np.where(pred[0]==1.0)[0][0])])
result


# In[ ]:




