from PIL import Image, ImageOps
image = Image.open('id/test/lee.jpg')

import numpy as np
from keras.models import load_model
from keras import optimizers
import pandas as pd
import json

# test file prep
size = (224,224)
img = ImageOps.fit(image, size, Image.ANTIALIAS)
img = np.reshape(img,[1,224,224,3])

# load label
with open('id/model/label_dict.txt', 'rb') as f:     
	old_labels = json.load(f)     
	labels = {v:k for k, v in old_labels.items()}

# load model
model = load_model('id/model/faceID_VGG16.h5')

# compile model
model.compile(loss='categorical_crossentropy',optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])



# top-5 prediction
Num_Prediction = 5   # Number of people most matching 
label = pd.DataFrame(list(labels.items()), columns = ['ID','Name'])
label['Probability'] = model.predict(img)[0]
label.sort_values(by=['Probability'], axis = 0, ascending = False, inplace = True)
label.reset_index(drop=True, inplace=True)
print(label.iloc[0:Num_Prediction,:])