from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg19 import preprocess_input
# from keras.applications.vgg16 import decode_predictions

from keras.preprocessing import image
from keras.applications.imagenet_utils import  decode_predictions
import numpy as np

model = VGG19(weights='imagenet', include_top=True)

x = []
for i in xrange(1,6):
    print i
    img_path = 'truck (' + str(i) + ').jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    sample = image.img_to_array(img)
    x.append(sample)

x = np.array(x)
print x.shape

# x = np.expand_dims(x, axis=0)
# print x.shape
x = preprocess_input(x)
print x.shape


preds = model.predict(x, verbose=1)

feat = decode_predictions(preds, top=3)
for f in feat:
    print f
    print ''

# im = image.array_to_img(features)
#
#
# from matplotlib import pyplot
#
# pyplot.imshow(im)