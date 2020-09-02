import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
from models import unet


model_path = '../gs-models/0.7711/'
loaded = tf.saved_model.load(model_path)

model = unet((None, None, 36), initial_exp=4, n_classes=5)
ws = []
for i, p in zip(loaded.variables, model.variables):
    ws.append(i.numpy())
model.set_weights(ws)
model.save('fcnn.h5')
