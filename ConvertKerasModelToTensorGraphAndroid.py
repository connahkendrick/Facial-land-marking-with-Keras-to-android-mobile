# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 11:39:31 2017

@author: 12102083
"""

import os
import numpy 
import pandas  
from sklearn.utils import shuffle  
from keras.wrappers.scikit_learn import KerasRegressor  
from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt 
import time 
from sklearn.pipeline import Pipeline 

from keras.models import Sequential  
from keras.layers import Dense, Activation 
from keras.layers import Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import SGD 
from keras.models import model_from_json   
from keras import backend as K 
from keras.models import model_from_config   
from keras.utils.np_utils import convert_kernel
import tensorflow as tf
#from tensorflow_serving.session_bundle import exporter 
# In[ ]: 
model = model_from_json(open(r'C:\Users\12102083\Documents\Visual Studio 2015\Projects\WebcamFaceDetection\WebcamFaceDetection\images\MyFaceTests\BaseNet\model32.json',).read())  
model.load_weights(r'C:\Users\12102083\Documents\Visual Studio 2015\Projects\WebcamFaceDetection\WebcamFaceDetection\images\MyFaceTests\BaseNet\model64_weights300epocsGD.h5')  
output_tensor = model.output  
tensor = k.get_
tf.train.write_graph(output_tensor, 'TensorGraphs', 'train.pb')

# In[ ]:

import keras
import tensorflow
from keras import backend as K
from tensorflow.contrib.session_bundle import exporter
from keras.models import model_from_config, Sequential
from tensorflow.python.framework.graph_util import convert_variables_to_constants 

print("Loading model for exporting to Protocol Buffer format...")
model_path = 'temp.json'
model = model_from_json(open('temp.json',).read())
model.load_weights('temp.h5') 

K.set_learning_phase(0)  # all new operations will be in test mode from now on
ops = []
for layer in model.layers:
   print("layer name = " + layer.name)
   if layer.__class__.__name__ in ['Convolution1D', 'Convolution2D', 'Convolution3D', 'AtrousConvolution2D']:
      original_w = K.get_value(layer.W)
      converted_w = convert_kernel(original_w)
      ops.append(tf.assign(layer.W, converted_w).op) 
      print("Hi")
      
K.get_session().run(ops)

sess = K.get_session()

#_W = W.eval(sess)
#_b = b.eval(sess) 

#W_2 = tf.constant(_W, name="constant_W")
#b_2 = tf.constant(_b, name="constant_b")

# serialize the model and get its weights, for quick re-building
#config = model.get_config()
#weights = model.get_weights()

# re-build a model where the learning phase is now hard-coded to 0
#new_model = Sequential.from_config(config)
#new_model.set_weights(weights)
#new_model.summary()
export_path = "TensorGraphs\\simple.pb"  # where to save the exported graph
export_version = 1  # version number (integer)

#saver = tensorflow.train.Saver(sharded=True)
#model_exporter = exporter.Exporter(saver)
#signature = exporter.classification_signature(input_tensor=model.input, scores_tensor=model.output)
#model_exporter.init(sess.graph.as_graph_def(), default_graph_signature=signature)
#model_exporter.export(export_path, tensorflow.constant(export_version), sess) 

#[print(n.name) for n in tf.get_default_graph().as_graph_def().node]
tf.global_variables_initializer()#initialize_all_variables(); 
minimal_graph = convert_variables_to_constants(sess, sess.graph_def, ["dense_3_b/Assign"]) # output
tf.train.write_graph(sess.graph_def, './tmp/beginner-export','beginner-graph.pb', as_text=False)
#tf.train.write_graph(minimal_graph, '.', 'minimal_graph.proto', as_text=False)
#tf.train.write_graph(minimal_graph, '.', 'minimal_graph.txt', as_text=True)
    
# In[ ]: 
    
import keras
import tensorflow
from keras import backend as K
from tensorflow.contrib.session_bundle import exporter
from keras.models import model_from_config, Sequential

print("Loading model for exporting to Protocol Buffer format...")
model_path = 'temp.h5'
model = keras.models.load_model(model_path)

K.set_learning_phase(0)  # all new operations will be in test mode from now on
sess = K.get_session()

# serialize the model and get its weights, for quick re-building
config = model.get_config()
weights = model.get_weights()

# re-build a model where the learning phase is now hard-coded to 0
new_model = Sequential.from_config(config)
new_model.set_weights(weights)

export_path = "TensorGraphs//simple.pb"  # where to save the exported graph
export_version = 1  # version number (integer)

saver = tensorflow.train.Saver(sharded=True)
model_exporter = exporter.Exporter(saver)
signature = exporter.classification_signature(input_tensor=model.input, scores_tensor=model.output)
model_exporter.init(sess.graph.as_graph_def(), default_graph_signature=signature)
model_exporter.export(export_path, tensorflow.constant(export_version), sess)