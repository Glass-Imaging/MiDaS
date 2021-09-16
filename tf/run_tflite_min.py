# Flex ops are included in the nightly build of the TensorFlow Python package. You can use TFLite models containing Flex ops by the same Python API as normal TFLite models. The nightly TensorFlow build can be installed with this command:
# Flex ops will be added to the TensorFlow Python package's and the tflite_runtime package from version 2.3 for Linux and 2.4 for other environments.
# https://www.tensorflow.org/lite/guide/ops_select#running_the_model

# You must use: tf-nightly
# pip install tf-nightly

import os
import glob
import cv2
import numpy as np
import time

# import tensorflow as tf
import tflite_runtime.interpreter as tflite

height = 768
width = 1024

# model_name = "midas_v21-f6b98070-768x1024.tflite"
model_name='midas_v21-f6b98070-768x1024-float32.tflite'
#image_name = "IMG_0776-768x1024.jpg"
image_name="img1.png"

# input
img = cv2.imread(image_name, cv2.IMREAD_UNCHANGED).astype(np.float32)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 65535.0

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
img = (img - mean) / std

tic = time.perf_counter()
img_resized = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)

tensor = np.float32(np.expand_dims(img_resized, axis=0))

# load model
print("Load model...")
interpreter = tflite.Interpreter(model_path=model_name, num_threads=16)  # , experimental_delegates=tf.lite.experimental.load_delegate()

print("Allocate tensor...")
interpreter.allocate_tensors()
print("Get input/output details...")
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("Get input shape...", input_details)
input_shape = input_details[0]['shape']
print(input_shape)
print(input_details)
print(output_details)

print("Set input tensor...")
interpreter.set_tensor(input_details[0]['index'], tensor)

print("invoke()...")
interpreter.invoke()
print(f"Elapsed time (inference): {time.perf_counter()-tic:0.4f} seconds")

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
print("get output tensor...")
output = interpreter.get_tensor(output_details[0]['index'])
# output = np.squeeze(output)
output = output.reshape(height, width)
# print(output)
prediction = np.array(output)
print("reshape prediction...")
prediction = prediction.reshape(height, width)

# output file
# prediction = cv2.resize(prediction, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
print(" Write image to: output.png")
depth_min = prediction.min()
depth_max = prediction.max()
img_out = (255 * (prediction - depth_min) / (depth_max - depth_min)).astype("uint8")
print("save output image...")
cv2.imwrite("output.png", img_out)

print("finished")
