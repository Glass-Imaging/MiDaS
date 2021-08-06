import tensorflow as tf

# make a converter object from the saved tensorflow file
converter = tf.lite.TFLiteConverter.from_saved_model('midas_v21-f6b98070.pb')

# converter = tf.lite.TFLiteConverter.from_saved_model('midas_v21_small-70d6b9c8.pb')

# converter.target_spec.supported_ops = [
#   tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
#   tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
# ]

# tell converter which type of optimization techniques to use
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.target_spec.supported_types = [tf.float32]
# converter.target_spec.supported_types = [tf.float16]
# to view the best option for optimization read documentation of tflite about optimization
# go to this link https://www.tensorflow.org/lite/guide/get_started#4_optimize_your_model_optional

# convert the model 
tf_lite_model = converter.convert()
# save the converted model 
open('midas_v21-f6b98070.tflite', 'wb').write(tf_lite_model)
