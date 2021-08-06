import tensorflow as tf
import numpy as np

graph_def = tf.compat.v1.GraphDef()
with tf.io.gfile.GFile("model-f6b98070.pb", 'rb') as f:
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

print("Imported model")

model_operations = tf.compat.v1.get_default_graph().get_operations()
input_node = '0:0'
output_layer = model_operations[len(model_operations) - 1].name 
print("Last layer name: ", output_layer)

converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
    "model-f6b98070.pb", input_arrays=['0'], output_arrays=[output_layer] # , input_shapes={"0" : [1, 3, 1024, 1024]}
)
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)

# import tf2onnx

# model_proto, external_tensor_storage = tf2onnx.convert.from_graph_def(graph_def,
#                 input_names=['0:0'], output_names=[output_layer + ':0'], # opset=None,
#                 # custom_ops=None, custom_op_handlers=None, custom_rewriter=None, 
#                 # inputs_as_nchw=None, extra_opset=None,
#                 # shape_override=None, target=None, large_model=False,
#                 # shape_override={"0:0" : [1, 3, 1024, 1024]},
#                 output_path="converted_model.onnx")
