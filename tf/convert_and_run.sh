# exit when any command fails
set -e

python make_onnx_model.py
onnx-tf convert -i "midas_v21-f6b98070.onnx" -o  "midas_v21-f6b98070.pb"
python convert_to_tflite.py
python run_tflite.py 
