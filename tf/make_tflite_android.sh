# exit when any command fails
set -e

python make_onnx_model.py
onnx-tf convert -i "midas_v21-f6b98070.onnx" -o  "midas_v21-f6b98070.pb" --device CUDA
python convert_to_tflite.py
mv midas_v21-f6b98070.tflite midas_v21-f6b98070-768x1024-qt-cu.tflite