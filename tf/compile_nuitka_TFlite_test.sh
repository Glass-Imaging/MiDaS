#!/usr/bin/env bash

python get-hints.py run_tflite_min.py

python nuitka-hints.py --show-modules \
    --plugin-enable=numpy --noinclude-scipy --noinclude-matplotlib --noinclude-scipy \
    --nofollow-import-to=botocore \
    --nofollow-import-to=boto3 \
    --nofollow-import-to=pandas \
    --include-data-file=/Users/tom/anaconda3/lib/python3.7/site-packages/cv2/.dylibs/*=./ \
    run_tflite_min.py

#only for 2.7 nightly release:
#    --include-data-file=/Users/tom/anaconda3/lib/python3.7/site-packages/tensorflow/core/platform/_cpu_feature_guard.so=./tensorflow/core/platform/_cpu_feature_guard.so \