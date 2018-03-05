# emerald-shiner-detector
Using TensorFlow Object Detection API to detect emerald shiner from experimental images.

All training images are from experiment video clips.

Step 1. Run 'img_pre.py' to preprocess raw data in video clips

Step 2: Train the object detection model (Faster R-CNN + Resnet101 on coco)

Step 3: run 'object_detection_output_bndbox_coordinates_scores.py' to return images and bounding box coordinates

Step 4: run 'img_2_fishCentroid.py' to compute fish mass center in each frame.
