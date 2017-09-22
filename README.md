# emerald-shiner-detector
Using TensorFlow Object Detection API to detect emerald shiner from experimental images.

All training images are from experiment video clips.

Step 1. Prepare data


Step 2: Train the object detection model

Step 3: run 'object_detection_output_bndbox_coordinates.py' to return images and bounding box coordinates

Step 4: run 'seeFish.py' to use background subtraction and bounding box coordinates to locate fish in each frame

Step 5: run 'smoothOut.py' to smooth out the outliers (incorrectly identified fish coordinates) by replacing with averages of its neighbors
