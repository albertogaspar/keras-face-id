# Keras Face Recognition

Simple face recognition with your webcam using Keras and OpenCV.

## Models used
The face recognition algorithm is described in [1] (NN2). No triplet loss is
used for trainig, instead I used contrastive loss. 
I plan to make some experiment with the ResNet version of the Inception
network in the future.

The face detection algorithm to obtain images from the webcam is the one described in [2]. 
Standard implemenation from OpenCV has been used.

## How to Use
- faceid.py: Train model on some dataset.  
- capture_webcam.py: Obtain some image via webcam and precompute embeddings of the subject of interest 
(i.e. the ground truth).
- faceid_webcam: face id via webcam of the subject being recorded. 
Ouput is positive (same identity as the one defined in the step before) or negative.
 
## References
[1] [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/abs/1503.03832)

[2] [Haar Feature-based Cascade Classifiers with OpenCV](https://docs.opencv.org/3.4.1/d7/d8b/tutorial_py_face_detection.html)