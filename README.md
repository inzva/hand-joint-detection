# hand-joint-detection

Check our presentation: https://docs.google.com/presentation/d/1rw3Gm_fOz8XKUeUcYJ1HeNCRx0UVpKuhUmYiSOqp7GI/edit?usp=sharing

#### Disclaimer: This repo is work in progress. Feel free to open issues. We are open to suggestions.

The project aims on detecting hand joint locations and classify gesture to take various actions on the OS, such as moving the mouse pointer.

#### For the date 26.10.2019, we are still working on:
- Debugging our gesture model, which currently does not work in a correct way.
- Smooth the mouse movements.
- Polish the filenames etc.

#### Here, you can find the explanations for the file and folder names:
- model: You should download keypoint detector's[2] weights and paste them in this directory.
- KeypointDetector.py[2]: to run the application, you should run this script.
- action_interface.py: includes the necessary code to move the mouse.
- config.py: Includes configurations flags such as webcam_height, webcam_width, use_gpu etc.
- model.h5 , model.json: The gesture model architecture and weights that we have developed.
- model.py: Includes code to run Convolutional Pose Machine[1] model, which is to run the Keypoint Detector.
- tracking_module.py: Includes some functions such as cropping and padding to track the hand joints.


1: Carnegie Mellon University, Hand Keypoint Detection in Single Images using Multiview Bootstrapping https://arxiv.org/abs/1704.07809

2: -

3: We also plan to use: Victor Dibia, HandTrack: A Library For Prototyping Real-time Hand TrackingInterfaces using Convolutional Neural Networks, https://github.com/victordibia/handtracking .

inzva AI Projects #3 - Hand Joint Detection (Hand Pose Estimation) and Its Applications.
