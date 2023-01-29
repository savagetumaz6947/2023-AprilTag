# AprilTag-2023

This repo is for AprilTag detection on a Raspberry Pi using WPILibPi.

## The config.json file
This file is where you change the settings for the AprilTag detector. The settings are pretty straghtforward. Be careful that the camera properties match the ones on the WPILibPi interface. I suggest copying the camera properties from the WPILibPi interface and pasting them into the config.json file.

## Setup on your computer
You must first calculate the camera intrinsics of your camera on your personal computer (NOT ON THE RASPBERRY PI).

1. Use the `requirements.txt` file to install the necessary python packages using `pip install -r requirements.txt`.
1. Tweak the settings of your camera in `config.json`, then run `python3 setup/0_set_camera_properties`.
1. Download and print this image of a checkerboard: [https://github.com/opencv/opencv/blob/4.x/doc/pattern.png?raw=true](https://github.com/opencv/opencv/blob/4.x/doc/pattern.png?raw=true), provided by OpenCV.
1. Run `python3 setup/1_take_pictures.py`. This program takes pictures from your camera. You must hold the checkerboard in front of the camera and press the spacebar to take a picture. Take at least 20 pictures. The pictures will be saved in the folder specified in `config.json`. You can take as many pictures as you want, then press `q` to quit.
    - See the `images_for_calibraton/samples` folder for an example of what the pictures should look like.
1. Adjust the NxN settings of the checkerboard in `config.json` to match the checkerboard you printed. Count the inner corners of the checkerboard, not the outer corners. You don't need to change this setting if you used the checkerboard above.
1. Run `python3 setup/2_calculate_intrinsics.py`. This program will calculate the camera intrinsics (camera matrix and distortion coefficients) and save them in the files specified in `config.json`.
1. Run `python3 detection.py`. Two windows will pop up. The first window is the original camera stream. The second window is the camera stream with the AprilTags boxed up. If the detection is working, you should see a green box around the AprilTag. If the detection is not working, you may need to tweak the settings in `config.json` and run this section again.

## Setup on the Raspberry Pi
You will need to use SSH to connect to the Raspberry Pi. You can use wpilibpi.local as the mDNS address. The default username is `pi` and the default password is `raspberry`.

### On the web interface
1. Upload the following files: `_helper.py`, `config.json`, `detection.py`, `mtx.npy`, `dist.npy`, and `requirements.txt` using the web interface. (You could also use the `scp` command to upload the files.)
1. Upload the `main_wpi.py` file as the main robot program, and change the application to `Uploaded Python file`.

### Using SSH
1. Run `pip install -r requirements.txt` to install the necessary python packages.

Now go back to the camera stream and logs to see if the program is working.