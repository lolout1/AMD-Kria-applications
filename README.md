# AMD-Kria-applications         https://youtu.be/ldR44BJjO9E

This repository will be an ongoing project of mine showcasing a variety of applications I have built on the Kria KV260. My first project was a hardware accelerated custom built facial recognition app which takes input from a usb webcam or .mp4 file via Gstreamer/VVAS and processes it to recognize my face. The output will be shown on DP/HDMI/X-11 forwarding.

The youtube link above shows how the program could be run on the KRIA KV260. The monitor running the script is connected to the KV260 via serial port while the monitor displaying the output of the script is connected to the Kria KV260 via hdmi. The video footage seen is the footage from a webcam connected to the Kria KV260 via USB 3.0 which is then processed via the facial recognition application to detect any faces and label them. The names and pictures of the people detected in my program was designed to be user-configurable when running the command to start the script.


To run the application, enter the following command in the terminal when all the necessary files are in your current directory.

./facialdetect_recog /dev/video2 3.jpg -f 1920x1080 or  ( ./executableName + /video_format # (ex. mp4 file or webcam input) + picturesofPeopletoRecognize.jpg + ( -f + Monitor Resolution) ex. -f 1080x720 or ex. -f 1920x1080 ).

The output will be a videostream displayed to the HDMI/display port connected to the KV260 where the fps will be +1/2 fps within the max fps capabilities of your webcam. This means latency is close to none and throughput is near perfect for webcam configurations up to 60 fps. 


To compare with other facial recognition applications I have included a benchmarks folder. [(https://github.com/lolout1/AMD-Kria-applications/blob/main/bench/README.md)](https://github.com/lolout1/AMD-Kria-applications/tree/main/bench)

It is worth nothing the KV260 can also run pre built facial recognition libraries such as dlib or faceNET from python. However, even on an actual computer with an above average GPU, the fps experiences a harsh penalty due to the ineficciency of these libraries. Running on Pynq via Jupyter Notebook on my Kria KV260, I experienced at lowest .5 fps peaking at 3-5 fps after siginificant optimization. It is clear to me I can do better if I utilize the various acceleration options.

To start, I decided to train, quantize, and compile my model to match the hardware (DPU IP/overlay) and software platform running on the Petalinux Environment on the Kria KV260. After, quantizastion and cross-compilation, deployment of the model to the Kria KV260 Petalinux 2022.2+ environment is achieved by writing one or more scripts to achieve goals such as inference, pre, and post processing which will then be compiled via CMAKE into a buildable executeable. In 'FaceRecognitionFinal' I have included my inference script including pre/post processing where cosine simularity is used to calculate a similarity value where a label will be generated for the face according to a thresh hold value. 

The Kria KV260 mounts the Kria-26 SOM to a development board providing a ready to use hardware set up without the need of purchasing custom hardware. The Xilinx Zynq UltraScale+ MPSoC combines a 6 real-time ARM Cortex processors running embedded linux with FPGA fabric enabling acceleration/optimization for a variety of applications. AMD/Xilinx provides a comprehensive toolchain to aid you in the development and deployment of applications. 

- Vitis AI
  Provides a comprehensive set of tools for optimizing, quantizing, compiling, and deploying AI models on Xilinx FPGAs. 
- VVAS
  Framework for video analytics capabiltiies such as utilizing Gstreamer which is much faster at communicating with the I/O peripherials compared to Open CV for example.
-   Vivado Platform
  Generates custom overlay, allows you to configure and customize DPU/any Hardware
  Vitis software toolchain
  
