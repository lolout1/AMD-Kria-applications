# AMD-Kria-applications

This repository will be an ongoing project of mine showcasing a variety of applications I have built on the Kria KV260. 


My current goal is to fully accelerate my custom built facial recognition app which takes imput from the usb input via Gstreamer/VVAS and processes it to recognize my face. The output will be shown on DP/HDMI/X-11 forwarding.

It is worth nothing the KV260 can run pre built facial recognition libraries such as dlib or faceNET from python. However, even on an actual computer with an above average GPU, the fps experiences a harsh penalty due to the ineficciency of these libraries. Running on Pynq via Jupyter Notebook on my Kria KV260, I experienced at lowest .5 fps peaking at 3-5 fps after siginificant optimization. It is clear to me I can do better if I utilize the various acceleration options.

To start, I decided to train, quantize, and compile my model to match the hardware (DPU IP and overlay) and software platform running on the Petalinux Environment on the Kria KV260. 






The Kria KV260 mounts the Kria-26 SOM to a development board providing a ready to use hardware set up without the need of purchasing custom hardware. The Xilinx Zynq UltraScale+ MPSoC combines a 6 real-time ARM Cortex processors running embedded linux with FPGA fabric enabling acceleration/optimization for a variety of applications. AMD/Xilinx provides a comprehensive toolchain to aid you in the development and deployment of applications. 

- Vitis AI
  Provides a comprehensive set of tools for optimizing, quantizing, compiling, and deploying AI models on Xilinx FPGAs. 
- VVAS
  Framework for video analytics capabiltiies such as utilizing Gstreamer which is much faster at communicating with the I/O peripherials compared to Open CV for example.
-   Vivado Platform
  Generates custom overlay, allows you to configure and customize DPU/any Hardware
  Vitis software toolchain
  
