# visualize_GUI
a GUI design to display the result of neural network

# Environments Requirement
CUDA = 10.2

Python = 3.7.0

PyTorch = 1.6

You can use conda vitual environments to run the script. For the necessary API, you clould use following command to install the tools
   * pip install torch==1.6.0 torchvision==0.7.0
   * pip install vtk
   * pip install pyqt5
   * pip install pyqt5-tools 
   * pip install sklearn
   * pip install open3d

The mentioned API are the basic API. In the training  process,if there is warning that some modul is missing. you could direct use pip install to install specific modul.
For example, if there is no installation of open3d in the process of running of the script, you could direct use pip install open3d to install conrresponding toolkits


# How to run
If you seccussfully train a model, you should save the pth(for pytorch) file under the directory of pipeline, and change the name to 'best.pth'. After that, you can directly use the following command to run the script.
```
python 
```
