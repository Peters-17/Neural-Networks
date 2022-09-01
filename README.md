# Neural-Networks
Score 100/100
# Assignment Goals
• Get Pytorch set up for your environment.
• Familiarize yourself with the tools.
• Implementing and training a basic neural network using Pytorch.
• Happy deep learning :)
# Summary
Home-brewing every machine learning solution is not only time-consuming but potentially error-prone. One of
the reasons we’re using Python in this course is because it has some very powerful machine learning tools. Besides
common scientific computing packages such as SciPy and NumPy, it’s very helpful in practice to use frameworks
such as Scikit-Learn, TensorFlow, Pytorch, and MXNet to support your projects. The utilities of these frameworks
have been developed by a team of professionals and undergo rigorous testing and verification.
In this homework, we’ll be exploring the Pytorch framework. Please complete the functions in the template
provided: intro pytorch.py.
# Part 1: Setting up the Python Virtual Environment
In this assignment, you will familiarize yourself with the Python Virtual Environment. Working in a virtual environment is an important part of working with modern ML platforms, so we want you to get a flavor of that through
this assignment. Why do we prefer virtual environments? Virtual environments allow us to install packages within
the virtual environment without affecting the host system setup. So you can maintain project-specific packages in
respective virtual environments.
We suggest that you use the CS lab computers for this homework. You can also work on your personal system
for the initial development, but finally, you will have to test your model on the CSL lab computers. Find more
instructions: How to access CSL Machines Remotely
The following are the installation steps for Linux (CSL machines are recommended). You will be working on
Python 3 ( instead of Python 2 which is no longer supported). Read more about Pytorch and Python version here.
To check your Python version use:

python -V or python3 -V

If you have an alias set for python=python3 then both should show the same version (3.x.x)

## Step 1: For simplicity, we use the venv module (feel free to use other virtual envs such as Conda).
To set up a Python Virtual Environment named Pytorch:
python3 -m venv /path/to/new/virtual/environment
For example, if you want to put the virtual environment in your home directory:
python3 -m venv --system-site-packages Pytorch
(Optional: If you want to learn more about Python virtual environments, a very good tutorial can be found here.)
## Step 2: Activate the environment:
Let’s suppose the name of our virtual environment is Pytorch (you can use any other name if you want). You can
activate the environment by the following command:
## Step3: From your virtual environment shell, run the following commands to upgrade pip and install the CPU
version of Pytorch 1.10:
pip install --upgrade pip
pip install torch==1.10.2+cpu torchvision==0.11.3+cpu torchaudio==0.10.2+cpu
-f https://download.pytorch.org/whl/cpu/torch_stable.html
You can check the version of the packages installed using the following command:
