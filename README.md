This is the README.

A note on compilation on Apple M1 Pro. 

Apple chips now have a new architecture (arm64), and programs compiled for x86-64 will no longer work on them. Here is how I navigated this to succesfully compile CLASS and its python wrapper.

For context, I was using miniconda3 to manage my environments. 

Compilation does not work straight out of the box because the default C compiler on the Apple M1 is clang, which does not support openmp. 

In order to get around this, you can install gcc using a system package manager like homebrew, i.e. run `brew install gcc`. This installs gcc in /opt/homebrew/Cellar/path/to/gcc. 
The specific compiler you want is gcc-11. In order to specify that CLASS use this to compile, set the CC variable in Makefile equal to this path. Now, create a new conda environment running:

conda create --name classy-forge python=3.9

conda activate classy-forge

Now we want to install some packages that classy depends on. These packages (like numpy and cython) contain compiled python code (.pyc files). In order to have everything play nicely, we want this code to be compiled using the same compiler as above. Run:

env CC=/opt/homebrew/Cellar/gcc/11.2.0_3/bin/gcc-11 conda install numpy scipy cython

This specifies the compiler conda will use. The reason we need to do this is because different compilers will name functions differently in .o files, which leads to linker errors (the python wrapper things a function is named differently than the c code thinks it is). 

Then cd into the class directory and run make clean if you have built before. Then run make and wait for compilation to finish. cd into the python directory and install the classy package. Once again we need to make sure conda uses the right compiler when building the package:

env CC=/opt/homebrew/Cellar/gcc/11.2.0_3/bin/gcc-11 python setup.py install --user

At this stage everything should be installed correctly. Run

python -c "import classy as Class"

To check. 

A few notes:

1. You can check which architecture an executable has been compiled for by running 

file path/to/exec

A common error is having a python version that has been compiled for the wrong architecture 
attempting to use a .so file that is compiled for the other one. By default, since it uses clang to install, conda will install arm64 versions. You can change this by exporting a conda env variable before setting up the environment (change it to x86).  
