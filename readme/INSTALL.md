# Installation

The code was tested on Ubuntu 16.04, with [Anaconda](https://www.anaconda.com/download) Python 3.7, CUDA 10.0, and [PyTorch]((http://pytorch.org/)) v1.0.
It should be compatible with PyTorch >=1.7 and python >=3.7 (you will need to switch DCNv2 version for PyTorch >1.0).
After installing Anaconda:

0. [Optional but highly recommended] create a new conda environment. 

    ~~~
    conda create --name PFTrack python=3.7
    ~~~
    And activate the environment.
    
    ~~~
    conda activate PFTrack
    ~~~

1. Install PyTorch:

    ~~~
    conda install pytorch torchvision -c pytorch
    ~~~
    
2. Install [COCOAPI](https://github.com/cocodataset/cocoapi):

    ~~~
    pip install cython; pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
    ~~~

4. Install the requirements

    ~~~
    pip install -r requirements.txt
    ~~~
    
5. Compile deformable convolutional (from [DCNv2](https://github.com/CharlesShang/DCNv2/)).

    ~~~
    cd $PFTrack_ROOT/src/lib/model/networks/
    # git clone https://github.com/CharlesShang/DCNv2/ # clone if it is not automatically downloaded by `--recursive`.
    cd DCNv2
    ./make.sh
    ~~~