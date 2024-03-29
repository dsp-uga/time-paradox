# Team Time Paradox  
Link to [contest page on kaggle](https://www.kaggle.com/c/landmark-retrieval-challenge#evaluation)

# Google Landmark Retrieval

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


Image retrieval is a fundamental problem in computer vision: given a query image, can you find similar images in a large database? This is especially important for query images containing landmarks, which accounts for a large portion of what people like to photograph.

We try to find similar images from a database of images given a query image

## Getting Started

If you follow the below instructions it will allow you to install and run the training or testing.

### Prerequisites

What things you need to install the software and how to install them

- [Python 3.6](https://www.python.org/downloads/release/python-360/)
- [Anaconda](https://www.anaconda.com/) - Python Environment virtualization so that you dont mess up your system environment
- [Keras](https://keras.io/) The best Deep Learning Tool PERIOD ;)
- [Tensorflow](https://www.tensorflow.org/) One of the API used as Backend of Keras

### Installing

#### Anaconda

Anaconda is a complete Python distribution embarking automatically the most common packages, and allowing an easy installation of new packages.

Download and install Anaconda from (https://www.continuum.io/downloads).
The link for Linux,Mac and Windows are in the website.Following their instruction will install the tool.
##### Running Environment

* Once Anaconda is installed open anaconda prompt(Windows/PC) Command Line shell(Mac OSX or Unix)
* Run ```conda env create -f environment.yml``` will install all packages required for all programs in this repository
###### To start the environment 

* For Unix like systems ```source activate gir```

* For PC like systems ```activate gir```

#### Keras

You can install keras using ``` pip ``` on command line
``` sudo pip install keras ```

The `environment.yml` file for conda is placed in [Extra](https://github.com/dsp-uga/team-huddle/tree/master/extra) for your ease of installation this has keras 

#### Tensorflow
Installing Tensorflow is straight forward using ``` pip ``` on command line

* If CPU then  ``` sudo pip install tensorflow ```
* If GPU then ``` sudo pip install tensorflow-gpu ```

The `environment.yml` file for conda is placed in [Extra](https://github.com/dsp-uga/team-huddle/tree/master/extra) for your ease of installation this has tensorflow

#### Downloading the dataset (Optional)

If you prefer to download the dataset rather than online
The code is present in extra/downloadfiles.py

To Run ``` python downloadfiles.py ``` This will download the whole data set including training and testing

In Folders ```\Train``` and ```\Test``` respectively


## Running and Training



  - **Required Arguments**
    - arg1: path to index.csv
    - arg2: path to hashes.json (this file will be generated by the system)
    - arg3: path to test.csv

 
## Results

As of the date this is written (April 27th) we are ranked 59 of 132 teams in the competition.

S.No| Configuration | Result
--- | --- | --- 
1  | VGG16 and Kmeans | 0.004



## Authors

* **Nihal Soans** - [nihalsoans91](https://github.com/nihalsoans91)
* **Vamsi Nadella** - [vamsi3309](https://github.com/vamsi3309)
* **Vinay Kumar** - [vinayawsm](https://github.com/vinayawsm)


See also the list of [contributors](https://github.com/dsp-uga/time-paradox/blob/master/CONTRIBUTORS.md) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments and References

* Hat tip to anyone who's code was used
* Udacity 



