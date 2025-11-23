# SMS-Spam-Detection

## Description

This project consists of a Python pipeline that trains a Machine Learning model with a dataset containing data related to SMS Spaming in order to check if a new input message is SPAM or not.

The pipeline consists of the following steps:

- Data loading
- Data pre-processing
- Creation of the ML model (Logistic Regression)
- Model training and testing
- CPrediction of SPAM for a new input message

## Requirements
- Docker >= v20.10 (tested with v29.0.2)

## Usage
Before running the pipeline, make sure to give yourself permission to use Docker.

Try this:
```
sudo usermod -aG docker $USER
newgrp docker
```
Unless you want to download every tool needed to run the script one by on, you first need to create a Docker image:
```
docker build -t {image name} .
```
### Example
``` 
docker build -t sms .
```

Now that you have a Docker image, run the following command:
```
docker run -it -v $(pwd)/models:/spam-detect/models {image name} '{message to be analyzed inside quotation marks}'
```
### Example
``` 
docker run -it sms "It's a me, Mario"
```
## Optional arguments

### --dataset
Specify a custom training dataset.

#### Requirements

- Place the dataset in the 'datasets' folder.
- The dataset must be in CSV format.
- Provide '--label' and '--text' flags to specify the column names for labels and SMS text.

#### Example
``` 
docker run -it -v $(pwd)/models:/spam-detect/models {image name} '{message to analyze inside quotation marks}' --dataset {name of custom dataset} --label {name of label column} --text {name of SMS text column}
``` 

``` 
docker run -it -v $(pwd)/models:/spam-detect/models sms 'Hi guys!!!' --dataset data-en-hi-de-fr.csv --label label --text text
``` 
### --verbose
Display detailed output, including:

- A glimpse of the raw dataset.
- A glimpse of the pre-processed dataset.
- Confusion matrix.
- Classification report. 
- AUC-ROC Score.

#### Example
```
docker run -it -v $(pwd)/models:/spam-detect/models {image name} '{message to be analyzed inside quotation marks}' --verbose
```

```
docker run -it -v $(pwd)/models:/spam-detect/models sms 'Congratualations!!! You've been selected to gain a brand new iPhone 17 Pro Max!!! Click on the link below!!!' --verbose
```
## References
- [Docker] (https://www.docker.com/)

- [chardet] (https://github.com/chardet/chardet )

- [pandas] (https://pandas.pydata.org/)

- [scikit-learn] (https://scikit-learn.org/stable/) 