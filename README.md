# Deploy a Sentiment Analysis Model


### Table of Contents

1. [Project Overview](#overview)
2. [Project Outline](#outline)
3. [Installation](#installation)
4. [File Descriptions](#files)
5. [Instructions](#instructions)
6. [Results](#results)
7. [Screenshots](#screenshots)
8. [Licensing, Authors, and Acknowledgements](#licensing)

## Project Overview<a name="overview"></a>

In this project we will construct a simple recurrent neural network for the purpose of determining the sentiment of a movie review using the IMDB data set. We will create this model using Amazon's SageMaker service. In addition, we will deploy our model and construct a simple web app which will interact with the deployed model. Our goal will be to have a simple web page which a user can use to enter a movie review. The web page will then send the review off to our deployed model which will predict the sentiment of the entered review.

![](https://github.com/gauravansal/Deploying-a-Sentiment-Analysis-Model/blob/master/Web%20App%20Diagram.svg)

The diagram above gives an overview of how the various services will work together. On the far right is the model which we trained above and which is deployed using SageMaker. On the far left is our web app that collects a user's movie review, sends it off and expects a positive or negative sentiment in return.

In the middle is where some of the magic happens. We will construct a Lambda function, which you can think of as a straightforward Python function that can be executed whenever a specified event occurs. We will give this function permission to send and recieve data from a SageMaker endpoint.

Lastly, the method we will use to execute the Lambda function is a new endpoint that we will create using API Gateway. This endpoint will be a url that listens for data to be sent to it. Once it gets some data it will pass that data on to the Lambda function and then return whatever the Lambda function returns. Essentially it will act as an interface that lets our web app communicate with the Lambda function.

## Project Outline<a name="outline"></a>

The general outline for SageMaker projects using a notebook instance are -

1. Download or otherwise retrieve the data.
2. Process / Prepare the data.
3. Upload the processed data to S3.
4. Train a chosen model.
5. Test the trained model (typically using a batch transform job).
6. Deploy the trained model.
7. Use the deployed model.

For this project, we will be following the steps in the general outline with some modifications. First, we will not be testing the model in its own step. We will still be testing the model, however, we will do it by deploying our model first and then using the deployed model by sending the test data to it. One of the reasons for doing this is so that we can make sure that our deployed model is working correctly before moving forward.

In addition, we will deploy and use your trained model a second time. In the second iteration we will customize the way that our trained model is deployed by including some of our own code. In addition, our newly deployed model will be used in the sentiment analysis web app.

## Installation<a name="installation"></a>

We will be using the below AWS platform services along with Data Processing & Machine Learning Libraries to implement the project. 
 - Amazon SageMaker service.
 - Amazon S3 service
 - Amazon IAM service
 - Amazon Lambda service
 - Amazon API Gateway service
 - Amazon CloudWatch service 
 - Data Processing & Machine Learning Libraries: NumPy, SciPy, Pandas, PyTorch, NLTK.

## File Descriptions<a name="files"></a>

There are three main folders and other files:
1. train
    - model.py: This contains the model object definition which we will use to construct our simple RNN model that we will use to perform Sentiment Analysis. 
    - requirements.txt: It contains required Python libraries which will be installed by SageMaker in training container before training script is run by SageMaker and will also tell SageMaker what Python libraries are required by our custom inference code.
    - train.py: It is the training script which will be executed when the model is trained and it contains the necessary code to train our model.
2. serve
    - model.py: It is the same python script that we used to construct our simple RNN model.
    - predict.py: The script which contains our custom inference code for predicting the sentiment of the input review.
    - requirements.txt: It is the same file which is used during model training and it will also tell SageMaker what Python libraries are required by our custom inference code.
    - utils.py: It is the script which contains the data pre-processing functions which will be used during the initial data processing.
3. website
    - index.html: It is a simple static html file which we will use to interact with our simple RNN model created and deployed through SageMaker. It is a web page which a user can use to enter a movie review. The web page will then send the review off to our deployed model which will predict the sentiment of the entered review. 
4. SageMaker Project.ipynb: It is a jupyter notebook which contains the necessary code to implement the entire project.
5. report.html: An HTML export of the 'SageMaker Project.ipynb' notebook.
6. Web App Diagram.svg: The diagram which gives an overview of how the various Amazon services will work together.

## Instructions<a name="instructions"></a>
The deployment project which we will be working on is intended to be done using Amazon's SageMaker platform. In particular, it is assumed that we have a working notebook instance in which we can clone the deployment repository. We will clone the `https://github.com/udacity/sagemaker-deployment.git` repository and then work on the project.


## Results<a name="results"></a>
* Using Amazon's SageMaker service, we had created & deployed the model. In addition, we had constructed a simple web app which interact with the deployed model using Amazon API Gateway and Lambda services integration.

* Below is an example of a review that was entered into the web app and the predicted sentiment of the example review.

	- Review can be found [here](https://www.imdb.com/review/rw5160204/?ref_=tt_urv).
	- Review entered: "Most of the time movies are anticipated like this they end up falling short, way short. Joker is the first time I was more than happy with the hype. Please ignore the complaints of "pernicious violence" as they are embarrassing to say the least. We haven't seen a comic movie this real before. If we ever "deserved" a better class of criminal - Phillips and Phoenix have delivered. This is dark, Joker IS dark and you will fall in love with the villain as you should. The bad guys are always more romantic anyway."
	- Output Sentiment: Your review was POSITIVE!

* Read the entire analysis in the below mentioned project notebook. 
[Project Notebook: Deploy a Sentiment Analysis Model](https://nbviewer.jupyter.org/github/gauravansal/Deploying-a-Sentiment-Analysis-Model/blob/master/SageMaker%20Project.ipynb)

## Screenshots<a name="screenshots"></a>

***Screenshot: Sentiment Analysis web app page with input review & output sentiment***
![Screenshot](https://github.com/gauravansal/Deploying-a-Sentiment-Analysis-Model/blob/master/Sentiment%20Analysis%20snapshot.JPG)

## Licensing, Authors, and Acknowledgements<a name="licensing"></a>

<a name="license"></a>
### License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<a name="acknowledgement"></a>
### Acknowledgements

This project was completed as part of the [Udacity Machine Learning Engineer Nanodegree](https://www.udacity.com/course/machine-learning-engineer-nanodegree--nd009t). The dataset used in this project is [IMDb dataset](http://ai.stanford.edu/~amaas/data/sentiment/). See ACL(Association for Computational Linguistics) 2011 paper[[bib]](http://ai.stanford.edu/~amaas/papers/wvSent_acl2011.bib) for more information regarding the dataset used.