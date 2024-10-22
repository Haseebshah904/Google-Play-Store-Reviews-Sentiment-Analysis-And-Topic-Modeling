# Google Play Store Reviews Sentiment Analysis And Topic Modeling
This project provides a comprehensive solution for analyzing Google Play Store reviews. It combines sentiment analysis and topic modeling to gain valuable insights into user feedback, categorizing reviews into positive, negative, or neutral sentiments, and extracting the key topics discussed. The project is deployed using Hugging Face with Gradio for an interactive interface, and Docker is utilized for containerization, ensuring ease of deployment.

# Project Overview

The Google Play Store Reviews Sentiment Analysis & Topic Modeling project is designed to analyze app reviews by:

+ Sentiment Analysis: Classifying reviews into positive, neutral, or negative categories.
+ Topic Modeling: Extracting key topics from the reviews to identify trends and areas of concern or praise.
This project helps app developers and businesses better understand user feedback by providing detailed insights into how their app is perceived.

# Features

+ Sentiment Classification: Classifies reviews into three categories: Positive, Negative, and Neutral.
+ Topic Modeling: Automatically identifies and extracts key topics from user reviews.
+ Interactive Interface: Deployed on Hugging Face using Gradio for real-time interaction.
+ Containerization: Packaged with Docker for smooth deployment and scalability.
  
# Installation

## Prerequisites

Ensure you have the following installed:

+ Python 3.7 or higher
+ Docker
+ Git
Clone the Repository
+ bash
+ Copy code
+ git clone https://github.com/yourusername/Google-Play-Store-Reviews-Sentiment-Analysis-And-Topic-Modeling.git
cd Google-Play-Store-Reviews-Sentiment-Analysis-And-Topic-Modeling

# Set Up the Python Environment
## 1.Install dependencies:
+ bash
+ pip install -r requirements.txt
 
## 2.Install docker if it is not already installed:

+ bash
+ sudo apt install docker.io
# Usage

## Running the Application

## 1.Local Deployment with Python: To run the application locally, simply execute the app.py script.
+ bash
+ python app.py
You can then access the Gradio interface via your local host to interact with the sentiment analysis and topic modeling features.

## 2.Containerized Deployment with Docker: Build and run the Docker container for the project:

+ bash
+ docker build -t sentiment-topic-modeling .
+ docker run -p 7860:7860 sentiment-topic-modeling
Once the container is up, navigate to http://localhost:7860 to interact with the Gradio interface.

# Interacting with the Application

+ Once the application is running, upload Google Play Store reviews or scrape new ones using the provided functionality.
+ The app will display sentiment analysis results and the key topics extracted from the reviews.

# Project Structure

+ Python File includes this file Google_Playstore_Reviews_Sentiment_+_Topic_Modeling_Project_Completed_.ipynb
+ Deployment includes Docker, app.py, requirements.txt

# Models and Libraries

## Sentiment Analysis

For sentiment classification, the project utilizes a pre-trained transformer model from the Hugging Face library. The model is fine-tuned on a sentiment analysis dataset to classify reviews as positive, negative, or neutral.

## Topic Modeling

For topic extraction, we use Top2Vec, a state-of-the-art topic modeling technique that identifies topics in the text based on sentence embeddings.

# Libraries
Some of the main libraries and tools used in the project include:

+ Gradio: For creating the interactive user interface.
+ Transformers: Pre-trained transformer models for sentiment classification.
+ Top2Vec: For topic modeling.
+ TensorFlow: Required for certain transformer models.
+ Matplotlib & WordCloud: For data visualization.
+ Refer to the requirements.txt file for the complete list of dependencies used in this project.

# Deployment

## Hugging Face Integration
The project is integrated with Hugging Face Spaces, using the Gradio library to build an interactive interface for users to input their own reviews and receive sentiment and topic modeling results in real time.
![Image Alt](https://github.com/Haseebshah904/Google-Play-Store-Reviews-Sentiment-Analysis-And-Topic-Modeling/blob/main/User%20interface%201.PNG?raw=true).
![Image Alt](https://github.com/Haseebshah904/Google-Play-Store-Reviews-Sentiment-Analysis-And-Topic-Modeling/blob/main/User%20interface%202.PNG?raw=true).

## Docker

To make the deployment process smooth and scalable, the project includes a Dockerfile to containerize the application. This allows for easy deployment across different environments without worrying about dependency issues.

# Contributing

Contributions are welcome! If you would like to contribute to this project, feel free to fork the repository and submit a pull request.
