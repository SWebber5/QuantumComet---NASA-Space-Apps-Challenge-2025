# NASA International Space Apps Challenge
## A World Away: Hunting for Exoplanets with AI
## Team: QuantumComet
## Members: Rachel Bonanno and Simon Webber

## High Level Project Summary
We developed a AI model (CNN architecture) trained on Kepler mission data preprocessed by AstroNet (see Citations), achieving 94.72% recall on the heldout data. This implies that a relatively simple model such as ours can significantly speed up exoplanet detection with minimal errors. Though we do not have a web interface or our own data preprocessing pipeline, our work displays how novice exoplanet enthusiasts can quickly yet effectively begin looking at exoplanet detection for themselves. 

## [Project "Demo"](https://github.com/SWebber5/QuantumComet---NASA-Space-Apps-Challenge-2025/blob/main/Competition/Astronet_Preprocessed_Data/challenge_submission.ipynb)

## [Final Project](https://github.com/SWebber5/QuantumComet---NASA-Space-Apps-Challenge-2025/tree/main/Competition/Astronet_Preprocessed_Data)

## Detailed Project Description
Our project is aimed toward beginner astronomers and space enthusiasts interested in discovering new exoplanets with automated methods. The project uses data from NASA's Kepler mission preprocessed by AstroNet and available for [download](https://drive.google.com/drive/folders/1Gw-o7sgWC1Y_mlaehN85qH5XC161EHSE?usp=sharing). We created a very basic convolutional neural network archtecture with hyperparameter tuning optimizing for validation recall - because ensuring that real exoplanets are classified as such is the most important - with tensorflow and keras tools. The results of the project have no computational benefits over AstroNet's work but aims to show the simplicity with which beginners can get started using preprocessed data. The overall goal of the project is to inspire young and new space fans to pursue exoplanet detection - and other astronomical phenomena - with the help of modern AI/ML tools; to get them involved in these processes and allow them to explore such opportunities.

## Challenges
By far the largest challenge we faced in this project was understanding the complex Kepler mission data in the time limit of the challenge (which we avoided by using AstroNet's preprocessed data). In the future, we would like to pursue deepening our own understanding of exoplanet detection, the data from the Kepler, K2, and TESS missions, and the previous work done on automated detection so that we could build upon such work.

## Use of AI
We used generative AI as a tool to quickly assist us in understanding key topics, strutures of data, and debugging tensorflow errors. Given our limited (non-existent) understanding of exoplanet detection, generative AI was extremely useful for highlighting key insights and providing resources (like the AstroNet paper) for future learning.

## Data Usage
The data ([1](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=cumulative),[2](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=TOI),[3](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=k2pandc)) provided by the Challege was looked at for understanding but ultimately not directly used in model training. However, the [AstroNet data](https://drive.google.com/drive/folders/1Gw-o7sgWC1Y_mlaehN85qH5XC161EHSE?usp=sharing) that we used is derived from the Kepler data (1) after their own preprocessing.


## Citations
Shallue, C. J., & Vanderburg, A. (2018). Identifying Exoplanets with Deep Learning: A Five-planet Resonant Chain around Kepler-80 and an Eighth Planet around Kepler-90. The Astronomical Journal, 155(2), 94.\
[Github Repo for AstroNet](https://github.com/google-research/exoplanet-ml/blob/master/README.md)