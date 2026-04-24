# AllyACC

This repository accompanies the paper: [Giving Meaning to Movements (CHI 2026)](https://a11y.ist.psu.edu/downloads/CHI_2026__Giving_Meaning_to_Movements.pdf).

It provides the implementation of AllyACC introduced in that paper.

## Android app

The Android app is included as the `AllyACCApp` submodule. Please see `AllyACCApp/README.md` for app-specific setup and usage details.

After cloning, run `git submodule update --init --recursive` to fetch the `AllyACCApp` folder.

## Data
The dataset is available in the `dataset` folder. 
For each participant, the data is split into training and test sets. 
Only the IMU data is included here. To protect participant privacy, 
we are not sharing the original video data at this stage. 
We plan to release a privacy-preserving version of the video data, 
either by de-identifying the videos through face and identifiable-feature 
blurring or by providing pose/skeleton representations that preserve the 
necessary motion information for activity recognition.

The code for the large model will be made available soon.
