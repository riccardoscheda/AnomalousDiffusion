# AnomalousDiffusion
This repository is dedicated for the project of the Patter Recognition course from the master in Applied Physics of the University of Bologna

In this project i'm trying to evaluate if the process of cell migration is an anomalous diffusion process.
Given a set of frames of a monolayer of cells acting migration,
in the first part of the project i try to recognize the cells from the background through a texture matching method, using the locally binary pattern of the images. After computing the LBP image, i classify two labels for the image through the PCA and K-means algorithm.

In the second part of the project i use the fronts of the cells in each frame found in the first part to classify the type of this
diffusion process.
