# ENEL645 Assignment 2

We finetuned Paligemma, a 3-Billion parameter vision-language model developed by Google, using LoRA to make the training process more efficient. We framed the task as a multiple-choice-question answering problem, and achieved an accuracy of ~86% on the test set.

Final training was done on the ARC cluster with 12 epochs, a batch size of 8, and a 4e-5 learning rate which we found, experimentally, to give the best results within reasonable time frame. Inference was done on the TALC cluster. 

To run the training on ARC, use the job.slurm file and the paligemma.py file, along with the garbage dataset provided. The output from ARC is given in slurm-33168754.out. A jupyter notebook file MODEL-PREDICTION-SUBMISSION.ipynb is included to show the inference results (metrics, confusion matrix, etc.). It also contains a copy of the code for finetuning and making predictions. 
