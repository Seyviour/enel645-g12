# ENEL645 Assignment 2

We finetuned Paligemma, a 3-Billion parameter vision-language model developed by Google, using LoRA to make the training process more efficient. We framed the task as a multiple-choice-question answering problem, and achieved an accuracy of ~86% on the test set.

Final training was done on the ARC cluster with a batch size of 8 and a 4e-5 learning rate which we found, experimentally, to give the best results. Inference was done on Ziheng's RTX 4070 Super GPU. 

To run the training on ARC, use the job.slurm file and the paligemma.py file, along with the garbage dataset provided. The output from ARC is given in slurm-33168754.out. A jupyter notebook file ENEL645_Fine_tune_PaliGemma_Inference.ipynb is included to show the inference results (metrics, confusion matrix, etc.). 
