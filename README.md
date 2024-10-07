# DeepSLE-Official Implementation

## Overview
Welcome to the GitHub repository for DeepSLE, the implementation of the paper titled "A deep learning system for detecting systemic lupus erythematosus and related complications from retinal images". This project detects systemic lupus erythematosus(SLE), lupus retinopathy(LR), and lupus nephritis(LN) using deep learning techniques. This paper is under review.

<!-- ## Configuration
Please refer to the requirements.txt file, and install the necessary dependencies using the following command:
```bash
pip install -r requirements.txt
``` -->

## Pre-training
The system was pretrained using the same model architecture and configurations as [RETFound](https://github.com/rmaphoh/RETFound_MAE), but using our own curated dataset. Details could be refer to [RETFound Paper](https://www.nature.com/articles/s41586-023-06555-x)

## Training 
```bash
python train_bwce.py
```

## Testing
```bash
python test.py
```
