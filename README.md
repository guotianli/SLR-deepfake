# SLR-deepfake
This project includes the code, dataset and complete processing flow of the paper "Score-based Likelihood Ratios for Deepfake Image Evidence Using Deep Learning Features", which is used to reproduce the experimental results. 

## 1. The dataset is from UADFV and is an open dataset.
### 1.1 Original Data
- Source: The dataset is from UADFV and is an open dataset. The path is: https://docs.google.com/forms/d/e/1FAIpQLScKPoOv15TIZ9Mn0nGScIVgKRM9tFWOmjh9eHKx57Yp-XcnxA/viewform
- Storage Path: `UADFV/datasets`
- Data Size: There are 49 files for both real and fake categories. Each file contains 32 images of the person.
- Annotation Rules: real and fake categories 

1.2 Cleaned/Processed Data
- Storage Path: `UADFV/preprocessing/dataset_json`
- Data Content: JSON format files
- Processing Purpose: Includes paths of face images and labels 

## 2. Complete Data Processing Flow
1. Video processing to image:
- Steps: Read the original image → Crop the face using 81 feature points → Resize to 256×256 → Save to `UADFV/datasets`
- Corresponding code: UADFV/preprocessing/preprocess.py
2. Save in JSON format for deep learning, pay attention to distinguishing the train\val\test subsets when saving to prevent data leakage:
- Steps: Store the processed images along with their paths and labels
- Corresponding code: UADFV/preprocessing/rearrange.py
3. Train the network:
- Steps: Train the network that performs well on the validation set, with good binary classification performance. This allows the extracted npy files to distinguish the features of real and fake classes. The training results are saved as a pth file in the UADFV/weights folder
- Corresponding code: UADFV/training/train.py
4. Test the network, results saved to UADFV/training/npy folder
- Steps: Use the pth to test the test set
- Corresponding code: test_0501_srm_img.py
5. View the shape of the npy file
- Corresponding code: 0npyShap.py
6. Reduce the dimensions of the real and fake class features of the npy for intuitive observation
- Corresponding code: 1TSNE.py
7. Record network performance and draw ROC curve
- Corresponding code: 2ROC.py
8. Calculate the similarity between the class centers and the npy of real and fake images
- Corresponding code: 3Metric.py
9. Draw probability density curve
- Corresponding code: 4PDF.py
10. Calculate the likelihood ratio
- Corresponding code: 5LR.py
11. Draw Tippett curve to judge the performance of the LR model
- Corresponding code: 6Tippett.py
12. Draw DET curve to judge the performance of the LR model
- Corresponding code: 7Det.py
13. Draw ECE curve to judge the performance of the LR model
- Corresponding code: 8ECE.py
14. Draw ELUB curve to solve the tail effect caused by probability density fitting
- Corresponding code: 9ELUB.py 

## 3. Replication Steps
1. Clone the repository: `git clone https://github.com/guotianli/SLR-deepfake.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the scripts in sequence. For hardcoded file paths, modify them according to the local path.
4. Output the results