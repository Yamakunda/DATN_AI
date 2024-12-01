# Link Dataset
https://drive.google.com/drive/u/1/folders/1dGwd-tVZIEc4dXY5SW13CfTlEUHZGDIw
# Dog Diseases Classification Modeling
This repository contains 2 folders to store our notebook for 2 different models. Both models trained using TensorFlow, and the dataset can be found in dataset folder.
# Dog Symptoms Model
The model aim to accurately classify 12 dog diseases based on symptoms given using ANN and it has been implemented in our application. 
| **Information** | **Value** |
| --- | --- |
| Preprocessing Technique | One Hot Encoding |
| Model Structure | Combination of Dense layers and Dropout layers |
| Model Input | List of symptomps |
| Model Output | Classification of Dog Diseases |

# Dog Skin Diseases Model 
The model aim to classify 4 dog skin diseases using CNN. The image dataset were collected independently by manually downloading it from Google Images. We didn't conduct image scraping due to data scarcity. Despite the fact that we're dealing with model mismatch problem, we kept deploying it in our application with some considerations.
| **Information** | **Value** |
| --- | --- |
| Preprocessing Technique | Offline augmentation, Image normalization |
| Model Structure | Classic CNN |
| Model Input | Image |
| Model Output | Classification of Dog Skin Diseases |
