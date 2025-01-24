# Anomaly_detection_Astronomy_data
In this repo we will perfrom Anomaly detection on astronomy data. From kaggle dataset, build an architecture that will detect anomalies in the data. Classify outliers vs regular data.
# Data Downloading
We downloaded data from kaggle "Astronomy Image Classification Dataset" using link "https://www.kaggle.com/datasets/abhikalpsrivastava15/space-images-category"

![2](https://github.com/user-attachments/assets/119fbf99-b832-41a7-9fa2-c4515a2c7fbc)
![5](https://github.com/user-attachments/assets/bbb7cdaa-9c1a-4ed5-9549-ccd08f1c51a5)
![57](https://github.com/user-attachments/assets/271e719f-bb44-47b6-83aa-1562605f2c91)



# Feature Extraction 
   - Use CLIP or DINO for extracting embeddings (pretrained weights are available).
   - Fine-tune if necessary for domain-specific anomalies.

# Classification of anomaly

## With Isolation forest 

`python isolation_forest.py`

## With Auto Encoders 
'python Auto_encoders.py'

## With ResNet + ViT + Isolation Forest  
`python ResNet_ViT_Isolation_Forest.py`


   
