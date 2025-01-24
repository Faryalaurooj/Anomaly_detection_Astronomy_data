# Anomaly_detection_Astronomy_data
In this repo we will perfrom Anomaly detection on astronomy data. From kaggle dataset, build an architecture that will detect anomalies in the data. Classify outliers vs regular data.
# Data Downloading
We downloaded data from kaggle "Astronomy Image Classification Dataset"

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


   
