# Smart Room Occupancy Prediction Data Pipeline

Built an end-to-end data pipeline to predict room occupancy using IoT sensor data. 
The project includes data cleaning, feature engineering, machine learning modeling, SQL analysis, and cloud storage using Python, Pandas, Scikit-learn, AWS S3, and Athena.

smart-room-occupancy-data-pipeline/
│
├── data/
│   ├── raw/
│   │   └── occupancy_raw.csv
│   ├── processed/
│   │   └── occupancy_cleaned.csv
│   └── output/
│       └── occupancy_predictions.csv
│
├── notebooks/
│   └── 01_exploratory_analysis.ipynb
│
├── src/
│   ├── 01_extract_data.py
│   ├── 02_transform_data.py
│   ├── 03_train_model.py
│   ├── 04_make_predictions.py
│   └── 05_upload_to_s3.py
│
├── sql/
│   ├── occupancy_by_hour.sql
│   ├── avg_co2_by_occupancy.sql
│   └── high_co2_periods.sql
│
├── reports/
│   ├── figures/
│   │   ├── co2_distribution.png
│   │   ├── occupancy_by_hour.png
│   │   └── feature_importance.png
│   └── model_report.txt
│
├── models/
│   └── occupancy_model.pkl
│
├── README.md
└── requirements.txt

