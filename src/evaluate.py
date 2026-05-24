import pandas as pd
import pickle 
from sklearn.metrics import accuracy_score
import yaml
import os
import mlflow
from urllib.parse import urlparse

# ! Tracking URI from Experiments
os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/vijay262005/ML_pipeline_01.mlflow"

# ! Username from Experiments - Repo Ownder
os.environ["MLFLOW_TRACKING_USERNAME"] = "vijay262005"

# ! Secret access key from Data
os.environ["MLFLOW_TRACKING_PASSWORD"] = "9da24b8499c61a98888907caf7183592abc4c819"


params = yaml.safe_load(open("params.yaml"))['train']

def evaluate(data_path,model_path):
  data = pd.read_csv(data_path)
  X = data.iloc[:,:-1]
  y = data.iloc[:,-1]

  mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

  # * Loading model from the disk 
  model = pickle.load(open(model_path,'rb'))

  predictions = model.predict(X)
  accuracy = accuracy_score(y,predictions)

  # * Log Metrics
  mlflow.log_metric("Accuracy", accuracy)
  print(f"Accuracy: {accuracy}")


if __name__ == "__main__":
  evaluate(params['data'],params['model'])