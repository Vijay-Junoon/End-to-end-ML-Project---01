import yaml
from sklearn.ensemble import RandomForestClassifier
import pickle
import mlflow
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from mlflow.models import infer_signature
import pandas as pd
import os
from sklearn.model_selection import train_test_split,GridSearchCV
from urllib.parse import urlparse

# ! Tracking URI from Experiments
os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/vijay262005/ML_pipeline_01.mlflow"

# ! Username from Experiments - Repo Ownder
os.environ["MLFLOW_TRACKING_USERNAME"] = "vijay262005"

# ! Secret access key from Data
os.environ["MLFLOW_TRACKING_PASSWORD"] = "9da24b8499c61a98888907caf7183592abc4c819"


# ! Method to control hyperparameter tuning using GridSearch
def hyperparameter_tuning(X_train,y_train,param_grid):
  classifier = RandomForestClassifier()
  grid_search = GridSearchCV(estimator=classifier,param_grid=param_grid,cv=3,n_jobs=-1,verbose=2)
  grid_search.fit(X_train,y_train)
  return grid_search

# ! Load parameters from params.yaml
params = yaml.safe_load(open("params.yaml"))["train"]


def train(data_path,model_path,random_state,n_estimators,max_depth):
  data = pd.read_csv(data_path)
  X = data.iloc[:,:-1]
  y = data.iloc[:,-1]

  mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

  with mlflow.start_run():
    #  ! Split dataset into training and test set
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=random_state)
    
    signature = infer_signature(X_train,y_train)

    param_grid = {
      'n_estimators': [100,200],
      'max_depth': [5,10],
      'min_samples_split' : [2,5],
      'min_samples_leaf' : [1,2]
    }

    # ! Perform hyperparameter tuning by calling function
    grid_search = hyperparameter_tuning(X_train,y_train,param_grid)

    # ! Obtaining the best model
    best_model = grid_search.best_estimator_

    # ! Predict and evaluate the model

    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    print(f"Accuracy: {accuracy}")

    #  ! Log additional metrics
    mlflow.log_metric("Accuracy",accuracy)
    mlflow.log_param("best_n_estimators",grid_search.best_params_['n_estimators'])
    mlflow.log_param("best_max_depth",grid_search.best_params_["max_depth"])
    mlflow.log_param("best_min_samples_split",grid_search.best_params_["min_samples_split"])
    mlflow.log_param("best_min_samples_leaf",grid_search.best_params_["min_samples_leaf"])
    
    # ! Log Confusion matrix and classification report
    cm = confusion_matrix(y_test,y_pred)
    cr = classification_report(y_test,y_pred)
    
    mlflow.log_text(str(cm),"Confusion_Matrix.txt")
    mlflow.log_text(cr,"Classification_Report.txt")

    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

    if tracking_url_type_store != "file":
      mlflow.sklearn.log_model(best_model,"model",registered_model_name="Best Model")
    else:
      mlflow.sklearn.log_model(best_model,"model",signature=signature)

    
    # ! Create a directory to save the model
    os.makedirs(os.path.dirname(model_path),exist_ok = True)

    filename = model_path
    pickle.dump(best_model,open(filename,"wb"))

    print(f"Model saved to {model_path}")

if __name__ == "__main__":
  train(params['data'],params['model'],params['random_state'],params['n_estimators'],params['max_depth'])