#### Github Repository Architecture

The tutorial assumes you have access to a github repository that contains all the code required to train and deploy an Azure Databricks ML model. 

<b> Note: </b> Without access to the repo you cannot proceed with the tutorial.

<b> By default the repo is located here:https://github.com/snapAzureMlLab/azuremllab </b>

The repository should have the following structure:

#### Environment Setup

- requirements.txt : It consists of a list of python packages which are needed by the train.py to run successfully on host agent (locally).

- set-environment-vars.sh : This script prepares the python environment i.e. install the Azure ML SDK and the packages specified in requirements.txt

- environment.yml :  build agent containing Python 3.6 and all required packages.

- setup.py : Importing the set of package for model

#### Pipelines

- /azdo_pipelines/base-pipeline.yml : a pipeline template used by build-pipeline. It contains steps performing linting, data and unit testing.

- /azdo_pipelines/build-pipeline.yml : a pipeline triggered when the code is merged into master. It performs linting,data integrity testing, unit testing, building and publishing an ML pipeline.

#### cluster_config

- cluster_config/cluster.py : Databricks Cluster config for creating a new cluster and terminate or delete cluster.

- cluster_config/cluster_manager.py : Use Existing Databricks cluster and terminate the cluster. 

- library.json : install the library azure - ml - sdk in databricks cluster 

#### Code

- /src/train/train.py : a training step of an ML training pipeline.

- /aml_service/pipelines/train_pipeline.py : Create a pipeline in azure machine learning for training model. 

- /aml_service/experiment/experiment.py : Create a experiment for Tracking the model 

- /aml_service/experiment/register_model.py : Registers a  trained mode into azure machine learning service.

- /aml_service/experiment/workspace.py  : Connect the azure machine learning workspace with azure devops.

- /aml_service/experiment/attach_compute.py : select the Target the cluster compute (databricks) for training the model

#### Scoring

- /src/score/score.py : a scoring script which is about to be packed into a Docker Image along with a model while being deployed to QA/Prod environment.

- /src/score/conda_dependencies.yml : contains a list of dependencies required by sore.py to be installed in a deployable Docker Image

- /src/score/inference_config.yml, deployment_config_aci.yml, deployment_config_aks.yml : configuration files for the AML Model Deploy pipeline task for ACI and AKS deployment targets.
