#!/bin/bash

# The required input paramters
RESOURCE_GROUP=$1
WORKSPACE_NAME=$2
COMPUTE_INSTANCE='comp-inst'
COMPUTE_CLUSTER='comp-clust'
RESOURCE_PROVIDER="Microsoft.MachineLearning"

# Check if the resource group exists
if [[ $(az group exists --name $RESOURCE_GROUP) == true ]]
then
    # If the resource group exists, delete it
    echo "Resource group $RESOURCE_GROUP exists. Deleting..."
    az group delete --name $RESOURCE_GROUP --yes --no-wait
else
    # If the resource group does not exist, print a message
    echo "Resource group $RESOURCE_GROUP does not exist. It is being created ..."

    echo "Register the Machine Learning resource provider:"
    az provider register --namespace $RESOURCE_PROVIDER

    echo "Create a resource group and set as default:"
    az group create --name $RESOURCE_GROUP --location australiaeast
    az configure --defaults group=$RESOURCE_GROUP

    echo "Create an Azure Machine Learning workspace $WORKSPACE_NAME:"
    az ml workspace create --name $WORKSPACE_NAME 
    az configure --defaults workspace=$WORKSPACE_NAME 

    # # Create compute instance
    echo "Creating a compute instance with name: " $COMPUTE_INSTANCE
    az ml compute create --name ${COMPUTE_INSTANCE} --size STANDARD_DS11_V2 --type ComputeInstance 

    # # Create compute cluster
    echo "Creating a compute cluster with name: " $COMPUTE_CLUSTER
    az ml compute create --name ${COMPUTE_CLUSTER} --size STANDARD_DS11_V2 --max-instances 2 --type AmlCompute 
fi




# ### How to use it?
# >  ./setup.sh <RESOURCE_GROUP> <WORKSPACE_NAME>

# Example:
# >  ./setup.sh tempRG tempWS
