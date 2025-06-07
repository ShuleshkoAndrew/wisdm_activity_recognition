#!/bin/bash

# Initialize DVC
dvc init

# Add data and models to DVC
dvc add data/raw/
dvc add data/processed/
dvc add models/

# Create .gitignore for DVC files
echo "# DVC files
/data/raw
/data/processed
/models
*.dvc" >> .gitignore

# Configure local DVC remote (you can change this to your preferred remote storage)
dvc remote add -d local /tmp/dvc-storage

echo "DVC setup completed successfully!"
echo "Note: You may want to configure a different remote storage location for your data."
