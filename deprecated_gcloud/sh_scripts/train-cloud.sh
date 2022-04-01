#!/bin/bash
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# This script performs cloud training for a PyTorch model.

echo "Submitting Slicegan Job"

# BUCKET_NAME: Change to your bucket name.
BUCKET_NAME=tha_bucket123

# The PyTorch image provided by AI Platform Training.
IMAGE_URI=gcr.io/cloud-ml-public/training/pytorch-gpu.1-10

# JOB_NAME: the name of your job running on AI Platform.
JOB_NAME=slicegan_test10

PACKAGE_PATH=./slicegan # this can be a GCS location to a zipped and uploaded package

# REGION: select a region from https://cloud.google.com/ml-engine/docs/regions
# or use the default '`us-central1`'. The region is where the job will be run.
REGION=europe-west4

# JOB_DIR: Where to store prepared package and upload output model.
JOB_DIR=gs://slicegan_bucket

# Datasets are set by datasets/download-taxi.sh script
TRAIN_FILES=${GCS_TAXI_TRAIN_SMALL}
EVAL_FILES=${GCS_TAXI_EVAL_SMALL}

gcloud ai-platform jobs submit training ${JOB_NAME} \
    --region ${REGION} \
    --master-image-uri ${IMAGE_URI} \
    --scale-tier BASIC \
    --module-name slicegan.run_slicegan_gcloud2 \
    --package-path ${PACKAGE_PATH} \
    --job-dir ${JOB_DIR} \
    -- \
    training=1

# 
# Stream the logs from the job
gcloud ai-platform jobs stream-logs ${JOB_NAME}

# Verify the model was exported
echo "Verify the model was exported:"
gsutil ls ${JOB_DIR}/model_*

