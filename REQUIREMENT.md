# Machine Learning and Deployment Requirements

This document defines delivery requirements for Bisakerja model training, inference, REST API serving, Generative AI integration, and final repository evidence.

## 1. Deep Learning Model Development

### 1.1 Model Architecture

Build a Deep Learning model using one of these TensorFlow approaches:

- TensorFlow Functional API
- TensorFlow Model Subclassing

The architecture must match the dataset and business problem selected for the Bisakerja job-fit and CV-analysis workflow.

### 1.2 Custom Component Implementation

The model must implement at least one advanced custom component:

- Custom Layer
- Custom Loss Function
- Custom Callback

### 1.3 Custom Training Loop

Training and evaluation must use a full custom loop with `tf.GradientTape`.

`model.fit()` must not be the primary training loop for the final delivery evidence.

### 1.4 Training Monitoring

Training must integrate TensorBoard for monitoring.

Evidence must include:

- training metrics
- evaluation metrics
- TensorBoard event logs committed or recorded as release artifacts
- artifact manifest entries with SHA-256 and byte size when logs are stored under release paths

### 1.5 Model Performance Target

The model must meet the selected problem target.

#### Classification

- Accuracy at least `85%`.

#### Regression

- Mean Absolute Error (MAE) at most `0.02`.

For Bisakerja production-readiness review, weak-label metrics alone are not enough. Human/recruiter-reviewed validation and slice coverage remain required for trustworthy product claims.

## 2. Model Storage and Deployment

### 2.1 Model Export

Export the trained model in a production-ready TensorFlow format:

- `.keras`
- or `SavedModel`

Exported artifacts must be accompanied by model card, feature config, calibration config, and manifest evidence when used by Model API.

### 2.2 Inference

Provide inference code that can:

- load the exported model
- build the approved input feature vector
- run prediction
- return bounded model-core output

## 3. REST API Development

### 3.1 API Framework

Provide a standalone REST API using one of these frameworks:

- FastAPI
- Flask

### 3.2 Model Integration

The REST API must:

- load the exported model
- accept validated user/backend input
- run inference
- return JSON output
- expose deterministic health/readiness behavior

For this project, Model API is internal-only and Backend API owns the public REST boundary.

## 4. Generative AI Integration

### 4.1 Additional Feature

Use a Generative AI API as an additional or secondary feature, such as:

- prediction explanation
- data summary
- recommendation copy based on model output
- chat assistant or other supporting AI feature

Core model inference must not depend on external GenAI calls. Backend wrapper behavior owns product-facing prose and fallbacks.

## 5. Deliverables

Final repository evidence must include:

- model training source in versioned notebooks
- custom TensorFlow component implementation
- custom training loop using `tf.GradientTape`
- exported model (`.keras` or `SavedModel`)
- inference code
- REST API using FastAPI or Flask
- Generative AI integration boundary or wrapper evidence
- TensorBoard logs or release artifact evidence
- usage and deployment documentation
- `requirements.txt`
- project README
- model card, artifact manifest, contract fixtures, and release-gate reports
