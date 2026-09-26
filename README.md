# CyberShield AI

## Explainable AI-Based Cyber Threat Intelligence and Intrusion Detection System

CyberShield AI is an Explainable AI-based Cyber Threat Intelligence and Intrusion Detection System designed to detect, classify, explain, and assess potentially malicious network activity using machine learning and Explainable AI techniques.

The project combines network traffic monitoring, machine learning-based intrusion detection, explainable predictions, threat severity assessment, backend services, database storage, web-based visualization, and security reporting into a modular cybersecurity system.

---

## Project Overview

Traditional Intrusion Detection Systems can identify suspicious network activity, but the reasoning behind a machine learning prediction may not always be clear to the user.

CyberShield AI aims to address this limitation by integrating machine learning with Explainable AI techniques. The system is designed to analyse network traffic, identify benign and malicious activity, classify different attack categories, and provide explanations for the predictions.

The project follows a phased development approach covering data acquisition, preprocessing, machine learning, explainability, live network monitoring, backend development, database integration, dashboard development, reporting, and end-to-end validation.

---

## Objectives

The main objectives of CyberShield AI are:

- Detect malicious network traffic using machine learning.
- Distinguish between benign and malicious network activity.
- Classify network traffic into different attack categories.
- Develop a reproducible data preprocessing pipeline.
- Compare multiple machine learning models for intrusion detection.
- Apply Explainable AI techniques to understand model predictions.
- Assess the severity of detected threats.
- Integrate ESP32-based Wi-Fi monitoring into the system.
- Provide a backend API for processing detection results.
- Store security events and prediction information.
- Present threat information through an interactive dashboard.
- Generate security reports for detected incidents.

---

## Key Features

### Machine Learning-Based Intrusion Detection

CyberShield AI uses machine learning models to analyse network traffic features and classify network activity.

The planned machine learning models include:

- Random Forest
- XGBoost
- LightGBM
- CatBoost
- Multi-Layer Perceptron (MLP)

The models will be evaluated using suitable classification metrics and compared before final model selection.

### Multi-Class Attack Classification

The system uses the following attack classification categories:

| Class | Description |
|---|---|
| Benign | Normal network traffic |
| DoS | Denial-of-Service traffic |
| DDoS | Distributed Denial-of-Service traffic |
| PortScan | Port scanning activity |
| BruteForce | Brute-force activity |
| WebAttack | Web-based attack traffic |
| Infiltration | Network infiltration activity |
| Bot | Bot-related traffic |
| Heartbleed | Heartbleed-related traffic |

The classification is based on network traffic characteristics rather than simply identifying a particular device as an attack source.

---

## Explainable AI

CyberShield AI incorporates Explainable AI techniques to provide interpretable information about machine learning predictions.

The planned XAI technologies include:

- SHAP
- LIME

The explainability layer is intended to identify important network traffic features that contribute to a particular prediction.

---

## Threat Severity Assessment

The system includes a planned threat severity assessment layer.

The severity assessment considers factors such as:

- Predicted attack category
- Model confidence
- Traffic characteristics
- Attack severity weight

The resulting threat information will be presented using security-oriented severity levels.

---

## ESP32-Based Network Monitoring

An ESP32 is planned as part of the live Wi-Fi monitoring architecture.

The ESP32 will be used as a wireless monitoring component for observing Wi-Fi-level network activity and transmitting relevant monitoring information to the processing system.

The ESP32 cannot directly provide the complete IP-flow feature set required by the machine learning model, particularly when traffic is protected by Wi-Fi encryption. Therefore, the project follows a hybrid architecture in which ESP32-based Wi-Fi monitoring is combined with host-side traffic capture and feature processing where required.

---

## System Architecture

The planned system follows this general processing flow:

```text
                    Network Environment
                           |
             +-------------+-------------+
             |                           |
        ESP32 Monitor              Host Traffic Capture
             |                           |
             +-------------+-------------+
                           |
                           v
                    Data Acquisition
                           |
                           v
                    Preprocessing
                           |
                           v
                 Feature Preparation
                           |
                           v
                 ML-Based Detection
                           |
                           v
                Attack Classification
                           |
                           v
                  Explainable AI
                           |
                           v
                 Threat Assessment
                           |
                           v
                     FastAPI API
                           |
              +------------+------------+
              |                         |
              v                         v
          Database                React Dashboard
              |                         |
              +------------+------------+
                           |
                           v
                     PDF Reports