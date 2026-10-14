# AI-Driven Phytosanitary Surveillance: In-Field Detection, Binary Classification, and Quantitative Disease Severity Estimation for Potato Minitubers

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/Ultralytics-YOLOv8n-green.svg)](https://github.com/ultralytics/ultralytics)
[![Framework](https://img.shields.io/badge/Academic%20Project-CSAI--511-orange.svg)]()
[![Partner](https://img.shields.io/badge/Industrial%20Partner-NADEC-darkblue.svg)](https://www.nadec.com.sa/)

> **Joint Research & Applied Innovation Initiative**  
> **Institution:** University of Hail | College of Computer Science and Engineering  
> **Program:** Executive Master of Artificial Intelligence (CSAI-511)  
> **Researcher:** Nada Al-Hilali  
> **Academic Supervisor:** Dr. Meshari Alazmi  
> **Industrial Partner:** National Agricultural Development Company (NADEC) — Hail Group

---

## 📌 Executive Summary

Potato minitubers produced in closed commercial greenhouse bays represent high-value nuclear foundation seed stock. However, forced-air circulation driven by evaporative cooling padwalls creates humid microclimatic corridors that accelerate foliar blight outbreaks (*Alternaria solani* and *Phytophthora infestans*). 

Traditional biosecurity relies on manual scouting along peripheral walkways, which suffers from observational fatigue, accessibility bias across dense benches, and late detection after secondary airborne spores have already disseminated.

This repository provides an automated, edge-deployable artificial intelligence surveillance framework that:
1. Detects foliar blight in standard RGB proximal imagery using a streamlined **YOLOv8 Nano** architecture under an epidemiologically grounded binary scheme (`Healthy` vs. `Infection`).
2. Quantifies exact foliar damage using automated **HSV color space segmentation** to calculate continuous **Disease Severity (%)**.
3. Maps continuous severity indices to an actionable **Three-Tier Agronomic Intervention Protocol**, targeting a **40% to 60% reduction in commercial fungicide use** and supporting automated **AUDPC** trajectory tracking.

---

## 📊 Experimental Results & Model Performance

The model was evaluated on a strictly isolated, out-of-sample test set comprising **20% of the total corpus (951 independent images)** with random seed stabilization (`Seed 42`) to guarantee zero data leakage.

### Quantitative Test Set Benchmark (951 Images)

| Evaluation Metric | Overall Performance | Class: `Healthy` (191) | Class: `Infection` (760) | Field Significance |
| :--- | :---: | :---: | :---: | :--- |
| **mAP@50** | **99.5%** | 99.4% | 99.5% | Exceptional bounding-box spatial localization |
| **mAP@50-95** | **99.5%** | 99.4% | 99.5% | High bounding-box tightness across IoU thresholds |
| **Precision** | **99.1%** | 98.2% | 100.0% | Virtually zero false alarms on healthy canopies |
| **Recall** | **99.5%** | 99.5% | 99.6% | 99.5%+ capture rate of early fungal lesions |
| **Inference Latency** | **1.9 ms / frame** | — | — | Edge-compatible speed for real-time robotic/rail scouting |

---

## 🔬 Methodology & Technical Architecture
