# neuromorphic-
Neuromorphic computing


# Neuromorphic Computing with Spiking Neural Networks (SNN)

## Overview

This project implements a **Spiking Neural Network (SNN)** using PyTorch in a **single Python file**.

Unlike traditional neural networks, neuromorphic systems:
- Communicate using **spikes (events)** instead of continuous values
- Mimic **biological neurons**
- Are more **energy-efficient** and **temporal-aware**

---

## 🧠 Model Description

We implement a **Leaky Integrate-and-Fire (LIF)** neuron model:

### Key Concepts

- **Membrane potential (mem)** accumulates input
- When it crosses a **threshold**, a spike is emitted
- The neuron **resets after firing**
- Potential **decays over time**

---

## ⚙️ Architecture


---

## ⏱ Temporal Dynamics

- The network runs for **multiple time steps (T = 25)**
- Same input is processed repeatedly
- Output is based on **total spike activity**

---

## 🔁 Training Method

Since spikes are non-differentiable:
- We use a **surrogate gradient**
- Approximates gradients during backpropagation

---

## 📊 Dataset

- MNIST handwritten digits
- Converted into noisy spike-like input

---

## 🚀 How to Run

### 1. Install dependencies

```bash
pip install torch torchvision


What This Demonstrates

Event-driven computation

Temporal learning

Biologically inspired AI

Neuromorphic principles in software

🔮 Future Extensions

STDP (Spike-Timing Dependent Plasticity)

Neuromorphic hardware (Loihi, SpiNNaker)

Graph-based spiking networks

Energy benchmarking vs ANN

📌 Summary

This repo shows how neuromorphic computing can be:

Implemented in a single file

Trained with modern deep learning tools

Extended toward biological realism and efficiency
