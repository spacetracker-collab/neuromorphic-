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


--
show also the Python code for, in GitHub for, the integrated information theory in neuromorphic computing for consciousness, and how to run it in Colab.

# Neuromorphic Computing + Integrated Information (Phi)

## Overview

This project combines:

- Spiking Neural Networks (SNN)
- Neuromorphic computing principles
- Integrated Information Theory (IIT)

---

## 🧠 What is implemented?

### 1. Neuromorphic Model
- Leaky Integrate-and-Fire neurons
- Temporal spike-based processing

### 2. Integrated Information (Φ proxy)

We approximate Φ as:

Φ ≈ Total system variance − Independent variance

This captures:
- Degree of **integration across neurons**
- Emergent **collective behavior**

---

## ⚙️ Architecture

Input → SNN → Spike Dynamics → Φ computation

---

## 🚀 Run in Colab

```bash
!git clone https://github.com/YOUR_USERNAME/iit-neuromorphic-snn.git
%cd iit-neuromorphic-snn
!pip install torch torchvision
!python iit_neuromorphic_snn.py


Higher Φ → more integrated neural activity

Suggests higher "coherence" in processing

Proxy for consciousness-like integration

---

# Global Workspace Theory (GWT) + Neuromorphic SNN

## Overview

This project implements a computational version of:

- Global Workspace Theory (Baars, Dehaene)
- Neuromorphic Spiking Neural Networks

---

## 🧠 Core Idea

Many unconscious modules compete for attention.

The winner:
→ enters the global workspace  
→ gets broadcast to the entire system  
→ becomes "conscious"

---

## ⚙️ Architecture

Input → Multiple Modules → Attention Competition → Global Workspace → Output

---

## 🔁 Dynamics

At each time step:
1. Modules process input
2. Attention selects important signals
3. Workspace broadcasts globally
4. System evolves over time

---

## 🚀 Run in Colab

```bash
!git clone https://github.com/YOUR_USERNAME/gwt-neuromorphic-snn.git
%cd gwt-neuromorphic-snn
!pip install torch torchvision
!python gwt_neuromorphic_snn.py


