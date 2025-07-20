# **Project**

This project is exploit reinforcement learning to make RL agent tune DRAM controller adaptively. It integrates powerful Zsim and ramulator simulators for DRAM performance and power analysis.

---

## **Components Overview**

### **1. OSS-Arch-Gym**

* Reinforcement learning framework for **controller tuning**.
* Provides interfaces and tools for RL-based architecture exploration.
* Supports multi-agent reinforcement learning and continual learning for adaptive system optimization.

---

### **2. ZSim-Ramulator**

* **CPU simulator** designed for memory access pattern extraction.
* Includes **DRAMPower** module for DRAM power evaluation.

---

### **3. Ramulator**

* **DRAM simulator** for task execution analysis and configuration optimization.
* **Key Features:**
  * Simulates DRAM access behavior under various workloads.
  * Supports DRAM tuning, performance evaluation, and analysis.


---

## **How to Use**

### **0. Workspace**
```bash
cs oss-arch-gym/sims/DRAM/
```

### **1. Learning**

Train agents with continual learning and multi-agent reinforcement learning

```bash
bash incre_wobc_womodel.sh
```

---

### **2. Inference**

Evaluate trained agent models

```bash
bash oss-arch-gym/sims/DRAM/inference_marl.sh
```

---

### **3. Analysis**

**(1) Format Compatibility**

```text
rename_file.ipynb
```

**(2) Visualize learning results**

```text
visualize_learn.ipynb
```

**Note:** Move the `rename` folder into `continuity_analysis` before running this notebook.

---

## **Key Features**
- Integration of **reinforcement learning** and **continual learning** to optimize DRAM configurations.
- Modular design for easy addition of new simulators or optimization algorithms.
- Detailed performance and power analysis for DRAM systems.
