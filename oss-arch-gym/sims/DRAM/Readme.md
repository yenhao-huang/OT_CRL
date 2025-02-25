# **Project Overview**

This repository provides scripts and tools for reinforcement learning-based workload optimization. It includes workflows for learning, inference, renaming files, and result analysis.

---

## **Workflow**

### **1. Learning**
- **Scripts**:  
  `incre_wobc_womodel.sh` / `incre_marl_multi.sh`  
  Perform training to obtain agent models.

### **2. Inference**
- **Script**:  
  `inference_marl.sh`  
  Evaluate the trained agent models.

### **3. Rename Files**
- **Notebook**:  
  `rename_file.ipynb`  
  Standardize the directory structure for result files.

### **4. Analysis**
- **Notebook**:  
  `visualize_learn.ipynb`  
  Analyze and visualize results.  
  **Note**: Move the `rename` folder to `continuity_analysis` for analysis.

---

## **Script Details**

### **`incre_wobc_womodel.sh`**
- **Required Changes**:  
  - `workloadset`: Define the workload set.  
  - `threadidx`: Specify the thread index.

### **`inference_marl.sh`**
- **Required Changes**:  
  - `ckpt_threadidx`: Thread index of the trained model.  
  - `threadidx`: Thread index for the output "inference".

### **`rename_file.ipynb`**
- **Required Changes**:  
  - `thread_idx`: Update with the "inference" thread index.

### **`single_agent.sh`**
**Output**  
Logs will be stored in:  logs/thread{thread_idx}

### `Single_workload.sh`  
- Allocate different workloads to different threads.

### **`train_single_agent.sh`**

#### **Parameters**:
- `load_checkpoint`: Whether to load a pretrained model (`default = false`).  
- `checkpoint_dir`: Path to the pretrained model when `load_checkpoint = true` (`default = ~/acme`).  
- `eval_episodes`:  
  - `0`: Training mode.  
  - `>=1`: Evaluation mode.  
- `num_steps`: Control total training steps.  
- `step_per_episode`: Control steps per episode.

### **`train_multiagent.sh`**

#### **Parameters**:
- `load_checkpoint`: Whether to load a pretrained model (`default = false`).  
- `checkpoint_dir`: Path to the pretrained model when `load_checkpoint = true` (`default = ~/acme`).  
- `only_acting`:  
  - `false`: Training mode.  
  - `true`: Evaluation mode.  
- `num_steps`: Control total training steps.  
- `step_per_episode`: Control steps per episode (`default = 1`).  
- `is_update_buffer`: Whether to update the buffer (`default = false`).  
- `rp_buf_size`: Replay buffer size.  
- `rp_ckpt_path`: Path to the replay buffer checkpoint.

#### **Modify Model Input**:
Update `DRAMEnv_RL` with:  
- `load_buf_path`  
- `save_buf_path`  
- `obs_shape`  
- `get_obs_reward`