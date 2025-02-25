import pickle
from pathlib import Path
import re
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import os
import tempfile

NUM_AGENTS = 7
ERR_LIMIT = 1e-4

def get_target(workload):
    targets = {
        "swaptions": (5.1, 0.7),
        "ferret": (118.2, 15.6),
        "freqmine": (205.0, 27.5),
        "hmmer": (22765.5, 1961.7),
        "copy": (3518.3, 594.3),
        "sphinx3": (32743.3, 3294.7),
        "namd": (27743.9, 2512.4),
        "xz": (30487.7, 4022.6),
        "GemsFDTD": (69334.6, 9777.9),
        "fotonik3d": (69345.7, 9206.1),
        "zeusmp": (90955.5, 14942.0),
        "cactusADM": (33773.1, 3329.5),
        "triad": (4405.8, 823.9),
        "lbm": (64996.2, 12162.8),
        "parest": (42506.4, 5674.6),
    }
    
    if workload not in targets:
        raise ValueError(f"No target value for this workload: {workload}")
    
    return targets[workload]

def calculate_reward(obs, target_latency, target_energy):
    if obs[2] < ERR_LIMIT:
        latency_normalize = target_latency / abs(obs[0] - target_latency)
        energy_normalize = target_energy / abs(obs[1] - target_energy)
        reward = latency_normalize * energy_normalize
        return [reward] * NUM_AGENTS
    else:
        return [-1e6] * NUM_AGENTS


def merge_and_format_rl(workload):
    tg_lat, tg_eg = get_target(workload)
    
    pattern = f"{workload}"
    
    merged_dict = {}
    files = Path(f"{BUFFER_DIR}/rl/").iterdir()
    rl_path_all = [file for file in files if re.search(pattern, file.name)]
    for rl_path in rl_path_all:
        with rl_path.open('rb') as f:
            rl_data = pickle.load(f)
        merged_dict.update(rl_data)
    
    files = Path(f"{BUFFER_DIR}/random/").iterdir()
    rd_path_all = [file for file in files if re.search(pattern, file.name)]
    for rd_path in rd_path_all:
        with open(rd_path, 'rb') as f:
            rd_data = pickle.load(f)
        for key, value in rd_data.items():
            merged_dict[key] = format_for_rl(value, tg_lat, tg_eg) if key not in merged_dict.keys() else merged_dict[key]
    
    return merged_dict

def format_for_rl(rd_data, tg_lat, tg_eg):
    obs, _ = rd_data
    reward = calculate_reward(obs, tg_lat, tg_eg)
    return obs, reward


def merge_and_format_rd(workload):
    pattern = f"{workload}"
    
    merged_dict = {}
    files = Path(f"{BUFFER_DIR}/random/").iterdir()
    rd_path_all = [file for file in files if re.search(pattern, file.name)]
    for rd_path in rd_path_all:
        with rd_path.open('rb') as f:
            random_data = pickle.load(f)
        merged_dict.update(random_data)
    
    files = Path(f"{BUFFER_DIR}/rl/").iterdir()
    rl_path_all = [file for file in files if re.search(pattern, file.name)]
    for rl_path in rl_path_all:
        with open(rl_path, 'rb') as f:
            rl_data = pickle.load(f)
        for key, value in rl_data.items():
            merged_dict[key] = format_for_rd(value) if key not in merged_dict.keys() else merged_dict[key]
            
    return merged_dict

def format_for_rd(rd_data):
    obs, _ = rd_data
    return obs, -1

def save_mapping(all_mapping, save_dir):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    for workload, mapping in all_mapping.items():
        with open(f"{save_dir}/{workload}.pkl", "wb") as f:
            pickle.dump(mapping, f)


def parallel_merge(workload_all, merge_func, num_workers=4):
    """
    使用多線程並行合併工作負載。
    
    參數:
    - workload_all: list，所有工作負載的標識符列表。
    - merge_func: function，合併函數（merge_and_format_rl 或 merge_and_format_rd）。
    - num_workers: int，並行線程數量。
    
    返回:
    - all_mapping: dict，所有工作負載的合併映射。
    """
    
    def process_workload(workload, merge_func):
        try:
            result = merge_func(workload)
            return (workload, result, None)
        except Exception as e:
            return (workload, None, e)

    all_mapping = {}
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # 提交所有工作負載到執行器
        futures = {executor.submit(process_workload, wl, merge_func): wl for wl in workload_all}
        
        # 使用 tqdm 顯示進度條
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {merge_func.__name__}"):
            workload, result, error = future.result()
            if error:
                print(f"Failed to process workload {workload}: {error}")
            else:
                all_mapping[workload] = result
    return all_mapping

def build_buffer_v6(workload_all, input_dir, output_dir, 
                   old_reward=-1e6, new_reward=0.0, num_workers=4):
    """
    處理工作負載的 Pickle 文件，替換特定的獎勵值，並保存到新的文件中。

    參數:
    - workload_all: list，工作負載標識符（文件名，不帶擴展名）。
    - input_dir: str，輸入 Pickle 文件的目錄。
    - output_dir: str，輸出 Pickle 文件的目錄。
    - old_reward: float，需要替換的舊獎勵值。
    - new_reward: float，新的獎勵值。
    - num_workers: int，並行處理的工作線程數。
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)  # 確保輸出目錄存在

    def process_workload(workload):
        try:
            # 定義輸入和輸出的文件路徑
            i_file = input_path / f"{workload}.pkl"
            o_file = output_path / f"{workload}.pkl"

            # 加載輸入文件中的映射
            with i_file.open("rb") as f:
                mapping = pickle.load(f)

            # 假設 mapping 是一個字典；如果不是，需根據實際情況調整
            if isinstance(mapping, dict):
                # 使用字典推導式高效更新映射
                updated_mapping = {k: (obs, [new_reward] * NUM_AGENTS if reward[0] <= old_reward else reward)
                                   for k, (obs, reward) in mapping.items()}
            else:
                raise "Not dictionary"

            # 使用臨時文件確保寫入過程的原子性
            with tempfile.NamedTemporaryFile("wb", delete=False, dir=output_path) as tmp_f:
                pickle.dump(updated_mapping, tmp_f)
                temp_name = tmp_f.name

            # 將臨時文件重命名為最終的輸出文件
            os.replace(temp_name, o_file)

            return (workload, True, None)
        except Exception as e:
            return (workload, False, str(e))

    # 使用 ThreadPoolExecutor 進行並行處理（適用於 I/O 密集型操作）
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # 提交所有工作負載到執行器
        futures = {executor.submit(process_workload, wl): wl for wl in workload_all}

        # 使用 tqdm 進度條監控處理進度
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing workloads"):
            workload, success, error = future.result()
            if not success:
                print(f"Failed to process workload {workload}: {error}")

def check_buffer_v6(workload_all, input_dir, old_reward=-1e6, num_workers=4):
    
    input_path = Path(input_dir)

    def process_workload(workload):
        try:
            # 定義輸入和輸出的文件路徑
            i_file = input_path / f"{workload}.pkl"

            # 加載輸入文件中的映射
            with i_file.open("rb") as f:
                mapping = pickle.load(f)

            # 假設 mapping 是一個字典；如果不是，需根據實際情況調整
            if isinstance(mapping, dict):
                # 使用字典推導式高效更新映射
                for k, (obs, reward) in mapping.items():
                    for r in reward:
                        if r <= old_reward:
                            print(r)
                            raise "Wrong Reward"
            else:
                raise "Not dictionary"

            return (workload, True, None)
        except Exception as e:
            return (workload, False, str(e))

    # 使用 ThreadPoolExecutor 進行並行處理（適用於 I/O 密集型操作）
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # 提交所有工作負載到執行器
        futures = {executor.submit(process_workload, wl): wl for wl in workload_all}

        # 使用 tqdm 進度條監控處理進度
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing workloads"):
            workload, success, error = future.result()
            if not success:
                print(f"Failed to process workload {workload}: {error}")


if __name__ == "__main__":
    is_merging = True
    is_build_v6 = False
    BUFFER_DIR = "buffer_v6/"
    
    workload_all = ["copy", "hmmer", "swaptions", "ferret", "freqmine", "fotonik3d", "GemsFDTD", "namd", "xz", "sphinx3", "cactusADM", "triad", "zeusmp", "lbm", "parest"]
    #workload_all = ["parest"]
    if is_merging:
        # Merge for RL
        all_mapping_rl = parallel_merge(workload_all, merge_and_format_rl)
        save_mapping(all_mapping_rl, f"{BUFFER_DIR}/merge_for_rl")

        # Merge for Random
        all_mapping_rd = parallel_merge(workload_all, merge_and_format_rd)
        save_mapping(all_mapping_rd, f"{BUFFER_DIR}/merge_for_rd")
    
    if is_build_v6:
        build_buffer_v6(workload_all, input_dir="buffer/merge_for_rl", output_dir="buffer_v6/rl")
        check_buffer_v6(workload_all, input_dir="buffer_v6/rl")