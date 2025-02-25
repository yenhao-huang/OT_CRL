import pandas as pd
import pickle
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler

class DRAM_ExpBuffer():
    def __init__(self, expertdata_path, workload, log_dir):
        self.pagepolicy = {0 : 0, 1 : 1}
        self.addressmapping = {0 : 0, 1 : 1, 2 : 2, 3 : 3, 4 : 4}
        self.rcd_mapper = dict([(config, i) for i, config in enumerate(range(12, 17))])
        self.rp_mapper = dict([(config, i) for i, config in enumerate(range(12, 17))])
        self.ras_mapper = dict([(config, i) for i, config in enumerate(range(19, 40))])
        self.rrd_mapper = dict([(config, i) for i, config in enumerate(range(3, 7))])
        self.refi_mapper = dict([(config, i) for i, config in enumerate(range(4680, 468000, 46800))])
        self.workload = workload
        self.expertdata_path = expertdata_path
        self.log_dir = log_dir

    def _one_hot(self, index, num_classes):
        """Generate a one-hot encoded vector with num_classes elements."""
        one_hot_vector = np.zeros(num_classes, dtype=float)
        one_hot_vector[index] = 1.0
        return one_hot_vector

    def action_to_model_format(self, action):
        pp = self.pagepolicy[action[0]]
        am = self.addressmapping[action[1]]
        rcd = self.rcd_mapper[action[2]]
        rp = self.rp_mapper[action[3]]
        ras = self.ras_mapper[action[4]]
        rrd = self.rrd_mapper[action[5]]
        refi = self.refi_mapper[action[6]]
        
        return [pp, am, rcd, rp, ras, rrd, refi]

    def to_action_prob(self, action):
        pp = self.pagepolicy[action[0]]
        am = self.addressmapping[action[1]]
        rcd = self.rcd_mapper[action[2]]
        rp = self.rp_mapper[action[3]]
        ras = self.ras_mapper[action[4]]
        rrd = self.rrd_mapper[action[5]]
        refi = self.refi_mapper[action[6]]
        
        pp_prob = self._one_hot(pp, 2)
        am_prob = self._one_hot(am, 5)
        ras_prob = self._one_hot(ras, 21)
        rcd_prob = self._one_hot(rcd, 5)
        refi_prob = self._one_hot(refi, 10)
        rp_prob = self._one_hot(rp, 5)
        rrd_prob = self._one_hot(rrd, 4)
        
        return [pp_prob, am_prob, rcd_prob, rp_prob, ras_prob, rrd_prob, refi_prob]

    def _filter_uname(self, df):
        df = df.rename(columns={"Unnamed: 0":"stats"})
        df.index = df["stats"]
        df = df.drop(["stats"], axis=1)
        df = df.T
        return df

    def get_initial_state(self):
        default_perf = pd.read_csv("/home/user/Desktop/oss-arch-gym/sims/DRAM/expert_data/offline_config_info.csv")
        default_perf = self._filter_uname(default_perf)
        init_state = default_perf.loc[default_perf.index == self.workload][["Latency", "Energy", "Error"]]
        return init_state.to_numpy()

    def save_expertdata(self):
        state_cols = ["latency", "energy", "err_rate"]
        action_cols = ["pagepolicy", "addressmapping", "rcd", "rp", "ras", "rrd", "refi"]
        is_sample, n_samples = True, 20
        dir_path = Path(f"{self.log_dir}/actor/")
        if dir_path.exists() and dir_path.is_dir():
            # Get the list of files in the directory
            subdirs = [d for d in dir_path.iterdir() if d.is_dir()]  # You can specify a pattern, e.g., '*.log' for log files
            path = subdirs[0] / "logs/actor/logs.csv"
        else:
            raise "No this log"
        
        # get current data
        df = pd.read_csv(path)

        ### get label
        df["edp"] = df["latency"] * df["energy"]
        select_df = df[df["err_rate"] < 1e-4]
        action_idx = select_df["edp"].idxmin()
        action = df.loc[action_idx][action_cols]
        action_prob = self.to_action_prob(action)
        actprob_all = []
        for _ in range(n_samples):
            actprob_all.append(action_prob)
            
        actprob_all = np.array(actprob_all, dtype=object)
        
        ### get state
        if is_sample:
            state = df[state_cols].sample(n=n_samples-1,  random_state=5)
        else:
            state = df[state_cols]
        init_state = self.get_initial_state()
        state = state.to_numpy()
        state = np.concatenate([init_state, state])
        
        # get prior data
        expdata_path = Path(self.expertdata_path)
        if expdata_path.exists():
            with expdata_path.open('rb') as file:
                prior_expdata = pickle.load(file)
            prior_state, prior_actprob, _ = prior_expdata
            
            # merge
            state = np.concatenate([prior_state, state])
            actprob_all = np.concatenate([prior_actprob, actprob_all])
            
        # scaler
        scaler = StandardScaler()
        scaler.fit(state)
        metadata = {"len" : len(state), "scaler" : scaler}
        data = (state, actprob_all, metadata)
        
        with expdata_path.open('wb') as file:
            pickle.dump(data, file)