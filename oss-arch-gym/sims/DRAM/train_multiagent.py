# python3
# Copyright 2018 DeepMind Technologies Limited. All rights reserved.
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


from absl import flags
from absl import logging
import os
import re
from typing import Optional
import sys
os.sys.path.insert(0, os.path.abspath('../../'))
os.sys.path.insert(0, os.path.abspath('../../acme'))

import acme
from acme import specs
from acme.agents.jax.multiagent.decentralized import agents
from acme.agents.jax.multiagent.decentralized import factories
from absl import app

from acme.utils.loggers.tf_summary import TFSummaryLogger
from acme.utils.loggers.terminal import TerminalLogger
from acme.utils.loggers.csv import CSVLogger
from acme.utils.loggers import aggregators
from acme.utils.loggers import base

import helpers
from acme.utils import loggers
from acme.utils import counting
import jax

from configs import arch_gym_configs
from arch_gym.envs import dramsys_wrapper_rl

from expdata import DRAM_ExpBuffer
from pathlib import Path
FLAGS = flags.FLAGS
_NUM_STEPS = flags.DEFINE_integer('num_steps', 101,
                                  'Number of env steps to run training for.')
_EVAL_EPISODES = flags.DEFINE_integer('eval_episodes', 0, 'Number of evaluation episode.') # 0: only train
_EPOSIDE_LEN = flags.DEFINE_integer('step_per_episode', 1, "StepS per episode")
_N_AGENTS = flags.DEFINE_integer('n_agents', 7, 'Number of gradient steps.')
_RL_AGO = flags.DEFINE_string('rl_algo', 'ppo', 'RL algorithm.')
_IS_UPDATE_BUFFER = flags.DEFINE_boolean('is_update_buffer', False, 'update config buffer')
_LD_RP_CKPT_PATH = flags.DEFINE_string('ld_rp_ckpt_path', None, 'load rp_ckpt_path.')
_RP_BUF_SIZE = flags.DEFINE_integer('rp_buf_size', 10, 'rp_buf_size')
_SGD_EPOCH = flags.DEFINE_integer('sgd_epoch', 32, 'sgd epoch per update')
_REG_METHOD = flags.DEFINE_string('reg_method', "None", 'regularization method')
_IS_BC = flags.DEFINE_bool('is_bc', False, "BC or not")
_ONLY_BC = flags.DEFINE_string('only_bc', "false", "only BC loss or not")
_LOAD_OFFLINE_EXPBUFFER = flags.DEFINE_string('is_start_exp', "False", "Start BC or not")
_END_EXPLORE_UPDATEIDX = flags.DEFINE_integer('end_explore_updateidx', 500, "Start BC or not")
_IS_NORM = flags.DEFINE_bool('is_norm', False, "Start BC or not")

# how many time we update
_BATCH_SIZE = flags.DEFINE_integer('batch_size', 128, 'Batch size.')
_SEED = flags.DEFINE_integer('seed', 0, 'Random seed.')
_THREAD_IDX = flags.DEFINE_integer('threadidx', 0, 'Thread idx')
_EPTHREAD_IDX = flags.DEFINE_integer('expbuf_threadidx', 0, 'EP thread idx')
_WORKLOAD = flags.DEFINE_string('workload', 'swaptions', 'Workload to run for training')
_RL_FORM = flags.DEFINE_string('rl_form', 'macme', 'RL form.')
_REWARD_FORM = flags.DEFINE_string('reward_form', 'both', 'Reward form.')
_REWARD_SCALE = flags.DEFINE_string('reward_scale', 'false', 'Scale reward.')

# Hyperparameters for PPO
_LEARNING_RATE = flags.DEFINE_float('learning_rate', 2e-5, 'Learning rate.')
_ENTROPY_COST = flags.DEFINE_float('entropy_cost', 0.1, 'Entropy cost.')
_VALUE_COST = flags.DEFINE_float('value_cost', 1, 'Value cost.')
_BC_COST = flags.DEFINE_float('bc_cost', 1, 'Value cost.')
_NORMALIZE_ADVANTAGE = flags.DEFINE_string('is_norm_adv', "False", 'Normalize advantage')
_NORMALIZE_VALUE = flags.DEFINE_bool('is_norm_value', False, 'Normalize advantage')
_SUMMARYDIR = flags.DEFINE_string('summarydir', './logs', 'Directory to save summaries.')

_LOAD_CHECKPOINT = flags.DEFINE_boolean('load_checkpoint', False, 'only acting.')
_ONLY_ACTING = flags.DEFINE_boolean('only_acting', False, 'only acting.')
_CHECKPOINT_DIR = flags.DEFINE_string('checkpoint_dir', "~/acme", 'checkpoint dir')

_EXP_WORKLOADS = flags.DEFINE_string('exp_workloads', "None", 'checkpoint dir')

def get_directory_name():
    _EXP_NAME = 'Thread_{}_Workload_{}_Algo_{}_rlform_{}_num_steps_{}_seed_{}_rewardscale_{}_batchsize_{}_sgdepoch_{}_rpbufsize_{}_regmethod_{}'.format(
        FLAGS.threadidx, 
        _WORKLOAD.value, 
        _RL_AGO.value, 
        _RL_FORM.value,
        _NUM_STEPS.value,
        _SEED.value,
        _REWARD_SCALE.value,
        _BATCH_SIZE.value,
        _SGD_EPOCH.value,
        _RP_BUF_SIZE.value,
        _REG_METHOD.value)
    return _EXP_NAME

def _logger_factory(logger_label: str, steps_key: Optional[str] = None, task_instance: Optional[int]=0) -> base.Logger:
  """logger factory."""
  _EXP_NAME = get_directory_name()
  if logger_label == 'actor':
      terminal_logger = TerminalLogger(label=logger_label, print_fn=logging.info)
      summarydir = os.path.join(FLAGS.summarydir,_EXP_NAME, logger_label)
      tb_logger = TFSummaryLogger(summarydir, label=logger_label, steps_key=steps_key)
      csv_logger = CSVLogger(summarydir, label=logger_label)
      serialize_fn = base.to_numpy
      logger = aggregators.Dispatcher([terminal_logger, tb_logger, csv_logger], serialize_fn)
      return logger
  elif re.match("learner", logger_label):
      terminal_logger = TerminalLogger(label=logger_label, print_fn=logging.info)
      summarydir = os.path.join(FLAGS.summarydir, _EXP_NAME, 'learner', logger_label)
      tb_logger = TFSummaryLogger(summarydir, label=logger_label, steps_key=steps_key)
      csv_logger = CSVLogger(summarydir, label=logger_label)
      serialize_fn = base.to_numpy
      logger = aggregators.Dispatcher([terminal_logger, tb_logger, csv_logger], serialize_fn)
      return logger
  elif logger_label == 'evaluator':
      terminal_logger = TerminalLogger(label=logger_label, print_fn=logging.info)
      summarydir = os.path.join(FLAGS.summarydir,_EXP_NAME, logger_label)
      tb_logger = TFSummaryLogger(summarydir, label=logger_label, steps_key=steps_key)
      csv_logger = CSVLogger(summarydir, label=logger_label)
      serialize_fn = base.to_numpy
      logger = aggregators.Dispatcher([terminal_logger, tb_logger, csv_logger], serialize_fn)
      return logger
  else:
    raise ValueError(
        f'Improper value for logger label. Logger_label is {logger_label}')

def update_thread():
    with open("/home/user/Desktop/oss-arch-gym/sims/DRAM/env_spec_ramulator/thread.txt", "w") as f:
        f.writelines([str(_THREAD_IDX.value)])

def update_target():
    if FLAGS.workload == "swaptions":
        target_latency, target_energy = 5.1, 0.7
    elif FLAGS.workload == "ferret":
        target_latency, target_energy = 118.2, 15.6
    elif FLAGS.workload == "freqmine":
        target_latency, target_energy = 205.0, 27.5
    elif FLAGS.workload == "hmmer":
        target_latency, target_energy = 22765.5, 1961.7
    elif FLAGS.workload == "copy":
        target_latency, target_energy = 3518.3, 594.3
    elif FLAGS.workload == "sphinx3":
        target_latency, target_energy = 32743.3, 3294.7
    elif FLAGS.workload == "namd":
        target_latency, target_energy = 27743.9, 2512.4
    elif FLAGS.workload == "xz":
        target_latency, target_energy = 30487.7, 4022.6
    elif FLAGS.workload == "GemsFDTD":
        target_latency, target_energy = 69334.6, 9777.9
    elif FLAGS.workload == "fotonik3d":
        target_latency, target_energy = 69345.7, 9206.1
    elif FLAGS.workload == "zeusmp":
        target_latency, target_energy = 90955.5, 14942.0
    elif FLAGS.workload == "lbm":
        target_latency, target_energy = 64996.2, 12162.8
    elif FLAGS.workload == "parest":
        target_latency, target_energy = 42506.4, 5674.6
    elif FLAGS.workload == "triad":
        target_latency, target_energy = 4405.8, 823.9
    elif FLAGS.workload == "cactusADM":
        target_latency, target_energy = 33773.1, 3329.5
    else:
        raise "No target value for this workload"

    with open(f"/home/user/Desktop/oss-arch-gym/sims/DRAM/env_spec_ramulator/target_{_THREAD_IDX.value}.txt", "w") as f:
        f.writelines(["latency,energy\n", f"{target_latency}, {target_energy}"])

def update_main_config():
    update_thread()
    update_target()  

def main(argv):
    update_main_config()

    # Init expdata
    if _ONLY_ACTING.value:
        # Check learn or inference
        expertdata_dir = Path(f'/home/user/Desktop/oss-arch-gym/sims/DRAM/expert_data/thread/')
        expertdata_path = expertdata_dir / f'Thread={_EPTHREAD_IDX.value}.pkl'
    else:
        # Check load offline expdata or not
        if _LOAD_OFFLINE_EXPBUFFER.value == "True":
            expertdata_dir = Path(f'/home/user/Desktop/oss-arch-gym/sims/DRAM/expert_data/thread/')
            if not expertdata_dir.exists():
                expertdata_dir.mkdir(parents=True, exist_ok=True)
            expertdata_path = expertdata_dir / f'Thread={_THREAD_IDX.value}.pkl'

        else:
            expertdata_dir = Path(f'/home/user/Desktop/oss-arch-gym/sims/DRAM/expert_data/exp32/')
            fname = 'expertdata'
            for workload in _EXP_WORKLOADS.value.split(","):
                fname += f"+{workload}"
            expertdata_path = expertdata_dir / f"{fname}_n=50.pkl"

    # Init environemnt
    train_env = dramsys_wrapper_rl.make_dramsys_env(
                workload=_WORKLOAD.value,
                rl_form=_RL_FORM.value,
                reward_formulation = _REWARD_FORM.value,
                reward_scaling = _REWARD_SCALE.value,
                max_steps = _EPOSIDE_LEN.value,
                num_agents = _N_AGENTS.value,
                is_update_buffer = _IS_UPDATE_BUFFER.value,
                expertdata_path = expertdata_path)
    
    train_environment_spec = specs.make_environment_spec(train_env)
    
    # Init agent
    agent_types = {
        str(i): factories.DefaultSupportedAgent.PPO
        for i in range(train_env.num_agents) 
    }
            
    if _NORMALIZE_ADVANTAGE.value == "True":
        norm_advantage = True
    else:
        norm_advantage = False

    ppo_configs = {
        'learning_rate': _LEARNING_RATE.value,
        'entropy_cost': _ENTROPY_COST.value, 
        'batch_size': _BATCH_SIZE.value, 
        'unroll_length':1, 
        'num_minibatches': 1,
        'replay_buffer_size': _RP_BUF_SIZE.value,
        'num_epochs': _SGD_EPOCH.value,
        'reg_method': _REG_METHOD.value,
        "is_bc": _IS_BC.value,
        "only_bc": _ONLY_BC.value,
        "expertdata_path": expertdata_path,
        "normalize_advantage" : norm_advantage,
        "normalize_value" : _NORMALIZE_VALUE.value,
        "value_cost" : _VALUE_COST.value,
        "bc_cost" : _BC_COST.value,
        "end_explore_updateidx" : _END_EXPLORE_UPDATEIDX.value,
    }

    config_overrides = {
        k: ppo_configs for k, _ in agent_types.items()
    }
    train_agents, eval_policy_networks = agents.init_decentralized_multiagent(
            agent_types=agent_types,
            environment_spec=train_environment_spec,
            seed=_SEED.value,
            ld_rp_ckpt_path=_LD_RP_CKPT_PATH.value,
            batch_size=_BATCH_SIZE.value,
            workdir=_CHECKPOINT_DIR.value,
            init_network_fn=helpers.init_default_dram_network,
            config_overrides=config_overrides,
            only_act=FLAGS.only_acting,
            logger_factory=_logger_factory,
            load_checkpoint=FLAGS.load_checkpoint,
    )

    with open(f"rp_ckpt_path/{_THREAD_IDX.value}.txt", "w") as f:
      f.write(train_agents.checkpoint_path)
    
    
    parent_counter = counting.Counter(time_delta=0.)
    train_counter = counting.Counter(parent_counter, prefix='actor', time_delta=0.)
    train_logger = _logger_factory('actor', train_counter.get_steps_key(), 0)
    train_loop = acme.EnvironmentLoop(
        train_env,
        train_agents,
        label='train_loop',
        logger=train_logger)

    eval_env = train_env
    eval_environment_spec = train_environment_spec
    eval_actors = train_agents.builder.make_actor(
        random_key=jax.random.PRNGKey(_SEED.value),
        policy_networks=eval_policy_networks,
        environment_spec=eval_environment_spec,
        variable_source=train_agents
    )
    eval_counter = counting.Counter(parent_counter, prefix='evaluator', time_delta=0.)
    eval_logger = _logger_factory('evaluator', eval_counter.get_steps_key(), 0)
    eval_loop = acme.EnvironmentLoop(
        eval_env,
        eval_actors,
        label='eval_loop',
        logger=eval_logger,
    )
    
    # Train/Inference
    if not FLAGS.only_acting:
        train_loop.run(num_steps=_NUM_STEPS.value)
        
        #save model
        train_agents.save()
        
        #save metadata of expert data 
        if _ONLY_BC.value != "True":
            log_dir = os.path.join(FLAGS.summarydir, get_directory_name())
            data = str(expertdata_path) + "," + _WORKLOAD.value + "," + log_dir
            with open(f"expert_data/metadata/{_THREAD_IDX.value}.txt", "w") as f:
                f.writelines(data)
    else:
        eval_loop.run(num_episodes=_EVAL_EPISODES.value)
        log_dir = os.path.join(FLAGS.summarydir, get_directory_name())
        data = _WORKLOAD.value + "," + log_dir
        with open(f"inference_logs/{_THREAD_IDX.value}.txt", "w") as f:
            f.writelines(data)

    
if __name__ == '__main__':
  app.run(main)