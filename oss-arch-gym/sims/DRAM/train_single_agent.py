import os
import sys
import json
from typing import Optional
import time

os.sys.path.insert(0, os.path.abspath('../../'))
from configs import arch_gym_configs
os.sys.path.insert(0, os.path.abspath('../../acme'))
print(os.sys.path)
import envlogger
from acme.agents.jax import ppo
from acme.agents.jax import sac
from acme import wrappers
from acme import specs

import helpers
from absl import app
from absl import flags
from absl import logging
from acme.utils import lp_utils
from acme.jax import experiments
from acme.agents.jax import normalization

from acme.utils.loggers.tf_summary import TFSummaryLogger
from acme.utils.loggers.terminal import TerminalLogger
from acme.utils.loggers.csv import CSVLogger
from acme.utils.loggers import aggregators
from acme.utils.loggers import base

from arch_gym.envs import dramsys_wrapper_rl
import launchpad as lp

print("Import Successful")

FLAGS = flags.FLAGS

# Workload to run for training

#_DRAM_TRACE = flags.DEFINE_string('dram_trace', 'swaptions', 'Workload to run for training')

workload = flags.DEFINE_string('workload', 'swaptions', 'Workload to run for training')

# select which RL algorithm to use 
_RL_AGO = flags.DEFINE_string('rl_algo', 'ppo', 'RL algorithm.')

# select which RL form to use
_RL_FORM = flags.DEFINE_string('rl_form', 'sa', 'RL form.')

# Hyperparameters for the environment


# Hyperparameters for each RL algorithm
_NUM_STEPS = flags.DEFINE_integer('num_steps', 100, 'Number of training steps.')
_NUM_EPISODES = flags.DEFINE_integer('num_episodes', 100, 'Number of training steps.')
_N_STEPS = flags.DEFINE_integer('step_per_episode', 1, 'Number of training steps.')
_THREAD_IDX = flags.DEFINE_integer('threadidx', 0, 'Number of training steps.')

_EVAL_EVERY = flags.DEFINE_integer('eval_every', 1, 'Number of evaluation steps.')
_EVAL_EPISODES = flags.DEFINE_integer('eval_episodes', 0, 'Number of evaluation episode.') # 0: only train
_SEED = flags.DEFINE_integer('seed', 1, 'Random seed.')
_LEARNING_RATE = flags.DEFINE_float('learning_rate', 2e-5, 'Learning rate.')
# Acceptable values for reward: power, latency, and both (both means latency & power)
_REWARD_FORM = flags.DEFINE_string('reward_form', 'both', 'Reward form.')
# Scale reward
_REWARD_SCALE = flags.DEFINE_string('reward_scale', 'false', 'Scale reward.')
# Hyperparameters for PPO 
_ENTROPY_COST = flags.DEFINE_float('entropy_cost', 0.1, 'Entropy cost.')
_PPO_CLIPPING_EPSILON = flags.DEFINE_float('ppo_clipping_epsilon', 0.2, 'PPO clipping epsilon.')
_CLIP_VALUE = flags.DEFINE_bool('clip_value', False, 'Clip value.')

# Hyperparameters for SAC
_N_STEP = flags.DEFINE_integer('n_step', 1, 'Number of gradient steps.')

# Experiment setup related parameters
_SUMMARYDIR = flags.DEFINE_string('summarydir', './logs', 'Directory to save summaries.')
_ENVLOGGER_DIR = flags.DEFINE_string('envlogger_dir', 'trajectory', 'Directory to save envlogger.')
_USE_ENVLOGGER = flags.DEFINE_bool('use_envlogger', False, 'Use envlogger.')
_RUN_DISTRIBUTED = flags.DEFINE_bool(
    'run_distributed', False, 'Should an agent be executed in a '
    'distributed way (the default is a single-threaded agent)')


_EXP_NAME = flags.DEFINE_string('exp_name', 'ppo_1agent', 'exp name')
_LOAD_CHECKPOINT = flags.DEFINE_bool('load_checkpoint', False, 'load checkpoint')
_CHECKPOINT_DIR = flags.DEFINE_string('checkpoint_dir', "~/acme", 'checkpoint dir')

class CheckpointingConfig:
  max_to_keep: int = 1
  directory: str = '~/acme'
  add_uid: bool = True
  replay_checkpointing_time_delta_minutes: Optional[int] = None
  time_delta_minutes: int = 5
  load_checkpoint: False

def update_thread():
    with open("/home/user/Desktop/oss-arch-gym/sims/DRAM/env_spec_ramulator/thread.txt", "w") as f:
        f.writelines([str(FLAGS.threadidx)])

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

def get_directory_name():
    _EXP_NAME = 'Thread_{}_Workload_{}_Algo_{}_rlform_{}_num_steps_{}_seed_{}_lr_{}_entropy_{}_rewardscale_{}'.format(FLAGS.threadidx, workload.value, _RL_AGO.value, _RL_FORM.value,_NUM_STEPS.value, _SEED.value, _LEARNING_RATE.value, _ENTROPY_COST.value, _REWARD_SCALE.value)
    
    return _EXP_NAME


def wrap_in_envlogger(env, envlogger_dir):
    metadata = {
        'agent_type': FLAGS.rl_algo,
        'rl_form': FLAGS.rl_form,
        'num_steps': FLAGS.num_steps,
        'env_type': type(env).__name__,
    }
    env = envlogger.EnvLogger(env,
                        data_directory = envlogger_dir,
                        metadata = metadata,
                        max_episodes_per_file = 1000)
    return env


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
  elif logger_label == 'learner':
      terminal_logger = TerminalLogger(label=logger_label, print_fn=logging.info)
      summarydir = os.path.join(FLAGS.summarydir,_EXP_NAME, logger_label)
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



def build_experiment_config():
    """Builds the experiment configuration."""

    env = dramsys_wrapper_rl.make_dramsys_env(
            workload=FLAGS.workload,
            rl_form=FLAGS.rl_form,
            reward_formulation = _REWARD_FORM.value,
            reward_scaling = _REWARD_SCALE.value,
            max_steps = FLAGS.step_per_episode)
    if FLAGS.use_envlogger:
        envlogger_dir = os.path.join(FLAGS.summarydir, get_directory_name(), FLAGS.envlogger_dir)
        if(not os.path.exists(envlogger_dir)):
            os.makedirs(envlogger_dir)
        env = wrap_in_envlogger(env, envlogger_dir)
    
    env_spec = specs.make_environment_spec(env)

    if FLAGS.rl_algo == 'ppo':
        config = ppo.PPOConfig(entropy_cost=FLAGS.entropy_cost,
                            learning_rate=FLAGS.learning_rate,
                            ppo_clipping_epsilon=FLAGS.ppo_clipping_epsilon,
                            clip_value=FLAGS.clip_value,
                            )
        ppo_builder = ppo.PPOBuilder(config)

        layer_sizes = (32, 32, 32)
        make_eval_policy = lambda network: ppo.make_inference_fn(network, True)


        checkpointing = CheckpointingConfig()
        checkpointing.directory = FLAGS.checkpoint_dir
        checkpointing.load_checkpoint = FLAGS.load_checkpoint
        
        return experiments.ExperimentConfig(
            builder=ppo_builder,
            environment_factory=lambda seed: env,
            network_factory=lambda spec: ppo.make_networks(env_spec, layer_sizes),
            policy_network_factory = ppo.make_inference_fn,
            eval_policy_network_factory = make_eval_policy,
            seed = FLAGS.seed,
            logger_factory=_logger_factory,
            max_num_actor_steps=FLAGS.num_steps,
            max_num_actor_episodes=FLAGS.num_episodes,
            checkpointing=checkpointing,
            )
    elif FLAGS.rl_algo == 'sac':
        config = sac.SACConfig(
            learning_rate=FLAGS.learning_rate,
            n_step=FLAGS.n_step,
        )
        sac_builder = sac.builder.SACBuilder(config)
        
        return experiments.ExperimentConfig(
            builder = sac_builder,
            environment_factory = lambda seed: env,
            network_factory = lambda spec: sac.make_networks(env_spec, (32, 32, 32)),
            seed = FLAGS.seed,
            logger_factory = _logger_factory,
            max_num_actor_steps = FLAGS.num_steps)
    else:
        raise ValueError(f'Improper value for rl_algo. rl_algo is {FLAGS.rl_algo}')


def main(_):
  start_time = time.time()
  update_main_config()

  sim_config = arch_gym_configs.sim_config
  #update_sim_configs(sim_config, _DRAM_TRACE.value)
  config = build_experiment_config()
  if FLAGS.run_distributed:
    program = experiments.make_distributed_experiment(
        experiment=config, num_actors=4)
    lp.launch(program, xm_resources=lp_utils.make_xm_docker_resources(program))
  else:
    experiments.run_experiment(
        experiment=config,
        eval_every=FLAGS.eval_every,
        num_eval_episodes=FLAGS.eval_episodes)
  elapsed_time = time.time() - start_time
  print(f"Elapsed time: {elapsed_time:.2f} seconds")
  

if __name__ == '__main__':
   app.run(main)
