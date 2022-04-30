import dreamerv2.common as common
import dreamerv2.agent as agent
from ray.tune.schedulers import pb2
from ray import tune
import os
import logging
import warnings
import re
import pathlib
import numpy as np
import tensorflow as tf
import collections
import ruamel_yaml as yaml
import argparse


def train(config, checkpoint_dir=None):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    logging.getLogger().setLevel('ERROR')
    warnings.filterwarnings('ignore', '.*box bound precision lowered.*')

    logdir = pathlib.Path(tune.get_trial_dir()).expanduser()
    for k, v in  config.items():
        if type(v) is np.float32:
            config[k] = float(v)
    config = common.Config(config)
    config.save(logdir / "config.yaml")

    def make_env(config):
        env = common.DMC(config.task, config.action_repeat,
            config.render_size, config.dmc_camera)
        env = common.NormalizeAction(env)
        return env

    outputs = [
        common.TerminalOutput(),
        common.JSONLOutput(logdir),
        common.TensorBoardOutput(logdir),
    ]

    tf.config.run_functions_eagerly(not config.jit)
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
    if config.precision == 16:
        from tensorflow.keras.mixed_precision import experimental as prec
        prec.set_policy(prec.Policy('mixed_float16'))

    train_replay = common.Replay(logdir / 'train_episodes', **config["replay"])
    eval_replay = common.Replay(logdir / 'eval_episodes', **dict(
        capacity=config.replay.capacity // 10,
        minlen=config.dataset.length,
        maxlen=config.dataset.length))
    step = common.Counter(train_replay.stats['total_steps'])
    logger = common.Logger(step, outputs, multiplier=config.action_repeat)
    metrics = collections.defaultdict(list)

    should_train = common.Every(config.train_every)
    should_log = common.Every(config.log_every)
    should_video_train = common.Every(config.eval_every)
    should_video_eval = common.Every(config.eval_every)
    should_expl = common.Until(config.expl_until // config.action_repeat)
    
    train_env = make_env(config)
    obs_space = train_env.obs_space
    act_space = train_env.act_space
    print('Create agent.')
    agnt = agent.Agent(config, obs_space, act_space, step)

    def per_episode(ep, mode):
        length = len(ep['reward']) - 1
        score = float(ep['reward'].astype(np.float64).sum())
        print(f'{mode.title()} episode has {length} steps and return {score:.1f}.')
        logger.scalar(f'{mode}_return', score)
        logger.scalar(f'{mode}_length', length)
        for key, value in ep.items():
            if re.match(config.log_keys_sum, key):
                logger.scalar(f'sum_{mode}_{key}', ep[key].sum())
            if re.match(config.log_keys_mean, key):
                logger.scalar(f'mean_{mode}_{key}', ep[key].mean())
            if re.match(config.log_keys_max, key):
                logger.scalar(f'max_{mode}_{key}', ep[key].max(0).mean())
        should = {'train': should_video_train, 'eval': should_video_eval}[mode]
        if should(step):
            for key in config.log_keys_video:
                logger.video(f'{mode}_policy_{key}', ep[key])
        replay = dict(train=train_replay, eval=eval_replay)[mode]
        logger.add(replay.stats, prefix=mode)
        logger.write()

        if mode =="eval":
            with tune.checkpoint_dir(step=step.value) as checkpoint_dir:
                path = pathlib.Path(checkpoint_dir) / 'variables.pkl'
                agnt.save(path)
                print("save check point to", path)
            tune.report(eval_reward=score)

    train_driver = common.Driver([train_env])
    train_driver.on_episode(lambda ep: per_episode(ep, mode='train'))
    train_driver.on_step(lambda tran, worker: step.increment())
    train_driver.on_step(train_replay.add_step)
    train_driver.on_reset(train_replay.add_step)

    eval_driver = common.Driver([make_env(config)])
    eval_driver.on_episode(lambda ep: per_episode(ep, mode='eval'))
    eval_driver.on_episode(eval_replay.add_episode)

    prefill = max(0, config.prefill - train_replay.stats['total_steps'])
    if prefill:
        print(f'Prefill dataset ({prefill} steps).')
        random_agent = common.RandomAgent(act_space)
        train_driver(random_agent, steps=prefill, episodes=1)
        eval_driver(random_agent, episodes=1)
        train_driver.reset()
        eval_driver.reset()
    
    train_dataset = iter(train_replay.dataset(**config.dataset))
    report_dataset = iter(train_replay.dataset(**config.dataset))
    eval_dataset = iter(eval_replay.dataset(**config.dataset))
    train_agent = common.CarryOverState(agnt.train)
    train_agent(next(train_dataset))
    if checkpoint_dir:
        variables_path = pathlib.Path(checkpoint_dir) / "variables.pkl"
        agnt.load(variables_path)
    else:
        print('Pretrain agent.')
        for _ in range(config["pretrain"]):
            train_agent(next(train_dataset))
    train_policy = lambda *args: agnt.policy(
        *args, mode='explore' if should_expl(step) else 'train')
    eval_policy = lambda *args: agnt.policy(*args, mode='eval')

    def train_step(tran, worker):
        if should_train(step):
            for _ in range(config.train_steps):
                mets = train_agent(next(train_dataset))
                [metrics[key].append(value) for key, value in mets.items()]
        if should_log(step):
            for name, values in metrics.items():
                logger.scalar(name, np.array(values, np.float64).mean())
                metrics[name].clear()
            logger.add(agnt.report(next(report_dataset)), prefix="train")
            logger.write(fps=True)

    train_driver.on_step(train_step)

    print("start training")
    while step < config.steps:
        logger.write()
        logger.add(agnt.report(next(eval_dataset)), prefix='eval')
        eval_driver(eval_policy, episodes=config.eval_eps)
        train_driver(train_policy, steps=config.eval_every)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-N", type=int, default=1)
    parser.add_argument("--task", type=str, default="walker_walk",
        choices=["walker_walk", "cheetah_run", "hopper_hop", 
            "acrobot_swingup", "finger_spin", "reacher_hard"])
    args = parser.parse_args()

    with open("configs.yaml", "r") as f:
        base_config = yaml.safe_load(f)
    config = common.Config(base_config["defaults"])
    dmc_vision = common.Config(base_config["dmc_vision"])
    config = dict(config.update(dmc_vision))
    config.update({
        "task": args.task,
        "steps": 1000000,
        "imag_horizon": tune.randint(5, 50),
        "log_every": 10000,
        "actor_grad": "both",
        "actor_grad_mix": tune.uniform(0, 1),
    })

    scheduler = pb2.PB2(
        time_attr="training_iteration",
        metric="eval_reward", mode="max",
        perturbation_interval=3,
        # burn_in_period=5,
        # hyperparam_mutations={
        #     "imag_horizon": tune.randint(5, 50),
        #     "actor_grad_mix": lambda: np.clip(np.random.uniform(-0.2, 1.2), 0, 1),
        # }
        hyperparam_bounds={
            "imag_horizon" : [5, 50],
            "actor_grad_mix": [0, 1],
        }
    )

    reporter = tune.CLIReporter(max_report_frequency=60)

    result = tune.run(
        train,
        config=config,
        scheduler=scheduler,
        local_dir=os.path.join("pbt_results", args.task),
        resources_per_trial={"cpu": 1, "gpu": 2/args.N-0.005},
        keep_checkpoints_num=4,
        num_samples=args.N,
        verbose=3,
        progress_reporter=reporter,
        reuse_actors=True,
    )

    print(result.get_best_config("eval_reward", "max"))
