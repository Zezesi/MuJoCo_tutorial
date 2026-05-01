
import os
import tensorboardX
import torch
import tqdm


import numpy as np

import utils
import nets
from scipy.io import savemat
import time
import mujoco
import mujoco.viewer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class panda_env:
    def __init__(self, action_space_size=8, state_space_size=15, num_of_attack_points=1):
        self.action_space_size = action_space_size
        self.state_space_size = state_space_size
        self.attack_point = np.array([0.0, 0.0, 0.0])
        self.num_of_attack_points = num_of_attack_points
        self.model = mujoco.MjModel.from_xml_path('franka_emika_panda/mjx_scene.xml')
        self.data = mujoco.MjData(self.model)

    def update_attack_point(self):
        x = np.random.uniform(low=-0.2, high=0.2)
        y = np.random.uniform(low=0.2, high=0.5)
        z = np.random.uniform(low=0.2, high=0.6)
        self.attack_point = np.array([x, y, z])

    def action_post_processing(self, action):
        action[0] = (action[0] - (-1)) * ((2.8973) - (-2.8973)) / ((1) - (-1)) + (-2.8973)
        action[1] = (action[1] - (-1)) * ((1.7628) - (-1.7628)) / ((1) - (-1)) + (-1.7628)
        action[2] = (action[2] - (-1)) * ((2.8973) - (-2.8973)) / ((1) - (-1)) + (-2.8973)
        action[3] = (action[3] - (-1)) * ((-0.0698) - (-3.0718)) / ((1) - (-1)) + (-3.0718)
        action[4] = (action[4] - (-1)) * ((2.8973) - (-2.8973)) / ((1) - (-1)) + (-2.8973)
        action[5] = (action[5] - (-1)) * ((3.7525) - (-0.0175)) / ((1) - (-1)) + (-0.0175)
        action[6] = (action[6] - (-1)) * ((np.pi) - (-np.pi)) / ((1) - (-1)) + (-np.pi)
        action[7] = (action[7] - (-1)) * ((0.04) - (0)) / ((1) - (-1)) + (0)
        return action

    def reward(self, action):
        curPos = self.data.site("gripper").xpos
        tarPos = self.data.site("attack_point0").xpos
        reward = np.array([-np.linalg.norm(curPos - tarPos)])
        if self.data.ncon > 0:
            reward[0] = reward[0] - 1
        return reward

    def step(self, action):
        self.data.ctrl = action
        mujoco.mj_step(self.model, self.data)
        state = np.concatenate((self.data.site("gripper").xpos, self.data.qpos / 3.0718), axis=0)
        for i_ in range(self.num_of_attack_points):
            attack_point_name = f"attack_point{i_}"
            state = np.concatenate((self.model.site(attack_point_name).pos, state), axis=0)
        reward = self.reward(action)
        return state, reward

    def reset(self):
        self.data.qpos = [-5.94958683e-17, 5.57178318e-03, -6.85235486e-06, -6.95284621e-02, -1.61440323e-04,
                          -7.17258051e-03, -5.46813142e-06, 6.91022958e-07, -9.37611953e-08]
        for i_ in range(self.num_of_attack_points):
            attack_point_name = f"attack_point{i_}"
            self.update_attack_point()
            self.model.site(attack_point_name).pos = self.attack_point
        mujoco.mj_step(self.model, self.data)
        time.sleep(0.1)  # make sure that it has enough time to reset in mujoco
        state = np.concatenate((self.data.site("gripper").xpos, self.data.qpos / 3.0718), axis=0)
        for i_ in range(self.num_of_attack_points):
            attack_point_name = f"attack_point{i_}"
            state = np.concatenate((self.model.site(attack_point_name).pos, state), axis=0)
        done = 1
        return state, done


class SACAgent:
    def __init__(
        self,
        state_space_size=15,
        action_space_size=8,
        log_std_low=-20,
        log_std_high=2,
        actor_net_cls=nets.StochasticActor,
        critic_net_cls=nets.BigCritic,
        hidden_size=1024,
    ):
        self.actor = actor_net_cls(
            state_space_size=state_space_size,
            action_space_size=action_space_size,
            log_std_low=log_std_low,
            log_std_high=log_std_high,
            hidden_size=hidden_size,
        )
        self.critic1 = critic_net_cls(state_space_size=state_space_size, action_space_size=action_space_size, hidden_size=hidden_size)
        self.critic2 = critic_net_cls(state_space_size=state_space_size, action_space_size=action_space_size, hidden_size=hidden_size)

    def to(self, device):
        self.actor = self.actor.to(device)
        self.critic1 = self.critic1.to(device)
        self.critic2 = self.critic2.to(device)

    def eval(self):
        self.actor.eval()
        self.critic1.eval()
        self.critic2.eval()

    def train(self):
        self.actor.train()
        self.critic1.train()
        self.critic2.train()

    def save(self, path):
        actor_path = os.path.join(path, "actor.pt")
        critic1_path = os.path.join(path, "critic1.pt")
        critic2_path = os.path.join(path, "critic2.pt")
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic1.state_dict(), critic1_path)
        torch.save(self.critic2.state_dict(), critic2_path)
        torch.save(self.actor.state_dict(), "actor.pt")
        torch.save(self.critic1.state_dict(), "critic1.pt")
        torch.save(self.critic2.state_dict(), "critic2.pt")

    def load(self, path):
        actor_path = os.path.join(path, "actor.pt")
        critic1_path = os.path.join(path, "critic1.pt")
        critic2_path = os.path.join(path, "critic2.pt")
        self.actor.load_state_dict(torch.load(actor_path))
        self.critic1.load_state_dict(torch.load(critic1_path))
        self.critic2.load_state_dict(torch.load(critic2_path))


    def sample_action(self, state, from_cpu=True):
        if from_cpu:
            state = self.process_state(state)
        self.actor.eval()
        with torch.no_grad():
            act_dist = self.actor.forward(state)
            act = act_dist.sample()
        self.actor.train()
        if from_cpu:
            act = self.process_act(act)
        return act

    def process_state(self, state):
        return torch.from_numpy(np.expand_dims(state, 0).astype(np.float32)).to(utils.device)

    def process_act(self, act):
        return np.squeeze(act.clamp(-1.0, 1.0).cpu().numpy(), 0)


def sac(
        env_env,
        log_std_low=-20,
        log_std_high=2,
        state_space_size=15,
        action_space_size=8,
        num_of_episodes=1,
        max_episode_steps=5000,
        name="sac",
):
    agent = SACAgent(log_std_low=log_std_low,
                     log_std_high=log_std_high,
                     state_space_size=state_space_size,
                     action_space_size=action_space_size,
                     actor_net_cls=nets.StochasticActor,
                     critic_net_cls=nets.BigCritic,
                     hidden_size=1024)


    save_dir = utils.make_process_dirs(name,base_path="eval")
    writer = tensorboardX.SummaryWriter(save_dir)
    writer.add_hparams(locals(), {})
    writer.close()  # very important, otherwise, error pops

    ###########
    ## SETUP ##
    ###########
    agent.load(path='')
    agent.to(device)
    agent.eval()

    episode = 0

    while viewer.is_running() and episode < num_of_episodes:
        ################
        ## PRINT INFO ##
        ################
        print(f"No.{episode + 1} Episode: Evaluating Deep Control Soft Actor Critic")

        path = os.path.join(save_dir, f'reward_ep{episode}.mat')
        reward_record = []
        path1 = os.path.join(save_dir, f'state_ep{episode}.mat')
        state_record = []
        path2 = os.path.join(save_dir, f'action_ep{episode}.mat')
        action_record = [[]]

        ###################
        ## TRAINING LOOP ##
        ###################
        with torch.no_grad():
            state, done = eval_env.reset()
            steps_iter = range(max_episode_steps)
            steps_iter = tqdm.tqdm(steps_iter)
            for step in steps_iter:
                step_start = time.time()
                action = agent.sample_action(state)
                action = eval_env.action_post_processing(action)
                next_state, reward = eval_env.step(action)
                reward_record.append(reward.item())
                state_record.append(next_state)
                action_record = np.concatenate((action_record, [action]), axis=1)
                state = next_state

                with viewer.lock():
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(eval_env.data.time % 2)

                viewer.sync()

                if (step + 1) >= max_episode_steps:
                    done = 1
            if done:
                df = {'reward': np.array(reward_record)}
                savemat(path, df)
                df1 = {'state': state_record}
                savemat(path1, df1)
                df2 = {'action': np.array(action_record)}
                savemat(path2, df2)
                episode += 1
                continue




if __name__ == "__main__":
    max_episode_steps = 5000
    num_of_episodes = 1
    eval_env = panda_env(action_space_size=8, state_space_size=15, num_of_attack_points=1)

    with mujoco.viewer.launch_passive(eval_env.model, eval_env.data) as viewer:
        time.sleep(2)  # wait 2 seconds

        sac(
            eval_env,
            log_std_low=-20,
            log_std_high=2,
            state_space_size=15,
            action_space_size=8,
            num_of_episodes=1,
            max_episode_steps=5000,
            name="sac",
        )
