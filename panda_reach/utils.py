
import os
import torch
import random
import numpy as np
from collections import deque
from itertools import chain

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def make_process_dirs(run_name, base_path="dc_saves"):
    base_dir = os.path.join(base_path, run_name)
    i = 0
    while os.path.exists(base_dir + f"_{i}"):
        i += 1
    base_dir += f"_{i}"
    os.makedirs(base_dir)
    return base_dir


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)



class Non_Episodic_ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state = torch.from_numpy(state).unsqueeze(0).type(torch.float32)
        action = torch.from_numpy(action).unsqueeze(0).type(torch.float32)
        reward = torch.from_numpy(reward).unsqueeze(0).type(torch.float32)
        next_state = torch.from_numpy(next_state).unsqueeze(0).type(torch.float32)
        done = torch.from_numpy(np.array([done])).unsqueeze(0).type(torch.float32)
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)


    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            latest_samples= list(self.buffer)
        else:
            latest_samples= list(self.buffer)[-batch_size:]

        states, actions, rewards, next_states, dones = zip(*latest_samples)

        # Convert to tensors
        state_batch = torch.cat(states,dim=0).type(torch.float32)
        action_batch = torch.cat(actions,dim=0).type(torch.float32)
        reward_batch = torch.cat(rewards,dim=0).type(torch.float32)
        next_state_batch = torch.cat(next_states,dim=0).type(torch.float32)
        done_batch = torch.cat(dones,dim=0).type(torch.float32)
        return state_batch,action_batch,reward_batch,next_state_batch,done_batch

    def random_sample(self, batch_size):
        if len(self.buffer) < batch_size:
            random_samples= list(self.buffer)
        else:
            random_samples = random.sample(self.buffer, batch_size)


        # Unzip the samples
        states, actions, rewards, next_states, dones = zip(*random_samples)

        # Convert to tensors
        state_batch = torch.cat(states, dim=0).type(torch.float32)
        action_batch = torch.cat(actions, dim=0).type(torch.float32)
        reward_batch = torch.cat(rewards, dim=0).type(torch.float32)
        next_state_batch = torch.cat(next_states, dim=0).type(torch.float32)
        done_batch = torch.cat(dones, dim=0).type(torch.float32)

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch


class Episodic_ReplayBuffer:
    def __init__(self, capacity,max_episode_steps):
        self.buffer = deque(maxlen=capacity)
        self.current_episode_buffer=deque(maxlen=max_episode_steps)
        self.max_episode_steps=max_episode_steps

    def push(self, state, action, reward, next_state, done):
        state = torch.from_numpy(state).unsqueeze(0).type(torch.float32)
        action = torch.from_numpy(action).unsqueeze(0).type(torch.float32)
        reward = torch.from_numpy(reward).unsqueeze(0).type(torch.float32)
        next_state = torch.from_numpy(next_state).unsqueeze(0).type(torch.float32)
        done = torch.from_numpy(np.array([done])).unsqueeze(0).type(torch.float32)
        experience = (state, action, reward, next_state, done)
        self.current_episode_buffer.append(experience)

    def renew(self):
        self.buffer.append(self.current_episode_buffer)
        self.current_episode_buffer=deque(maxlen=self.max_episode_steps)
    def sample(self, batch_size):
        episode_buffer=self.current_episode_buffer
        if len(episode_buffer) < batch_size:
            latest_samples= list(episode_buffer)
        else:
            latest_samples= list(episode_buffer)[-batch_size:]

        states, actions, rewards, next_states, dones = zip(*latest_samples)

        # Convert to tensors
        state_batch = torch.cat(states,dim=0).type(torch.float32)
        action_batch = torch.cat(actions,dim=0).type(torch.float32)
        reward_batch = torch.cat(rewards,dim=0).type(torch.float32)
        next_state_batch = torch.cat(next_states,dim=0).type(torch.float32)
        done_batch = torch.cat(dones,dim=0).type(torch.float32)
        return state_batch,action_batch,reward_batch,next_state_batch,done_batch

    def random_sample(self, batch_size):
        if len(self.buffer)>0:
           episode_buffer = random.sample(self.buffer,1)[0]
        else :
           episode_buffer=self.current_episode_buffer
        if len(episode_buffer) < batch_size:
            random_samples= list(episode_buffer)
        else:
            random_samples = random.sample(episode_buffer, batch_size)


        # Unzip the samples
        states, actions, rewards, next_states, dones = zip(*random_samples)

        # Convert to tensors
        state_batch = torch.cat(states, dim=0).type(torch.float32)
        action_batch = torch.cat(actions, dim=0).type(torch.float32)
        reward_batch = torch.cat(rewards, dim=0).type(torch.float32)
        next_state_batch = torch.cat(next_states, dim=0).type(torch.float32)
        done_batch = torch.cat(dones, dim=0).type(torch.float32)

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch



def learn_standard(
    save_dir,
    ReplayBuffer,
    target_agent,
    agent,
    actor_optimizer,
    critic_optimizer,
    log_alpha_optimizer,
    log_alpha,
    target_entropy,
    batch_size=8,
    gamma=0.99,
    critic_clip=True,
    actor_clip=True,
    update_policy=True,
):

    batch = ReplayBuffer.sample(batch_size)

    # prepare transitions for models
    state_batch, action_batch, reward_batch, next_state_batch, done_batch = batch
    state_batch = state_batch.to(device)
    next_state_batch = next_state_batch.to(device)
    action_batch = action_batch.to(device)
    reward_batch = reward_batch.to(device)
    done_batch = done_batch.to(device)

    agent.train()
    ###################
    ## CRITIC UPDATE ##
    ###################
    alpha = torch.exp(log_alpha)
    with torch.no_grad():
        action_dist_s1 = agent.actor(next_state_batch)
        action_s1 = action_dist_s1.rsample()
        logp_a1 = action_dist_s1.log_prob(action_s1).sum(-1, keepdim=True)
        target_action_value_s1 = torch.min(
            target_agent.critic1(next_state_batch, action_s1),
            target_agent.critic2(next_state_batch, action_s1),
        )
        td_target = reward_batch + gamma * (1.0 - done_batch) * (
            target_action_value_s1 - (alpha * logp_a1)
        )

    # update critics
    agent_critic1_pred = agent.critic1(state_batch, action_batch)
    agent_critic2_pred = agent.critic2(state_batch, action_batch)
    td_error1 = td_target - agent_critic1_pred
    td_error2 = td_target - agent_critic2_pred
    critic_loss = 0.5 * (td_error1 ** 2 + td_error2 ** 2)
    critic_loss = critic_loss.mean()
    critic_optimizer.zero_grad()
    critic_loss.backward()
    if critic_clip:
        torch.nn.utils.clip_grad_norm_(
            chain(agent.critic1.parameters(), agent.critic2.parameters()), critic_clip
        )
    critic_optimizer.step()

    if update_policy:
        ##################
        ## ACTOR UPDATE ##
        ##################
        dist = agent.actor(state_batch)
        agent_actions = dist.rsample()
        logp_a = dist.log_prob(agent_actions).sum(-1, keepdim=True)
        actor_loss = -(
            torch.min(
                agent.critic1(state_batch, agent_actions),
                agent.critic2(state_batch, agent_actions),
            )
            - (alpha.detach() * logp_a)
        ).mean()
        actor_optimizer.zero_grad()
        actor_loss.backward()
        if actor_clip:
            torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), actor_clip)
        actor_optimizer.step()

        ##################
        ## ALPHA UPDATE ##
        ##################
        alpha_loss = (-alpha * (logp_a + target_entropy).detach()).mean()
        log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        log_alpha_optimizer.step()

    alpha_op = torch.exp(log_alpha.clone().detach())
    torch.save(alpha_op, 'alpha.pt')
    torch.save(alpha_op, os.path.join(save_dir,'alpha.pt'))






def learn_standard_rd(
    save_dir,
    ReplayBuffer,
    target_agent,
    agent,
    actor_optimizer,
    critic_optimizer,
    log_alpha_optimizer,
    log_alpha,
    target_entropy,
    batch_size=8,
    gamma=0.99,
    critic_clip=True,
    actor_clip=True,
    update_policy=True,
):

    batch = ReplayBuffer.random_sample(batch_size)

    # prepare transitions for models
    state_batch, action_batch, reward_batch, next_state_batch, done_batch = batch
    state_batch = state_batch.to(device)
    next_state_batch = next_state_batch.to(device)
    action_batch = action_batch.to(device)
    reward_batch = reward_batch.to(device)
    done_batch = done_batch.to(device)

    agent.train()
    ###################
    ## CRITIC UPDATE ##
    ###################
    alpha = torch.exp(log_alpha)
    with torch.no_grad():
        action_dist_s1 = agent.actor(next_state_batch)
        action_s1 = action_dist_s1.rsample()
        logp_a1 = action_dist_s1.log_prob(action_s1).sum(-1, keepdim=True)
        target_action_value_s1 = torch.min(
            target_agent.critic1(next_state_batch, action_s1),
            target_agent.critic2(next_state_batch, action_s1),
        )
        td_target = reward_batch + gamma * (1.0 - done_batch) * (
            target_action_value_s1 - (alpha * logp_a1)
        )

    # update critics
    agent_critic1_pred = agent.critic1(state_batch, action_batch)
    agent_critic2_pred = agent.critic2(state_batch, action_batch)
    td_error1 = td_target - agent_critic1_pred
    td_error2 = td_target - agent_critic2_pred
    critic_loss = 0.5 * (td_error1 ** 2 + td_error2 ** 2)
    critic_loss = critic_loss.mean()
    critic_optimizer.zero_grad()
    critic_loss.backward()
    if critic_clip:
        torch.nn.utils.clip_grad_norm_(
            chain(agent.critic1.parameters(), agent.critic2.parameters()), critic_clip
        )
    critic_optimizer.step()

    if update_policy:
        ##################
        ## ACTOR UPDATE ##
        ##################
        dist = agent.actor(state_batch)
        agent_actions = dist.rsample()
        logp_a = dist.log_prob(agent_actions).sum(-1, keepdim=True)
        actor_loss = -(
            torch.min(
                agent.critic1(state_batch, agent_actions),
                agent.critic2(state_batch, agent_actions),
            )
            - (alpha.detach() * logp_a)
        ).mean()
        actor_optimizer.zero_grad()
        actor_loss.backward()
        if actor_clip:
            torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), actor_clip)
        actor_optimizer.step()

        ##################
        ## ALPHA UPDATE ##
        ##################
        alpha_loss = (-alpha * (logp_a + target_entropy).detach()).mean()
        log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        log_alpha_optimizer.step()

    alpha_op = torch.exp(log_alpha.clone().detach())
    torch.save(alpha_op, 'alpha.pt')
    torch.save(alpha_op, os.path.join(save_dir, 'alpha.pt'))


