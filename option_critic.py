## Simple implementation of the [Option-Critic Architecture](https://arxiv.org/abs/1609.05140) on the Four-rooms environment.

from fourrooms import FourRooms
from utils import *

from time import sleep

import numpy as np

from IPython.display import clear_output
import matplotlib.pyplot as plt

def train_four_rooms(args, log, tb_writer, environ, session_number):

    # Evaluation reward while training 

    # Discount
    discount = 0.99

    # Learning rates - termination, intra-option, critic
    lr_term = 0.25
    lr_intra = 0.25
    lr_critic = 0.5

    # Epsilon for epsilon-greedy for policy over options
    epsilon = 1e-1

    # Temperature for softmax
    temperature = 1e-2

    # Number of runs
    nruns = 1

    # Number of episodes per run
    nepisodes = 1000

    # Maximum number of steps per episode
    nsteps = 1000

    # Random number generator for reproducability
    rng = np.random.RandomState(1234)

    # The possible next goals (all in the lower right room)
    possible_next_goals = [103] #[68, 69, 70, 71, 72, 78, 79, 80, 81, 82, 88, 89, 90, 91, 92, 93, 99, 100, 101, 102, 103]

    # History of steps and average durations
    history = np.zeros((nruns, nepisodes, 2))

    option_terminations_list = []
    noptions = 4
    for run in range(nruns):
        
        env = FourRooms(environ)
        # env.reset()
        # plt.imshow(env.render(show_goal=True), cmap='Blues')
        # plt.axis('off')
        # plt.show()

        nstates = env.observation_space.shape[0]
        nactions = env.action_space.shape[0]
        
        # Following three belong to the Actor
        
        # 1. The intra-option policies - linear softmax functions
        option_policies = [SoftmaxPolicy(rng, lr_intra, nstates, nactions, temperature) for _ in range(noptions)]
        
        # 2. The termination function - linear sigmoid function
        option_terminations = [SigmoidTermination(rng, lr_term, nstates) for _ in range(noptions)]
        
        # 3. The epsilon-greedy policy over options
        policy_over_options = EpsGreedyPolicy(rng, nstates, noptions, epsilon)
        
        # Critic
        critic = Critic(lr_critic, discount, policy_over_options.Q_Omega_table, nstates, noptions, nactions)
        
        print('Goal: ', env.goal)
        total_reward = 0 
        eval_reward = 0
        sum_reward = 0
        print("--------------------------------------------------------------------------------------")
        print("--------------------------------------------------------------------------------------")

        for episode in range(nepisodes):
            total_episodes = session_number*nepisodes+episode
            # Change goal location after 1000 episodes 
            # Comment it for not doing transfer experiments
            # if episode == 1000:
            #     env.goal = rng.choice(possible_next_goals)
            #     print('New goal: ', env.goal)
            
            state = env.reset()
            
            option = policy_over_options.sample(state)
            action = option_policies[option].sample(state)
            
            critic.cache(state, option, action)
            
            duration = 1
            option_switches = 0
            avg_duration = 0.0
            for step in range(nsteps):
                state, reward, done, _ = env.step(action)
                total_steps = episode*nsteps+step

                if(step%10==0 and step!=0):
                    evaluation_reward = eval_reward/10,
                    cumulative_reward = total_reward/total_steps
                    average_reward = eval_reward/total_steps
                    #log[args.log_name].info("Reward {}".format(reward))

                    #log[args.log_name].info("Evaluation Reward {} at step {}, run {}".format(eval_reward/10, total_steps, run))
                    #log[args.log_name].info("Average Reward {} at step {}, run {}".format(eval_reward/total_steps, total_steps, run))

                    eval_reward = 0
                
                # Termination might occur upon entering new state
                if option_terminations[option].sample(state):
                    option = policy_over_options.sample(state)
                    option_switches += 1
                    avg_duration += (1.0/option_switches)*(duration - avg_duration)
                    duration = 1
                    
                action = option_policies[option].sample(state)
                
                # Critic update
                critic.update_Qs(state, option, action, reward, done, option_terminations, total_steps, episode, session_number*nepisodes, log, args, tb_writer)
                
                # Intra-option policy update with baseline
                Q_U = critic.Q_U(state, option, action)
                Q_U = Q_U - critic.Q_Omega(state, option)
                option_policies[option].update(state, action, Q_U)
                
                # Termination condition update
                option_terminations[option].update(state, critic.A_Omega(state, option))
                
                duration += 1
                
                total_reward +=reward
                eval_reward +=reward
                sum_reward +=(discount**total_steps)*reward
                print(sum_reward)
                tb_writer.add_scalars("reward", {"reward": sum_reward}, total_episodes)

    
                if done:
                    break
                    
            history[run, episode, 0] = step
            history[run, episode, 1] = avg_duration

            #log[args.log_name].info("Average Duration of{} for run {}".format(avg_duration,run))
            tb_writer.add_scalars("Average Duration", {"avg_duration": avg_duration}, run)

            #log[args.log_name].info("Option Switches are {} for run {}".format(option_switches,run))
            tb_writer.add_scalars("Option Switches for Runs", {"option_switches": option_switches}, run)

           
        option_terminations_list.append(option_terminations)


