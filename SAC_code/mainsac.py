import pandas as pd
import numpy as np
import datetime
import os
import time
import pickle
import tensorflow as tf
from shutil import copy
from sac import Agent, RandomNormal_decay
from PowerFlow_V3_sac import Environment
import matplotlib.pyplot as plt
import sklearn
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from matplotlib import pyplot as plt
from scipy.io import savemat

###
# Processing SETUP file to load the configurations
###

current_directory = os.getcwd()
for file in os.listdir(current_directory):
    if file.endswith('SETUP.txt'):
        input_file = file
        break
input_file = 'SETUP.txt'
bat_name = [] #Battery name
bat_bus = []  #Battery bus
soc_max = [] #Upper limit of battery SOC - expressed in %
soc_min = [] #Lowet limit of battery SOC - expressed in %
soc_t0 = [] #Initial battery SOC - expressed in %
with open(input_file) as f:
    for line in f:
        if line.startswith('#'):     #SKIP ALL LINES OF SETUP FILE STARTING WITH #
            continue
        else:
            line = line.strip()
            text, value = line.split(' = ')  #SPLIT THE REST OF THE LINES IN TEXT AND VALUE. They are separated by = SIGN
            if text == 'load_checkpoint_actor': #Determines whether a new agent is trained or whether a trained agent is evaluated
                if value == 'True': #choose "True", if: evaluating a trained agent or "False" if training a new agent
                    load_checkpoint_actor = True
                else:
                    load_checkpoint_actor = False
            if text == 'retrain':   #Determines whether retraining is performed on a trained agent
                if value == 'True':
                    retrain = True
                else:
                    retrain = False
            if text == 'testing_dir': # directory when testing the agent (i.e. load_checkpoint_actor = True)
                testing_dir = value
            if text == 'scenario_name':#SCENARIO NAME
                scenario_name = value
            if text == 'network':  #GRID TO BE USED
                network = value
            if text == 'num_timesteps': # NUMBER OF TIMESTEPS IN ONE SCENARIO (24 * 4 = 96)
                num_timesteps = int(value)
            if text == 'n_games':  # NUMBER OF SCENARIOS TO BE CONSIDERED FOR TESTING / EVALUATION
                n_games = int(value)
            if text == 'vmax': #UPPER LIMIT VOLTAGE IN THE BUS
                vmax = float(value)
            if text == 'vmin': # LOWER LIMIT VOLTAGE IN THE BUS
                vmin = float(value)
            if text == 'w1':   #Weight for reactive power exchange with main grid
                w1 = float(value)
            if text == 'w2':   #Weight for absolute value of active power exchange
                w2 = float(value)
            if text == 'w3':    #Weight for active power exchange with main grid
                w3 = float(value)
            if text == 'w4':    #Weight for mismatch from DP
                w4 = float(value)
            if text == 'w5':    #Weight for voltage constraints
                w5 = float(value)
            if text == 'w6':    #Weight for terminal battery
                w6 = float(value)
            if text == 'w7':    #Weight for excessive battery charging
                w7 = float(value)
            if text == 'w8':    #Weight for mismatch from predefined battery
                w8 = float(value)
            if text == 'w9':    #Weight for charging battery when PV production is high
                w9 = float(value)
            if text == 'actor_lr': #Actor learning rate
                actor_lr = float(value)
            if text == 'critic_lr':#Critic learning rate
                critic_lr = float(value)
            if text == 'gamma': #Discount factor in RL setup
                gamma = float(value)
            if text == 'layer_size': #Size of the hidden layers of the Neural network
                layer_size = int(value)
            if text == 'batch_size': #Size of the batch sampled from the replay buffer and used for updating the network parameters
                batch_size = int(value)
            if text == 'batch_split': #Determines the ratio between non-terminal and terminal states in batch (e.g. 1 = 100% non-terminal states in batch)
                batch_split = float(value)
            if text == 'max_size': #Determines the maximum size of the replay buffer to store experienced trajectories
                max_size = int(value)
            if text == 'tau': #Determines the target soft update parameter
                tau = float(value)
            if text == 'learning_step': #Determines the number of timesteps between two learning processes
                learning_step = int(value)
            if text == 'dt': #Determines the duration of one timestep
                dt = float(value)
            if text == 'n_batteries': #Determines the number of batteries connected (Evaluations were only performed for one battery)
                n_batteries = float(value)
f.close

###
# Definition of the required paths and creation of the corresponding directories
###
eleni4bus_disp_path = os.getcwd() + '/Data/Eleni4BusNetwork/Dispatch/'
eleni34bus_disp = os.getcwd() + '/Data/Eleni34BusNetwork/'
if network == 'eleni4Bus':
    scenario_path = eleni4bus_disp_path + scenario_name
elif network == 'eleni34Bus':
    scenario_path = eleni34bus_disp + scenario_name
else:
    raise ValueError('Input valid network, optimization or scenario name')

result_parent_dir = os.path.abspath(os.getcwd()) + '/Trained_models/'
results_path_date = datetime.datetime.now().strftime("%Y_%m_%d")
results_folder = os.path.join(result_parent_dir, results_path_date)

if not os.path.isdir(results_folder):
    os.makedirs(results_folder, mode=755)
if not load_checkpoint_actor and not retrain: # create new folder for agent in training
    results_path = os.path.join(results_folder, datetime.datetime.now().strftime("%H-%M-%S"))
    os.mkdir(results_path)
    copy(os.path.abspath(input_file),results_path)
    # logger for saved models
    def printLog(log):
        with open(results_path + '/log.txt','a') as file:
            print(log, file = file)
else: # load trained agent from existing folder
    results_path = result_parent_dir + testing_dir

np.random.seed(0)

###
# Set battery properties depending on network and number of batteries
###
if network == "eleni4Bus":
    max_app_power = 0.24
    n_actions = 4
    base = 5
    bat_name = ["bat1"]
    bat_bus = [4]
    soc_max = [100]
    soc_min = [0]
    soc_t0 = [33.3333333]
    max_e_mwh =[0.166666666/base] #Max battery capacity
    min_e_mwh = [0] #Min battery capacity

elif network == "eleni34Bus":
    max_app_power = 2 #MAXIMUM APPARENT POWER
    base = 8.333333  #SINGLE-PHASE-BASE POWER - TO COMPUTE VALUES IN PER UNIT SYSTEM
    max_e_mwh =[1/base]
    min_e_mwh = [0] #Min battery capacity

    if n_batteries == 1:  #THIS IS OUR CASE - 1 BATTERY
        n_actions = 4   #NUMBER OF ACTIONS FOR RL SETUP (4 AS OUTLINED IN REPORT)
        bat_name = ["bat1"]
        bat_bus = [34]  #Battery bus
        soc_max = [100]
        soc_min = [0]
        soc_t0 = [33.3333333]

    elif n_batteries == 2:
        n_actions = 6
        bat_name = ["bat1", "bat2"]
        bat_bus = [34,35]
        soc_max = [100,100]
        soc_min = [0,0]
        soc_t0 = [33.3333333,33.3333333]

eta_ch = [1.0] #charging efficiency
eta_dis = [1.0] #discharging efficiency

soc_T_max = soc_max #Upper capacity limit to be considered in the terminal constraint
soc_T_min = soc_min #Lower capacity limit to be considered in the terminal constraint

#Maximum and minimum threshold for each action - What are 0.95 and 0.05 (percentages I suppose???)
max_action = [max_app_power/base*0.95,max_app_power/base*0.05, 1,1] if n_actions == 4 else [max_app_power/base*0.95,max_app_power/base*0.05,max_app_power/base*0.95,max_app_power/base*0.05,1,1]     # MW, 4 Bus grid: 0.24 MW, 34 Bus grid: 2MW
min_action = [-i for i in max_action]

if retrain:
    batch_size_save = batch_size
    batch_size = 5
elif load_checkpoint_actor:
    batch_size = 5

###
# Initialize noise process
###
gaussian_decay = RandomNormal_decay(shape = [n_actions], mean = 0.0,
                                    stddev_start = 0.01, stddev_end = 0.0001,
                                    decay = 0.000005, epsilon_start = 1,
                                    epsilon_end = 0.001, epsilon_decay = 0.001)#stddev_end = 0.01

###
# Initialize grid object
###
env = Environment(network, scenario_path, num_timesteps = num_timesteps,
                  vmax = vmax, vmin = vmin,
                  bat_bus = bat_bus, max_e_mwh = max_e_mwh,
                  min_e_mwh = min_e_mwh, bat_name = bat_name, eta_ch = eta_ch,
                  eta_dis = eta_dis, soc_t0 = soc_t0, soc_max = soc_max,
                  soc_T_max = soc_T_max, soc_T_min =soc_T_min,
                  soc_min = soc_min,
                  w1 = w1, w3 = w3, w4 = w4, w5 = w5, w6 = w6, w7 = w7, w8 = w8, w9 = w9, dt = dt,
                  load_checkpoint_actor = load_checkpoint_actor)

env.reset(load_scenarios = True) #reset grid and get initial state
_, input_dims = env.get_state()

###
# Initialize agent object
###
agent = Agent(noise = gaussian_decay, alpha = actor_lr, beta = critic_lr, input_dims=input_dims,
              max_action = max_action, gamma = gamma, n_actions=n_actions, max_size=max_size,
              tau=tau, hidden_layer_dim1 = layer_size, hidden_layer_dim2 = layer_size,
              batch_size=batch_size, reward_scale = 1, chkpt_dir = results_path)


###
# Declaration of the required variables
###
if load_checkpoint_actor:
    SOC_history = np.zeros((2,n_games,num_timesteps), dtype= np.float64)
else:
    SOC_history = np.zeros((2,99,num_timesteps), dtype= np.float64)

score_history = []
reward_history = []
voltage_reward = []
pcc_reward = []
p_act = []
q_act = []
soc_terminal_reward = []
get_action_timer = []   # time to provide action --> will be averaged over all scenarios
env_step_timer = [] # time to update environment and run power flow --> will be averaged over all scenarios
agent_learn_timer = [] # time to update the neural network weights --> will be averaged over all scenarios
store_timer = [] # time to store the current agent setup --> will be averaged over all scenarios
axis = []
reward_history = np.zeros((n_games*num_timesteps), dtype= np.float64)
best_score = float('-inf')

for i in range(num_timesteps):
    axis.append((i+1)/4)

if load_checkpoint_actor:
    pcc_disp = np.zeros((n_games,num_timesteps),dtype=np.float64)
    pcc_dispq = np.zeros((n_games,num_timesteps),dtype=np.float64)
    pcc_actual = np.zeros((n_games,num_timesteps),dtype=np.float64)
    pcc_actualq = np.zeros((n_games,num_timesteps),dtype=np.float64)
else:
    pcc_disp = np.zeros((1000,num_timesteps),dtype=np.float64)
    pcc_actual = np.zeros((1000,num_timesteps),dtype=np.float64)

###
# Load trained agent weights for evaluating if load_checkpoint_actor = True
###
if load_checkpoint_actor:
    n_steps = 0
    while n_steps <= agent.batch_size:
        observation = env.reset(load_scenarios= False)
        action = list(np.zeros(n_actions))
        next_observation, reward, _, done, _ = env.step(action)
        agent.remember(observation, action, reward, next_observation, done)
        n_steps += 1
    agent.learn()
    agent.load_models()
    evaluate = True
else:
    evaluate = False

###
# Load trained agent weights for retraining if retrain = True
###
if retrain:
    n_steps = 0
    while n_steps <= agent.batch_size:
        observation = env.reset(load_scenarios= False)
        action = list(np.zeros(n_actions))
        next_observation, reward, _, done, _ = env.step(action)
        agent.remember(observation, action, reward, next_observation, done)
        n_steps += 1
    agent.learn()
    agent.load_models()
    evaluate = False
    agent.batch_size = batch_size_save

###
# Iterate through scenarios
###
for i in range(n_games):
    observation = env.reset(load_scenarios=False)
    done = False
    score = 0

    get_action_timer2 = []
    env_step_timer2 = []
    agent_learn_timer2 = []
    store_timer2 = []

    ###
    # Iterate through time steps
    ###
    while not done:

        ### Select action
        tic1 = time.perf_counter()
        action = agent.choose_action(observation, evaluate) #NO ADDITIONAL NOISE ADDED FOR NOW
        toc1 = time.perf_counter()

        ### State transition and calculation of reward
        tic2 = time.perf_counter()
        next_observation, reward, action, done, violation = env.step(action,evaluate)
        score += reward
        toc2 = time.perf_counter()

        ### Store Transition
        agent.remember(observation,action,reward,next_observation,done) # buffer

        ### Update neural network weights
        tic3 = time.perf_counter()
        if not load_checkpoint_actor and env.timestep % learning_step == 0:
            agent.learn()
        toc3 = time.perf_counter()

        ### Declare next state as current state
        observation= next_observation
        score += reward
        avg_score = np.mean(score_history[-100:]) #After every 100 scenario, weights are saved

        ### Store weights if performance increased
        tic4 = time.perf_counter()
        if avg_score > best_score:
            best_score=avg_score
            if not load_checkpoint_actor:
                agent.save_models()
                print('... saving done ...')
        toc4 = time.perf_counter()

        ### Store PCC actions and SOC trajectory of current scenario
        if load_checkpoint_actor:
            pcc_actual[i][env.timestep-1] = env.net.res_ext_grid.p_mw *base*1000*3
            pcc_actualq[i][env.timestep-1] = env.net.res_ext_grid.q_mvar *base*1000*3
            pcc_disp[i][env.timestep-1] = action[-2]*base*1000*3
            pcc_dispq[i][env.timestep-1] = action[-1]*base*1000*3
        else:
            pcc_actual[i%1000][env.timestep-1] = env.net.res_ext_grid.p_mw *base*3000
            pcc_disp[i%1000][env.timestep-1] = action[-2]*base*3000

        if load_checkpoint_actor:
            for element in env.net.storage.index:
                SOC_history[element][i][env.timestep-1] = env.net.storage.loc[element,'soc_percent']
        elif i > n_games-100:
            for element in env.net.storage.index:
                SOC_history[element][i-n_games+99][env.timestep-1] = env.net.storage.loc[element,'soc_percent']

        reward_history[i*96 + env.timestep-1] = reward;

        get_action_timer2.append(toc1-tic1)
        env_step_timer2.append(toc2-tic2)
        agent_learn_timer2.append(toc3-tic3)
        store_timer2.append(toc4-tic4)

    print('episode ', i , 'score %.1f' %score, 'avg score %.1f' % avg_score,'best score %.1f' % best_score)
    score_history.append(score)

    if not load_checkpoint_actor:

        template2 = 'Action time: {}, Environment step time: {}, Learning step time: {}, Storing time: {}'

        print(template2.format(sum(get_action_timer2)/len(get_action_timer2),
                           sum(env_step_timer2)/len(env_step_timer2),
                           sum(agent_learn_timer2)/len(agent_learn_timer2),
                           sum(store_timer2)/len(store_timer2)))


    ### Create interim PCC and score figures
    if i%1000 == 0 and not load_checkpoint_actor:

        pcc_disp = pcc_disp.mean(axis=0)
        pcc_actual = pcc_actual.mean(axis=0)

        f = plt.figure()
        f.set_size_inches(50, 20)
        plt.title(str(i),fontsize = 50)
        plt.plot(axis,pcc_disp,label='DP')
        plt.plot(axis,pcc_actual, label = 'Actual')
        plt.legend(loc="upper left")
        plt.ylabel('Power at PCC [kW]', fontsize = 50)
        plt.xlabel('Time [h]', fontsize = 50)
        plt.legend(loc=2, prop={'size': 60})
        plt.xticks(np.arange(0, 24.25, 4.0), fontsize = 30)
        plt.yticks(fontsize = 30)
        plt.grid(visible = True)
        f.savefig(f'{results_path}/PCC_part.png')
        plt.close()

        plt.plot(score_history)
        plt.title(str(i))
        plt.savefig(f'{results_path}/Score_part.png')

        pcc_disp = np.zeros((1000,num_timesteps),dtype=np.float64)
        pcc_actual = np.zeros((1000,num_timesteps),dtype=np.float64)

### Average PCC actions for DP and SOC trajectory
disp_p = pcc_disp
pcc_disp = pcc_disp.mean(axis=0)
pcc_actual = pcc_actual.mean(axis=0)

if load_checkpoint_actor:
    disp_q = pcc_dispq
    pcc_dispq = pcc_dispq.mean(axis=0)
    pcc_actualq = pcc_actualq.mean(axis=0)

SOC_history[0] = SOC_history[0].mean(axis=0)
SOC_history[1] = SOC_history[1].mean(axis=0)
#score_history = pd.DataFrame(score_history)

###
#Create final figures
###

if not load_checkpoint_actor:
    f = plt.figure()
    f.set_size_inches(43, 23.5)
    plt.plot(score_history,linewidth=4)
    plt.ylabel('Score', fontsize = 70)
    plt.xlabel('Scenario', fontsize = 70)
    plt.xticks(fontsize = 50)
    plt.yticks(fontsize = 50)
    plt.grid(visible = True)
    f.savefig(f'{results_path}/Score.png')
    plt.close()

f = plt.figure()
f.set_size_inches(43, 23.5)
if network == 'eleni34Bus' and n_batteries ==2:
    plt.plot(axis,SOC_history[1][0], label = 'Battery 2',linewidth=4)
plt.plot(axis,SOC_history[0][0],label='Battery 1',linewidth=4)
ax = plt.gca()
ax.set_ylim([0,101])
plt.legend(loc="upper left",prop={'size': 60})
plt.ylabel('SOC [%]', fontsize = 70)
plt.xlabel('Time [h]', fontsize = 70)
plt.xticks(np.arange(0, 24.25, 4.0),fontsize = 50)
plt.yticks(fontsize = 50)
plt.grid(visible = True)
print("PATH",results_path)
f.savefig(f'{results_path}/SOC.png')
plt.close()

f = plt.figure()
f.set_size_inches(43, 23.5)
if not load_checkpoint_actor:
    plt.plot(axis,pcc_actual, label = 'Actual',linewidth=4)
plt.plot(axis,pcc_disp,label='DP',linewidth=4)
plt.legend(loc="upper left")
plt.ylabel('Power at PCC [kW]', fontsize = 70)
plt.xlabel('Time [h]', fontsize = 70)
plt.legend(loc=2, prop={'size': 60})
plt.xticks(np.arange(0, 24.25, 4.0), fontsize = 50)
plt.yticks(fontsize = 50)
plt.grid(visible = True)
f.savefig(f'{results_path}/PCC.png')
plt.close()

###
# Store averaged DP
###
if load_checkpoint_actor:
    savemat(f'{results_path}/dP_avg.mat',{'AppliedDP_P':pcc_disp/base/1000/3})
    savemat(f'{results_path}/dP.mat',{'AppliedDP_P':disp_p/base/1000/3})
    savemat(f'{results_path}/dP_Q_avg.mat',{'AppliedDP_Q':pcc_dispq/base/1000/3})
    savemat(f'{results_path}/dP_Q.mat',{'AppliedDP_Q':disp_q/base/1000/3})
