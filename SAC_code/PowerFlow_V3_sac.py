import os
import pandapower as pp
import pandapower.networks as nw
import numpy as np
import pickle
import pandapower.topology as top
import statistics
import time
import random

class Environment():
    def __init__(self, network = None, data_path = None,
                 vmax = 1.05, vmin = 0.95, vnom = 1.0,
                 loading_percent = 100, num_timesteps = 96, bat_bus = [5],
                 soc_t0 = [50], soc_T_max = [100], soc_T_min = [50],
                 p_max = [1], q_max = [1], max_e_mwh = [8],
                 min_e_mwh = [0], bat_name = [None], eta_ch = [0.95],
                 eta_dis = [0.95], soc_max = [100], soc_min = [0],
                 w1 = 1, w2 = 1, w3 = 1, w4 = 1, w5 = 1, w6 = 1, w7 = 1, w8 = 1, w9 = 1, dt = 0.25,
                 load_checkpoint_actor = False):

        self.network = network
        self.data_path = data_path
        self.vmax = vmax
        self.vmin = vmin
        self.vnom = vnom
        self.loading_percent = loading_percent
        self.bat_bus = bat_bus
        self.soc_t0 = soc_t0
        self.soc_T_max = soc_T_max
        self.soc_T_min = soc_T_min
        self.p_max = p_max
        self.q_max = q_max
        self.max_e_mwh = max_e_mwh
        self.min_e_mwh = min_e_mwh
        self.bat_name = bat_name
        self.eta_ch = eta_ch
        self.eta_dis = eta_dis
        self.soc_max = soc_max
        self.soc_min = soc_min
        self.done = False
        self.timestep = 0
        self.num_timesteps = num_timesteps
        self.episode_count = 0
        self.dt = dt
        self.load_checkpoint_actor = load_checkpoint_actor # wether system is training or testing

        #reward weights
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w4 = w4
        self.w5 = w5
        self.w6 = w6
        self.w7 = w7
        self.w8 = w8
        self.w9 = w9


    ###
    # Reset the grid
    ###
    def reset(self, load_scenarios, evaluate = False):
        self.timestep = 0
        self.done = False
        self.net = None

        ### Create the net with loading the forecasts file
        self.create_net(evaluate = evaluate, load_scenarios=load_scenarios)

        for load in self.net.load.index:
            self.net.load.loc[load,'scaling']=1

        print ('Running episode ', self.episode_count)

        ### Run powerflow to get the initial state (no control action)
        pp.runpp(self.net, algorithm = 'bfsw', init_vm_pu = 'flat')

        ### Get initial state
        state, _ = self.get_state()
        return state

    ###
    # Creation of the simulation of the grid
    ###
    def create_net(self,evaluate, load_scenarios):
        if self.network == 'eleni34Bus':
            eleni_path = os.getcwd() + '/Data/Eleni34BusNetwork/eleni34Bus.json'
            self.net = pp.from_json(eleni_path)

        elif self.network == 'eleni4Bus':
            eleni_path = os.getcwd() + '/Data/Eleni4BusNetwork/eleni4Bus.json'
            self.net = pp.from_json(eleni_path)

        else:
            raise ValueError('chosen network unavailable')

        ### Create batteries
        self.create_battery(evaluate)

        ### Load forecasts file
        if load_scenarios:
            with open(self.data_path, 'rb') as f:
                self.scenarios = pickle.load(f)
            self.num_episodes = len(self.scenarios.columns)
            print("NUMBER OF SCENARIOS", self.num_episodes)


        ### Apply first forecasts to the grid
        self.get_net_data()

    ###
    # Create the batteries in the grid
    ###
    def create_battery(self,evaluate):
        for i, _ in enumerate(self.bat_bus):
            pp.create_storage(self.net, bus = self.bat_bus[i], p_mw = 0,
                              max_e_mwh = self.max_e_mwh[0], q_mvar = 0,
                              soc_percent = self.soc_t0[i],
                              min_e_mwh = self.min_e_mwh[0],
                              name = self.bat_name[i])

            # add efficiency and nominal soc to storage dataframe
            self.net.storage.loc[i,'eta_ch']= self.eta_ch[0]
            self.net.storage.loc[i,'eta_dis'] = self.eta_dis[0]
            self.net.storage.loc[i,'soc_nom'] = self.soc_t0[i]
            self.net.storage.loc[i,'soc_max'] = self.soc_max[i]
            self.net.storage.loc[i,'soc_min'] = self.soc_min[i]
            # self.net.storage.loc[i, 'p_max'] = self.p_max[i]
            # self.net.storage.loc[i, 'q_max'] = self.q_max[i]

    ###
    # Apply loads and renewables forecasts to the grid for current timestep and scenario
    ###
    def get_net_data(self):
        timestep = self.timestep

        ### start from first episode if number of episodes in main is > than actual num_episodes
        if self.episode_count > self.num_episodes-1:
            self.episode_count = 0

        episode = self.episode_count

        if timestep < self.num_timesteps:

            if self.network == 'eleni34Bus':
                for load in self.net.load.index:
                    if self.net.load.name[load] != 'Battery Q' and self.net.load.name[load] != 'Battery Q2':
                        load_value = self.scenarios[episode].loc[timestep][self.net.load.name[load]]
                        self.net.load.loc[load, 'p_mw'] = load_value.real
                        self.net.load.loc[load, 'q_mvar'] = load_value.imag


            elif self.network == 'eleni4Bus':
                for load in self.net.load.index:
                    if self.net.load.name[load] == 'Prosumer1':
                        self.net.load.loc[load, 'p_mw'] = self.scenarios[episode].loc[timestep][self.net.load.name[load]].real
                        self.net.load.loc[load, 'q_mvar'] = self.scenarios[episode].loc[timestep][self.net.load.name[load]].imag
                    elif self.net.load.name[load] == 'Prosumer2':
                        self.net.load.loc[load, 'p_mw'] = self.scenarios[episode].loc[timestep][self.net.load.name[load]].real
                        self.net.load.loc[load, 'q_mvar'] = self.scenarios[episode].loc[timestep][self.net.load.name[load]].imag
                    else:
                        pass # do nothing for the Q load of the battery at bus 2

            else:
                raise ValueError('no data available, please choose a valid network')


    ###
    # Correct battery actions to make sure that they are not overcharged or discharged too deeply.
    # Update the SOC of the batteries.
    # Apply the charging/discharging rates on the grid
    ###
    def update_battery(self, action):
        action_set = np.zeros(len(action))

        for i in self.net.storage.index:
            soc = self.net.storage.soc_percent[i]/100.
            max_e_mwh = self.net.storage.max_e_mwh[i]
            min_e_mwh = self.net.storage.min_e_mwh[i]
            e_mwh = soc*max_e_mwh
            eta_ch = self.net.storage.eta_ch[i]
            eta_dis = self.net.storage.eta_dis[i]

            ### Make sure the batteries are not overcharged or discharged to deeply
            if action[2*i] >= 0: ### charging
                ### clip action, if it would lead to SOC constraint violation
                action_set[2*i] = min(action[2*i], (max_e_mwh - e_mwh)/(eta_ch*self.dt))
            else: ### discharging
                action_set[2*i] = -min(abs(action[2*i]), eta_dis*(e_mwh - min_e_mwh)/self.dt) # clip action
            action_set[2*i+1] = action[2*i+1]

            ### Battery p_mw applied on the virtual DC bus
            self.net.storage.loc[i, 'p_mw'] = action_set[2*i]
            ### Battery q_mvar is always zero, because Q is applied to the real bus
            self.net.storage.loc[i, 'q_mvar'] = 0 # action[2*i+1] if len(action_set) > 1 else 0 # only for the cigre grid

            p_mw = self.net.storage.p_mw[i]

            if p_mw >= 0: ###charging
                e_mwh_new = e_mwh + p_mw*eta_ch*self.dt
                soc_new = e_mwh_new/max_e_mwh*100

            else: ###discharging
                e_mwh_new = e_mwh + p_mw/eta_dis*self.dt
                soc_new = e_mwh_new/max_e_mwh*100

            self.net.storage.loc[i, 'soc_percent'] = soc_new

        ### Battery reactive power is applied directly as load on the real bus
        for load in self.net.load.index:
            if self.net.load.name[load] == 'Battery Q' :
                self.net.load.loc[load, 'q_mvar'] = action_set[1] #if len(action_set) > 1 else 0
                self.net.load.loc[load, 'p_mw'] = 0
            elif self.net.load.name[load] == 'Battery Q2':
                self.net.load.loc[load, 'q_mvar'] = action_set[3] #if len(action_set) > 1 else 0
                self.net.load.loc[load, 'p_mw'] = 0
            else:
                pass

        ### Pass PCC actions
        action_set[-2] = action[-2]
        action_set[-1] = action[-1]

        ### action updated according to backup controller
        return action_set

    ###
    # Return the current state of the grid (time, current batteries SOC, current bus voltages,
    # forecasts for the next time step, current power at PCC)
    ###
    def get_state(self):
        state = []
        base = self.net.sn_mva

        ### Bus voltages
        voltage = []
        if self.network == 'eleni4Bus':
            for bus in self.net.bus.index:
                if self.net.bus.name[bus] != 'Battery Bus': # exclude virutal battery bus voltage from state space
                    if self.net.bus.name[bus] != 'Slack': # exclude slack bus voltage (because constant)
                        voltage.append(self.net.res_bus.vm_pu[bus])

        elif self.network == 'eleni34Bus':
            for bus in self.net.bus.index:
                if bus != 34 and bus != 35: # exclude virtual battery buses
                    voltage.append(self.net.res_bus.vm_pu[bus])

        ### Battery SOC (absolute)
        soc = []
        for battery in self.net.storage.index:
            soc.append(self.net.storage.soc_percent[battery]/100.)

        ### Power at PCC
        pcc_power = []
        pcc_re_power = []
        pcc_tot_power = []
        for ext_grid in self.net.ext_grid.index:
            pcc_power.append(self.net.res_ext_grid.p_mw[ext_grid])
            pcc_re_power.append(self.net.res_ext_grid.q_mvar[ext_grid])
            pcc_tot_power =np.concatenate((pcc_power,pcc_re_power))

        ### Time
        t = [self.timestep/self.num_timesteps] # normalized

        ### Load and renewables forecasts
        active_demand = []
        reactive_demand = []

        if self.network == 'eleni4Bus':
            for load in self.net.load.index:
                if self.net.load.name[load] != 'Battery Q': #exclude Q load from battery in state space
                    active_demand.append(self.net.res_load.p_mw[load])
                    reactive_demand.append(self.net.res_load.q_mvar[load])

        elif self.network == 'eleni34Bus':
            for load in self.net.load.index:
                if self.net.load.name[load] != 'Battery Q' and self.net.load.name[load] != 'Battery Q2': #exclude Q load from battery in state space
                    active_demand.append(self.net.load.p_mw[load])
                    reactive_demand.append(self.net.load.q_mvar[load])

        ### Observation state
        state = np.concatenate((t, soc, voltage, active_demand, pcc_tot_power))
        input_dims = [len(state)]

        return state, input_dims

    ###
    # Check whether a terminal state is reached or not. If so, set done flag to True
    ###
    def check_done(self):
        ### if terminal state, set done flag and give reward
        if self.timestep == self.num_timesteps:
            self.done = True
            self.episode_count += 1 # increase episode counter

        return self.done

    ###
    # Compute the reward for the selected actions and the resulting new state
    ###
    def get_reward(self, state, actions):

        violation = 0
        voltage_reward = 0
        soc_terminal_reward = 0
        pcc_reward = 0
        monetary_reward = 0
        reward = 0

        base = self.net.sn_mva

        ### check squared voltage violation
        voltage = self.net.res_bus.vm_pu**2
        v_high = self.vmax**2
        v_low = self.vmin**2
        if self.network == 'eleni4Bus' or self.network == 'eleni34Bus':
            voltage_reward = sum(max(voltage[bus]-v_high, v_low - voltage[bus], 0)
                                  for bus in self.net.bus.index if bus not in self.bat_bus) # exclude virtual battery bus
            for bus in self.net.bus.index:
                if bus not in self.bat_bus:
                    if voltage[bus]-v_high > 0:
                        violation += 1
                    elif voltage[bus] - v_low < 0:
                        violation +=1
        else: voltage_reward = sum(max(voltage[bus]-v_high, v_low - voltage[bus], 0)
                              for bus in self.net.bus.index)

        ### check terminal SOC deviation
        soc = self.net.storage.soc_percent
        if self.timestep == self.num_timesteps:
            for i in self.net.storage.index:
                soc_terminal_reward += abs((self.soc_T_max[i]/100 +
                                                self.soc_T_min[i]/100)/2 - soc[i]/100)


        ### Penalize excessive battery charging
        p_mw_bat = self.net.res_storage.p_mw
        bat_cycling_reward = sum(abs(p_mw_bat[battery]) for battery in self.net.storage.index)


        ### check dispatch plan deviation at PCC
        p_disp = actions[-2]
        q_disp = actions[-1]

        p_act = self.net.res_ext_grid.p_mw
        q_act = self.net.res_ext_grid.q_mvar

        pcc_delta = complex(p_act,q_act) - complex(p_disp,q_disp)
        pcc_reward = abs(pcc_delta)**2

        ### Check battery trajectory
        SOC = [0.33, 0.16, 0.17, 0.37, 0.82, 0.81, 0.63, 0.1]
        hours = [0, 16, 32, 48, 60, 76, 80, 96]

        soc_trajectory_reward = 0
        if self.network == 'eleni34Bus' and self.timestep in hours:
            for i in self.net.storage.index:
                soc_trajectory_reward += abs(self.net.storage.soc_percent[i]/100 - SOC[hours.index(self.timestep)])

        ### Make sure that batteries are charged when high PV production and vice versa
        load_sum = 0
        for load in self.net.load.index:
            load_sum += self.net.load.loc[load, 'p_mw']
        for batt in self.net.storage.index:
            load_sum += self.net.storage.loc[batt, 'p_mw']

        ### Calculate reward
        reward = -abs(q_act[0]) * self.w1- abs(p_act[0]) * self.w2 - p_act[0] * self.w3 - pcc_reward * self.w4 - voltage_reward * self.w5 - soc_terminal_reward * self.w6 -bat_cycling_reward *self.w7 - soc_trajectory_reward * self.w8 - load_sum * self.w9

        self.voltage_reward = voltage_reward
        self.pcc_reward = pcc_reward
        self.p_act = p_act[0]
        self.q_act = q_act[0]
        self.soc_terminal_reward = soc_terminal_reward

        return reward/self.num_timesteps, violation # NOTE: all partial rewards are normalized by the num_timesteps

    ###
    # Compute the next state based on the selected actions
    ###
    def step(self, action, evaluate = False):
        # receive action from agent (based on previous state), update battery
        # state and receive new load data. Run powerflow calculation an receive
        # new grid and battery state and compute reward

        ### Update SOC based on action chosen and return action with backup controller correction
        actions = self.update_battery(action)

        ### Apply actions on grid
        for ext_grid in self.net.ext_grid.index:
            self.net.ext_grid.loc[ext_grid, 'q_mvar'] = actions[-1] #if len(action_set) > 1 else 0
            self.net.ext_grid.loc[ext_grid, 'p_mw'] = actions[-2]

        ### Add uncertainty to forecasts
        if not evaluate:
            for load in self.net.load.index:
                rand = random.uniform(0.95,1.05)
                if self.net.load.name[load] != 'Battery Q' and self.net.load.name[load] != 'Battery Q2':
                    p_mw = self.net.load.loc[load, 'p_mw']
                    q_mvar = self.net.load.loc[load, 'q_mvar']
                    self.net.load.loc[load, 'p_mw'] = p_mw * rand
                    self.net.load.loc[load, 'q_mvar'] = q_mvar *rand


        self.timestep+=1

        ### Run PowerFlow with selected actions and current system state as input
        pp.runpp(self.net, algorithm = 'bfsw',init_vm_pu='results')

        ### load new forecasts for loads and renewables
        self.get_net_data()

        ### get new system state from network
        state, _ = self.get_state()

        ### calculate reward for selected actions and new state
        reward, violation = self.get_reward(state,actions)

        ### check if episode done
        done = self.check_done()

        return state, reward, actions, done, violation
