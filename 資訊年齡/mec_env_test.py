import numpy as np
import queue
from scipy.special import jv
import os
# from replay_buffer import ReplayBuffer

class Offload:

    def __init__(self, args):

        os.environ['CUDA_VISIBLE_DEVICES']=args.cuda
        # INPUT DATA
        self.epsilon = 0.9
        self.n_iot = args.num_iot
        self.n_edge = args.num_edge
        self.n_time = args.num_time
        self.n_size = args.task_size
        self.current_time = 0
        self.last_action = 0
        self.current_iot = 0
        self.queue_limit = 1000
        self.wait_max = 10
        self.gamma = args.gamma
        self.ddpg_gamma = 1.5

        if args.LOGNORMAL == 1:
            self.LOGNORMAL = True
        else:
            self.LOGNORMAL = False
        self.lognormal_variance = args.LOG_VARIANCE

        if args.FRACTION == 1:
            self.FRACTION = True
        else:
            self.FRACTION = False

        self.wait_state = np.zeros(1)
        self.STATE_STILL = args.STATE_STILL
        # self.replay_buffer = ReplayBuffer(n_iot=self.n_iot, max_size=1000000, state_dim=1, action_dim=1, batch_size=64)


        # CONSIDER A SCENARIO RANDOM IS NOT GOOD
        # LOCAL CAP SHOULD NOT BE TOO SMALL, OTHERWISE, THE STATE MATRIX IS TOO LARGE (EXCEED THE MAXIMUM)

        self.comp_cap_iot = args.comp_iot
        if args.comp_edge < 0.001:
            self.comp_cap_edge = args.comp_cap_edge
        else:
            self.comp_cap_edge = args.comp_edge * np.ones(self.n_edge)
        self.comp_density  = args.comp_density
        self.n_size = args.task_size # TASK SIZE / M bits
        self.bit_size = args.task_size * 1e6  # Mbits → bits

        # ACTION: 0, local; 1, edge 0; 2, edge 1; ...; n, edge n - 1
        self.n_actions = 1 + self.n_edge
        self.n_features = self.n_edge
        self.wait_features = 1

        # === Energy Model Parameters ===
        self.kappa = 1e-28
        self.f_iot = 1.2e9
        self.phi = 100
        self.beta = np.random.uniform(1e-11, 1e-9)

    def execute_action(self, RL, state):
        action = RL.choose_action(state)
        return action

    def random_action(self):
        action = np.random.randint(self.n_actions)
        return action

    def auto_action(self, state):
        index = np.argmin(state)
        if state[index] < 4:
            action = index + 1
        else:
            action = 0
        return action

    def iot_process(self, size, compacity, density):
        expected_time = size * density / compacity
        if self.LOGNORMAL:
            actual_time = np.random.lognormal(np.log(expected_time), self.lognormal_variance)
        else:
            actual_time = np.random.exponential(expected_time)
        # actual_time = expected_time

        # === 本地運算能量 ===
        L = density * self.bit_size*100
        E_loc = self.kappa * L * (self.f_iot ** 2)
        self.E_loc_now = E_loc
        self.E_tx_now = 0.0

        return actual_time, expected_time

    def edge_process(self, size, compacity, density):
        expected_time = size * density / compacity
        if self.LOGNORMAL:
            actual_time = np.random.lognormal(np.log(expected_time), self.lognormal_variance)
        else:
            actual_time = np.random.exponential(expected_time)
        # actual_time = expected_time

        # === 傳輸能量 ===
        E_tx = self.beta * self.bit_size  # 1.66
        self.E_tx_now = E_tx
        self.E_loc_now = 0.0

        return actual_time, expected_time

    def Rayleigh(self, h_last):
        T = 1
        fd = 10
        rou = jv(0, 2*np.pi*fd*T)
        alpha = 1
        p = 1
        B = 1
        data_size = 0.1
        device_num = self.n_iot

        e = complex(np.random.randn(), np.random.randn()/np.sqrt(2))
        h = rou*h_last + np.sqrt(1-rou*rou)*e
        g = abs(h) * abs(h) * alpha
        sigma2 = 0.1
        gamma = np.zeros(device_num)
        for i in range(device_num):
            gamma[i] = g[i] * p / (sum(g)*p - g[i]*p + sigma2)

        C = B * np.log2(1+gamma)
        transmit_time = data_size / C
        return transmit_time, h

    def reset(self, RL):
        self.current_time = 0 #重置時間與基本變數
        self.action_store = np.zeros(self.n_iot, dtype=int)

        #AoI 初始化 AoI相關變數
        self.duration_store = np.zeros([self.n_iot, 2])
        self.aoi = np.zeros(self.n_iot)
        self.aoi_sum = np.zeros(self.n_iot)
        self.aoi_average = np.zeros(self.n_iot)
        self.wait_time = np.zeros(self.n_iot)
        self.old_aoi = 0.0
        self.new_aoi = 0.0

        self.Start_Time = np.zeros(self.n_iot) #初始化任務計時器與 Queue 狀態
        self.End_Time = np.zeros(self.n_iot)

        self.queue_indicator = np.zeros(self.n_iot, dtype=int)

        self.Z_queue = np.zeros(self.n_iot)
        self.E_loc_history, self.E_tx_history, self.E_harv_history, self.net_count = [],[],[],[]

        self.loc_count = 0
        self.tx_count = 0
        self.drop_count = 0
        self.episode_used = 0

        self.DRQN_reward, self.AoI_record, self.dz_record = [],[],[]
        self.w_reward, self.w_time, self.run_time = [],[],[]

        # QUEUE INITIALIZATION: 0 -> iot; 1 -> remain time
        # edge节点等待计算队列
        self.Queue_edge_wait = list() #初始化 Edge server 等待佇列
        for edge in range(self.n_edge):
            self.Queue_edge_wait.append(queue.Queue())

        # TASK INDICATOR
        # iot task 0:remain 1:iot 2:edge+1 3:start_time
        self.task_iot = list()  #初始化 IoT 任務資訊表
        for iot in range(self.n_iot):
            self.task_iot.append(np.zeros(4))

        # edge处理中的iot  初始化 Edge server 狀態
        self.task_edge = -1 * np.ones(self.n_edge, dtype=int)
        #edge队列长度
        self.queue_length_edge = np.zeros(self.n_edge, dtype=int)
        self.current_state = np.hstack((self.queue_length_edge))
        self.task_edge_next = self.task_edge
        for edge in range(self.n_edge):
            self.queue_length_edge[edge] = self.Queue_edge_wait[edge].qsize()

        self.current_duration = 0

        #初始化动作  初始化 IoT 任務的初始動作（auto_action）
        for iot in range(self.n_iot):

            iot_state = np.hstack((self.queue_length_edge))
            action = self.auto_action(iot_state)
            self.action_store[iot] = action

            self.task_iot[iot][1] = iot
            self.task_iot[iot][2] = action
            self.task_iot[iot][3] = self.current_time

            if action > 0: #分配到edge  設定任務並分配到 edge 或 local
                edge = action - 1
                process_duration, expected_time = self.edge_process(self.n_size, self.comp_cap_edge[edge], self.comp_density)
                if self.task_edge[edge] == -1:
                    self.task_edge[edge] = iot
                else:
                    self.Queue_edge_wait[edge].put(iot)
                    process_duration *= self.queue_limit
                    self.queue_indicator[iot] = 1
                self.queue_length_edge[edge] += 1

            else: #分配到iot
                process_duration, expected_time = self.iot_process(self.n_size, self.comp_cap_iot, self.comp_density)

            self.task_iot[iot][0] = process_duration


        #task_iot 排序  排序任務，決定最先完成的
        self.task_iot = sorted(self.task_iot, key=lambda x:x[0])


        #state store initial         初始化狀態儲存空間
        self.state_store_now = list()
        for iot in range(self.n_iot):
            current_state = np.hstack((self.queue_length_edge))
            self.state_store_now.append(current_state)

        self.state_store_last = self.state_store_now.copy()

        self.wait_state_store_now = list()
        for iot in range(self.n_iot):
            wait_state = [self.current_duration]
            self.wait_state_store_now.append(wait_state)

        self.wait_state_store_last = self.wait_state_store_now.copy()

        self.wait_mode = np.zeros(self.n_iot, dtype=int)

    #allocate step
    def render(self):
        #選出「下一個會發生的事件」
        self.task_iot = sorted(self.task_iot, key=lambda x:x[0]) #先依照 剩餘時間 排序，取最前面（最先結束）的任務當作「下一個事件」。
        self.current_iot = round(self.task_iot[0][1]) #這次輪到做決策的 IoT 裝置 index。

        #current_time update 系統時間往前推進，等於「下一個事件發生所需的時間」。
        time_passed = self.task_iot[0][0]
        self.current_time = self.current_time + time_passed

        #iot remain update  所有在「運行中」的任務都把剩餘時間扣掉
        for iot in range(self.n_iot):
            iot_index = round(self.task_iot[iot][1])
            if self.queue_indicator[iot_index] == 0:
                self.task_iot[iot][0] -= time_passed

        #如果当前任务不在等待模式  如果這個任務是「等待模式」就跳過以下計算 1是等待模式
        if self.wait_mode[self.current_iot] == 0: #要更新 edge 狀態 + AoI

            #如果该任务在edge上运行  若事件是「某 edge 上的任務完成」→ 處理 edge 佇列
            if self.task_iot[0][2] > 0:  # action>0 表示在 edge 上跑
                current_edge = round(self.task_iot[0][2]) - 1
                if self.Queue_edge_wait[current_edge].empty():
                    self.task_edge[current_edge] = -1 # 該 edge 變成閒置
                else:
                    #对应edge等待序列释放一个至运行状态
                    task_iot = self.Queue_edge_wait[current_edge].get() # 拉一個來跑
                    for index in range(self.n_iot):
                        if self.task_iot[index][1] == task_iot:
                            self.task_iot[index][0] /= self.queue_limit # 解除「排隊拉長時間」的效果
                            iot_index = round(self.task_iot[index][1])
                            self.queue_indicator[iot_index] = 0 # 不再算排隊
                    self.task_edge[current_edge] = task_iot # 指派新的正在運行的 IoT
                self.queue_length_edge[current_edge] -= 1


            # aoi update

            self.current_duration = self.current_time - self.task_iot[0][3]
            self.last_duration = self.duration_store[self.current_iot,1]
            self.duration_store[self.current_iot, 0] = self.last_duration
            self.duration_store[self.current_iot, 1] = self.current_duration
            if self.last_duration + self.current_duration != 0:
                self.wait = self.wait_time[self.current_iot]
                self.aoi = 0.5 * ((self.last_duration + self.wait + self.current_duration) ** 2
                                     - self.current_duration ** 2)/(self.last_duration + self.wait + self.current_duration)
                self.aoi_sum[self.current_iot] += (self.last_duration + self.wait + self.current_duration) ** 2 - self.current_duration ** 2   # 加總梯形面積
                self.aoi_average[self.current_iot] = 0.5 * self.aoi_sum[self.current_iot] / self.current_time  #SMG*****************************************************************
                self.old_aoi = self.new_aoi
                self.new_aoi = abs(self.aoi_sum[self.current_iot] - self.gamma * self.current_time)   #γ·(Y+Z)***************************************************************45?
                # self.ddpg_aoi = abs(self.aoi_sum[self.current_iot] - self.ddpg_gamma * self.current_time)

            # === (1) 時間間隔 Δt ===
            delta_t = time_passed
            # === (2) 太陽能收穫功率（隨機產生） ===
            P_harv = np.random.uniform(0.3, 0.5)  #太陽能板 0.2,0.4能量不足 0.4,0.6能量充足 0.3,0.5差不多
            # === (3) 收穫能量 ===
            E_harv = P_harv * delta_t
            # === (4) 存起來，方便後面算 E_net 與虛擬佇列 ===
            self.E_harv_now = E_harv

            E_used = self.E_tx_now + self.E_loc_now
            E_harv = self.E_harv_now
            E_net = E_used - E_harv
            self.episode_used += E_used

            if not hasattr(self, 'E_loc_history'):
                self.E_loc_history, self.E_tx_history, self.E_harv_history = [], [], []
            self.E_loc_history.append(self.E_loc_now)
            self.E_tx_history.append(self.E_tx_now)
            self.E_harv_history.append(self.E_harv_now)
            self.net_count.append(E_net)

            # 更新虛擬佇列
            if not hasattr(self, 'Z_energy'):
                self.Z_energy = np.zeros(self.n_iot)
            if not hasattr(self, 'Z_last'):
                self.Z_last = np.zeros(self.n_iot)
            self.Z_energy[self.current_iot] = max(0, self.Z_energy[self.current_iot] + (E_net - 0))
            # DPP 懲罰
            # dpp_penalty = self.Z_energy[self.current_iot] * (E_net - 0) 改點1

            # === 取變化量做懲罰 ===
            self.dZ = self.Z_energy[self.current_iot] - self.Z_last[self.current_iot]
            self.Z_last[self.current_iot] = self.Z_energy[self.current_iot]

            # state_store update

            # self.current_state = np.hstack((self.queue_length_edge, self.wait_time[self.current_iot]))
            # self.current_state = np.hstack((min(self.queue_length_edge),self.wait_time[self.current_iot]))
            # self.current_state = np.array(min(self.queue_length_edge))
            self.current_state = np.array(self.queue_length_edge)                                    #environment observation***************************************
            self.state_store_last[self.current_iot] = self.state_store_now[self.current_iot]
            self.state_store_now[self.current_iot] = self.current_state

            # self.wait_state = np.hstack((self.queue_length_edge, self.current_duration))
            # self.wait_state = np.array(self.queue_length_edge)
            # self.wait_state = np.array([min(self.queue_length_edge)])
            self.wait_state = np.array([min(self.queue_length_edge), self.current_duration])
            self.wait_state_store_last[self.current_iot] = self.wait_state_store_now[self.current_iot]
            self.wait_state_store_now[self.current_iot] = self.wait_state


            # 训练元组                                                         獎勵等資料打包成 RL 訓練
            self.observation = self.state_store_last[self.current_iot]
            self.action = self.action_store[self.current_iot]
            if self.FRACTION:
                self.aoi_term = self.new_aoi / 1000
            else:
                self.aoi_term = self.aoi / 100
            lambda_dz = getattr(self, "lambda_dz", 20)  #權重 20 0.1   權重NO.1
            lambda_Z = 0.1   #權重NO.2
            self.reward = - (self.aoi_term + lambda_dz * self.dZ + lambda_Z * self.Z_energy[self.current_iot])   #改點2
            self.observation_next = self.state_store_now[self.current_iot]
            self.DRQN_reward.append(self.reward)
            self.AoI_record.append(self.aoi_term)
            self.dz_record.append(self.dZ*20)  #改點3

            self.wait_observation = self.wait_state_store_last[self.current_iot]
            self.wait_action = self.wait_time[self.current_iot]
            if self.FRACTION:
                self.wait_reward = -( self.new_aoi / 10000 + (lambda_dz * 0.1) * self.dZ + (lambda_Z * 0) * self.Z_energy[self.current_iot])    #改點4  權重NO.3 權重NO.4
                # self.wait_reward = -( self.new_aoi / 10000 + (lambda_dz * 0.1) * self.dZ )
            else:
                self.wait_reward = - self.aoi / 100
            self.wait_observation_next = self.wait_state_store_now[self.current_iot]
            self.w_reward.append(self.wait_reward*0.1)

        return self.current_iot, self.current_state, self.wait_state

    def execute_offload(self, action, process_duration):

        self.wait_mode[self.current_iot] = 0

        #action_store update
        self.action_store[self.current_iot] = action

        #task main list update
        self.task_iot[0][2] = action
        self.task_iot[0][3] = self.current_time

        # 计算相应的持续时间
        if action == 0:
            self.task_iot[0][0] = process_duration
            self.loc_count += 1
            self.run_time.append(process_duration)
        else:
            current_edge = action - 1
            self.task_iot[0][0] = process_duration
            self.tx_count += 1
            self.run_time.append(process_duration)
            if self.task_edge[current_edge] == -1: # 這個 edge 現在是空閒的：直接讓它開工
                self.task_edge[current_edge] = self.current_iot
                self.task_iot[0][0] = process_duration
            else: # 這個 edge 忙：先排到它的等待佇列
                self.Queue_edge_wait[current_edge].put(self.current_iot)
                self.task_iot[0][0] = self.queue_limit * process_duration
                iot_index = round(self.task_iot[0][1])
                self.queue_indicator[iot_index] = 1 # 標記「這台在排隊中」
            self.queue_length_edge[current_edge] += 1

        self.delay = self.current_duration


        return None


    # wait time step
    def execute_wait(self, wait_action):

        self.wait_mode[self.current_iot] = 1
        self.task_iot[0][0] = wait_action
        self.wait_time[self.current_iot] = wait_action
        self.w_time.append(wait_action)
        return None