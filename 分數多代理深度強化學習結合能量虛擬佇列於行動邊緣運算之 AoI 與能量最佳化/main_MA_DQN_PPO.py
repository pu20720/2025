# DRQN+RNN(GRU)-PPO MA
from csv import reader

import numpy as np
import torch
import matplotlib.pyplot as plt


# 检查CUDA是否可用
def check_cuda():
    if torch.cuda.is_available():
        print(f"Found {torch.cuda.device_count()} CUDA devices")
        return True
    else:
        print("No CUDA devices available, using CPU")
        return False


from mec_env_test import Offload
import os
import argparse
import random
from MA_DQN import DRQN, EpisodeBuffer
import gc
import time
import psutil


def safety_check(threshold_ram=95, threshold_gpu=95):
    """檢查 RAM / GPU 使用率，超過就暫停或強制釋放"""
    # --- CPU / RAM ---
    mem = psutil.virtual_memary()
    if mem.percent > threshold_ram:
        print(f"[Warning] RAM usage {mem.percent:.1f}% > {threshold_ram}% - 暫停釋放中...")
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(3)
        if psutil.virtual_memory().percent > threshold_gpu:
            print("[Memory still high] 程式安全退出以防黑屏")
            exit(0)

    # --- GPU (需安裝 nvidia-smi) ---
    try:
        import subprocess
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"]
        )
        used, total = map(float, result.decode().strip().split(','))
        gpu_usage = used / total * 100
        if gpu_usage > threshold_gpu:
            print(f"[Warning] GPU memory {gpu_usage:.1f}% > {threshold_gpu}% — 釋放中...")
            torch.cuda.empty_cache()
            time.sleep(3)
            if gpu_usage > threshold_gpu:
                print("[GPU still full] 程式安全退出以防 driver 崩潰")
                exit(0)

    except Exception:
        pass  # 如果沒裝 NVIDIA driver 就略過


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 设置随机数种子
# setup_seed(23)


def train(RL, RL_wait, args):
    TRANSMIT = args.TRANSMIT
    device = torch.device(args.cuda)
    # agent_memory = list()
    # for iot in range(env.n_iot):
    #     agent_memory.append(EpisodeMemory())

    # 設定（檔案路徑、統計容器）*******************************************************************************************
    reward_average_list = list()
    aoi_average_list = list()
    gamma_list = list()
    # drop_ratio_list = list()

    duration_list = np.zeros([env.n_iot, env.n_actions])
    duration_average_list = np.zeros([env.n_iot, env.n_actions])
    duration_count_list = np.zeros([env.n_iot, env.n_actions], dtype=int)
    wait_average_list = list()
    ori_average_list = list()
    actor_loss_average = list()
    critic_loss_average = list()
    d3qn_loss_average = list()
    transmit_average = list()

    episode_loc_count = []
    episode_tx_count = []
    episode_net_count = []
    episode_drop_count = []
    episode_DRQN_reward = []
    episode_w_reward = []
    episode_w_ave_time = []
    episode_run_ave_time = []
    episode_AoI = []
    episode_dz = []
    total_used = 0

    folder = args.folder
    sub_folder = args.subfolder
    filename1 = 'aoi_average_list.txt'
    filename2 = 'reward_average_list.txt'
    filename3 = 'gamma_list.txt'
    # filename4 = 'drop_ratio_list.txt'
    filename5 = 'wait_action_list.txt'
    filename6 = 'ori_action_list.txt'
    filename7 = 'actor_loss.txt'
    filename8 = 'critic_loss.txt'

    if args.TRANSMIT:
        filename10 = 'transmit_time.txt'

    folderpath = folder + '/' + sub_folder + '/'

    if not os.path.exists(folderpath):
        os.makedirs(folderpath)

    filepath1 = folderpath + filename1
    filepath2 = folderpath + filename2
    filepath3 = folderpath + filename3
    # filepath4 = folderpath + filename4
    filepath5 = folderpath + filename5
    filepath6 = folderpath + filename6
    filepath7 = folderpath + filename7
    filepath8 = folderpath + filename8
    if TRANSMIT:
        filepath10 = folderpath + filename10
    args_file = folderpath + 'args.txt'
    # np.savetxt(args_file, args)
    args_dict = args.__dict__
    with open(args_file, 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in args_dict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')
    # 重置 & 暫存*******************************************************************************************
    episode_time_list = list()
    for episode in range(args.num_episode):
        start_time = time.time()
        env.reset(RL)  # 初始化

        # Reset replay buffer
        agent_record = list()
        for iot in range(env.n_iot):
            agent_record.append(EpisodeBuffer())  # MA_DQN 公式48

        step_wait = np.zeros(env.n_iot, dtype=int)
        reward_all = np.zeros(env.n_iot)
        step = np.zeros(env.n_iot, dtype=int)
        step0 = 0
        # drop_all = 0

        gc.collect()
        torch.cuda.empty_cache()

        current_wait = np.zeros(env.n_iot)
        count_all = 0
        current_wait_store = list()

        # initialize hidden state for DRQN, GRU in PPO 清空/初始化各種計數器、PPO/DRQN 的 hidden state。
        # rnn_state = torch.zeros([1, args.rnn_input_size]).to(device)
        # h = RL[iot].q.init_hidden_state(batch_size=args.batch_size, training=False).to(device)
        # h, c = RL[iot].q.init_hidden_state(batch_size=args.batch_size, training=False)
        hidden_state = torch.zeros([1, args.rnn_input_size])
        hidden_state_c = torch.zeros([1, args.rnn_input_size])
        ppo_rnn_state = torch.zeros([1, args.rnn_hidden_dim]).to(device)
        ppo_hidden_state = torch.zeros([1, args.rnn_hidden_dim]).to(device)

        for i in range(env.n_iot):
            current_wait_store.append(np.zeros(env.n_iot))

        last_wait_store = list()  # 存每個 IoT 裝置之前的等待狀態 (用零初始化)
        for i in range(env.n_iot):
            last_wait_store.append(np.zeros(env.n_iot))
        wait_action_list = list()  # 用來存訓練過程中 IoT 的行為選擇紀錄 (等待的決策)
        ori_action_list = list()  # 用來存訓練過程中 IoT 的行為選擇紀錄 (例如 offload 的決策)
        actor_loss_list = list()  # 用來紀錄每個演算法的 loss（Actor-Critic、D3QN）還有傳輸次數，方便之後畫圖或分析。
        critic_loss_list = list()
        d3qn_loss_list = list()
        transmit_list = list()

        state_list = list()  # 存歷史狀態
        a_logprob_list = list()  # 動作的 log probability，訓練 PPO 的時候要用
        a_wait_logprob_list = list()  # 動作的 log probability，訓練 PPO 的時候要用
        state_val_list = list()  # Critic 輸出的 state value
        ar_list = list()  # Advantage-related 資料 (PPO 訓練需要)
        for i in range(args.num_iot):
            state_list.append(list())
            a_logprob_list.append(list())
            a_wait_logprob_list.append(list())
            state_val_list.append(list())
            ar_list.append(list())

        # d3qn epsilon 沒使用 （前期隨機、後期收斂）
        epislon = np.zeros(env.n_iot)  # 存每個 IoT agent 的 ε-greedy 探索率
        if episode <= args.OFFLOAD_EXPLORE:
            for iot in range(env.n_iot):
                if args.OFFLOAD_EXPLORE != 0:
                    epislon[iot] = 1 - episode / args.OFFLOAD_EXPLORE
                else:
                    epislon[iot] = 0
        for iot in range(env.n_iot):
            RL[iot].epsilon = 0  # 代表作者這裡其實強制 關掉隨機探索（只用策略選動作）

        # 單步主迴圈（while True）— 先拿到 state************************************************************

        f = 0
        while (True):
            if env.current_time > args.num_time:  # 如果超過單個 episode 的最大模擬時間，就跳出。
                break
            else:
                terminal = False  # 這份程式多設一個旗標，可能是標記「task完成或中斷」，和 done 分開處理
                done = False  # 通常是「環境結束」的旗標（RL 標準）。

            # RL take action and get next observation and reward
            for i in range(env.n_iot):
                current_wait[i] -= env.task_iot[0][3]
                """第 i 個 IoT 目前還需要等多久（秒數）-=表示「目前最前面這個 task 的執行時間 / 經過的時間片」。（程式把它當作「全局時間往前推進多少」）"""
                if current_wait[i] < 0:
                    current_wait[i] = 0

            while True:
                render_result = env.render()
                if render_result is not None:
                    break
            current_iot, current_state, wait_state = render_result

            # 兩條分支：Waiting(PPO) vs Offload(DRQN)*******************************************************
            # Waiting 分支****************************************************************
            if env.wait_mode[current_iot] == 0:
                # saving reward and is_terminals

                if WAIT and step_wait[current_iot] > 0:  # 把「上一個步驟」的軌跡塞進 PPO buffer
                    RL_wait[current_iot].buffer_new.states.append(state_list[current_iot])  # 狀態
                    RL_wait[current_iot].buffer_new.actions.append(ar_list[current_iot])  # 動作
                    RL_wait[current_iot].buffer_new.logprobs.append(a_wait_logprob_list[current_iot])  # 機率
                    RL_wait[current_iot].buffer_new.state_values.append(state_val_list[current_iot])  # 價值
                    # RL_wait[current_iot].buffer_new.hidden_states_c.append(ppo_hidden_state)
                    RL_wait[current_iot].buffer_new.rewards.append(env.wait_reward)  # 回饋
                    RL_wait[current_iot].buffer_new.is_terminals.append(done)  # 終止
                    RL_wait[current_iot].buffer_new.count += 1

                agent_record[current_iot].put([env.observation, env.action, env.reward, env.observation_next,
                                               terminal])  # 同步記 R L（D3QN）的 transition 與 reward
                reward_all[env.current_iot] += env.wait_reward

                if WAIT:
                    if args.model_training == 1 and RL_wait[
                        current_iot].buffer_new.count == args.wait_batch_size:  # 觸發 PPO 更新（buffer 湊滿就 update）
                        if RL_wait[
                            current_iot].buffer.count == 0:  # # 第一次更新：把 new buffer 直接變成正式 buffer，初始化舊 hidden，做一次 update
                            ppo_rnn_old = torch.zeros([1, args.rnn_hidden_dim]).to(device)
                            ppo_hidden_old = torch.zeros([1, args.rnn_hidden_dim]).to(device)
                            RL_wait[current_iot].buffer = RL_wait[current_iot].buffer_new
                            RL_wait[current_iot].update(ppo_rnn_old, ppo_hidden_old)
                            RL_wait[current_iot].buffer_new.clear()
                            ppo_rnn = ppo_rnn_state
                            ppo_hidden = ppo_hidden_state
                        else:  # 之後的更新：保留上一批的隱狀態做對照（old policy），再用新隱狀態更新（new policy）
                            ppo_rnn_old = ppo_rnn
                            ppo_hidden_old = ppo_hidden
                            ppo_rnn = ppo_rnn_state
                            ppo_hidden = ppo_hidden_state
                            RL_wait[current_iot].update(ppo_rnn_old, ppo_hidden_old)
                            RL_wait[current_iot].buffer = RL_wait[current_iot].buffer_new
                            RL_wait[current_iot].buffer_new.clear()

                    wait_action, a_wait_logprob_list[current_iot], state_list[current_iot], state_val_list[current_iot], \
                    ar_list[current_iot], ppo_rnn_state, ppo_hidden_state = \
                        RL_wait[current_iot].select_action(env.wait_state, ppo_rnn_state, ppo_hidden_state)
                    wait_action = (wait_action / 2 + 0.5) * args.action_range
                    step_wait[current_iot] += 1


                else:
                    # D3QN only
                    wait_action = args.wait_time  ## 固定等待時間，不學習

                wait_action_list.append(wait_action)  # 寫回環境 & 記帳
                # ori_action_list.append(ori_action)

                current_wait[current_iot] = wait_action
                env.execute_wait(wait_action)



            # Offload 分支（else:）********************************************************
            else:
                # Offload

                # action = env.auto_action( env.queue_length_edge)
                action = RL[current_iot].q.sample_action(current_state, epislon[current_iot])

                # action = np.random.randint(env.n_edge+1)
                if action == 0:  # 本地
                    process_duration, expected_time = env.iot_process(env.n_size, env.comp_cap_iot, env.comp_density)


                else:
                    current_edge = action - 1  # offload 給第 (action-1) 個 edge server
                    process_duration, expected_time = env.edge_process(env.n_size, env.comp_cap_edge[current_edge],
                                                                       env.comp_density)

                if process_duration < args.drop_coefficient * duration_average_list[current_iot][
                    action]:  # 這裡其實在模擬 系統是否能接受這個時間 太慢的任務會直接被 丟包
                    env.Start_Time[current_iot] = env.current_time
                    env.execute_offload(action, process_duration)
                    # drop_all += 1
                    count_all += 1
                else:  # task drop
                    original_wait = env.wait_time[current_iot]
                    env.execute_wait(process_duration)
                    env.wait_time[current_iot] += original_wait  ##
                    count_all += 1
                    env.drop_count += 1
                duration_list[current_iot][action] += process_duration  # 更新統計資訊（平均處理時間）
                duration_count_list[current_iot][action] += 1
                duration_average_list[current_iot][action] = duration_list[current_iot][action] / \
                                                             duration_count_list[current_iot][action]
                step[current_iot] += 1

            memorylen = len(agent_record[env.current_iot])  # 訓練條件觸發 累積到一批資料後，觸發一次 RL 訓練
            if args.model_training == 1 and memorylen > 0 and memorylen % args.batch_size == 0:
                RL[env.current_iot].train(agent_record[env.current_iot], device)
                agent_record[env.current_iot].clear()

                # episode_time = time.time() - time1
        # print("Episode Time: "+str(episode_time))
        # for i in range(env.n_iot):
        #     agent_memory[i].put(agent_record[i])
        aoi_average = 0
        # 計算 reward 與 AoI 的平均
        reward_average = 0

        for iot in range(env.n_iot):
            reward_average += reward_all[iot] / step[iot]  # 每個 IoT 的平均 reward

        reward_average /= env.n_iot  # 所有 IoT 的平均 reward

        aoi_average = np.mean(env.aoi_average)  # 這一回合所有 IoT 的 平均 AoI

        if len(actor_loss_list) > 0:
            actor_average = sum(actor_loss_list) / len(actor_loss_list)  # PPO actor 的 loss（等待時間調整的學習）
            actor_loss_average.append(actor_average)

            critic_average = sum(critic_loss_list) / len(critic_loss_list)  # PPO critic 的 loss（估計 value function）
            critic_loss_average.append(critic_average)

            d3qn_average = sum(d3qn_loss_list) / len(d3qn_loss_list)  # D3QN 的 loss（offloading 決策的學習）
            d3qn_loss_average.append(d3qn_average)  # 算出 episode 平均後存進各自的 list

        reward_average_list.append(reward_average)  # 每一回合的結果都記錄下來，用來畫曲線或分析 trend。
        aoi_average_list.append(aoi_average)
        gamma_list.append(env.gamma)

        # === 能量平均值統計 ===
        if hasattr(env, "E_loc_history") and len(env.E_loc_history) > 0:
            # 對所有 IoT 求平均（假設每個 IoT 在這回合都至少有一次能耗）
            E_loc_ave = np.mean(env.E_loc_history)
            E_tx_ave = np.mean(env.E_tx_history)
            E_harv_ave = np.mean(env.E_harv_history)
            Z_ave = np.mean(getattr(env, "Z_energy", np.zeros(env.n_iot)))
            net_ave = np.mean(env.net_count)

            # 把平均值記錄起來以便之後存檔
            if not hasattr(env, "log_energy"):
                env.log_energy = {"E_loc": [], "E_tx": [], "E_harv": [], "Z": [], "Net": []}
            env.log_energy["E_loc"].append(E_loc_ave)
            env.log_energy["E_tx"].append(E_tx_ave)
            env.log_energy["E_harv"].append(E_harv_ave)
            env.log_energy["Z"].append(Z_ave*0.1)
            env.log_energy["Net"].append(net_ave)

            env.E_loc_history.clear()
            env.E_tx_history.clear()
            env.E_harv_history.clear()
            env.net_count.clear()
        else:
            E_loc_ave = E_tx_ave = E_harv_ave = Z_ave = 0.0

        if args.num_episode - episode <= 100:
            total_used += env.episode_used

        episode_loc_count.append(env.loc_count)
        episode_tx_count.append(env.tx_count)
        episode_drop_count.append(env.drop_count)
        episode_DRQN_reward.append(np.mean(env.DRQN_reward))
        episode_w_reward.append(np.mean(env.w_reward))
        episode_AoI.append(np.mean(env.AoI_record))
        episode_dz.append(np.mean(env.dz_record))
        episode_w_ave_time.append(np.mean(env.w_time))
        episode_run_ave_time.append(np.mean(env.run_time))

        if len(wait_action_list) > 0:
            wait_average = sum(wait_action_list) / len(wait_action_list)  # 等待動作（PPO）在這回合的平均值
            wait_average_list.append(wait_average)
        if len(ori_action_list) > 0:
            ori_average = sum(ori_action_list) / len(ori_action_list)  # 原始行為的平均值（目前註解掉沒用到）
            ori_average_list.append(ori_average)

        # if episode < 6: #輸出訓練過程資訊
        #     print('Episode '+ str(episode) + ' Ave_wait: '+ str(wait_average) + ' AoI: '+ str(aoi_average))
        # else:
        #     print('Episode '+ str(episode) + ' Ave_wait: '+ str(wait_average) + ' AoI: '+ str(aoi_average) + ' Ave_AoI: ' + str(np.mean(aoi_average_list[-5:])))
        if episode < 6:
            print('Episode ' + str(episode) +
                  ' Ave_wait: ' + str(wait_average) +
                  ' AoI: ' + str(aoi_average) +
                  ' E_loc_ave: ' + str(np.mean(E_loc_ave)) +
                  ' E_tx_ave: ' + str(np.mean(E_tx_ave)) +
                  ' E_harv_ave: ' + str(np.mean(E_harv_ave)) +
                  ' Z_avg: ' + str(np.mean(Z_ave)))
        else:
            print('Episode ' + str(episode) +
                  ' Ave_wait: ' + str(wait_average) +
                  ' AoI: ' + str(aoi_average) +
                  ' Ave_AoI: ' + str(np.mean(aoi_average_list[-5:])) +
                  ' E_loc_ave: ' + str(np.mean(E_loc_ave)) +
                  ' E_tx_ave: ' + str(np.mean(E_tx_ave)) +
                  ' E_harv_ave: ' + str(np.mean(E_harv_ave)) +
                  ' Z_avg: ' + str(np.mean(Z_ave)))

        # gmma更新                                                         46**********************************************************************************************
        if episode > 0 and episode % 50 == 0:  # 引導 γ 接近 A / (Y + Z)   每 50 回合更新一次	避免不穩定跳動，像 soft update 的做法   (用在mec225行)
            env.gamma += 0.5 * (aoi_average - env.gamma)  # 收斂y
            print("[Cooling] 暫停 2 秒釋放記憶體")
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(2)

        # gat RETRAIN
        # if episode > 0 and episode % 200 == 0 and episode < 1001:
        # RL_wait = DDPG(state_dim=1, action_dim=1)

        episode_time = time.time() - start_time  # 記錄單次 episode 的耗時
        episode_time_list.append(episode_time)  # 也印出平均 episode 的耗時
        print('Episode Time: ' + str(episode_time))
        print('Average Episode Time: ' + str(np.mean(episode_time_list)))
        if episode % 100 == 0:  # 每 100 回合就把目前的曲線資料寫到檔案

            np.savetxt(filepath1, aoi_average_list)
            np.savetxt(filepath2, reward_average_list)
            np.savetxt(filepath3, gamma_list)
            # === 儲存能量紀錄 ===
            filepath_energy = folderpath + 'energy_log.txt'
            energy_data = np.column_stack([
                env.log_energy["E_loc"],
                env.log_energy["E_tx"],
                env.log_energy["E_harv"],
                env.log_energy["Z"]
            ])
            np.savetxt(filepath_energy, energy_data, header="E_loc_ave E_tx_ave E_harv_ave Z_ave", fmt="%.6e")
    print('reward100', np.mean(episode_DRQN_reward[-100:]))
    print('aoi_r100', np.mean(episode_AoI[-100:]))
    print('dz_r100', np.mean(episode_dz[-100:]))
    print('aoi100', np.mean(aoi_average_list[-100:]))
    print('Z100', np.mean(env.log_energy["Z"][-100:]))
    print('total used = ', total_used)
    print('wait time100 = ', np.mean(episode_w_ave_time[-100:]))
    print('run time100 = ', np.mean(episode_run_ave_time[-100:]))
    print('Loc count = ', np.mean(episode_loc_count[-100:]), ', Edge count = ', np.mean(episode_tx_count[-100:]), ', drop_count = ', np.mean(episode_drop_count[-100:]))
    print('game over')
    # === 能量變化趨勢圖 ===
    if hasattr(env, "log_energy"):
        plt.figure(figsize=(10, 5))
        plt.plot(env.log_energy["E_loc"], label="E_loc (local energy)")
        plt.plot(env.log_energy["E_tx"], label="E_tx (transmit energy)")
        plt.plot(env.log_energy["E_harv"], label="E_harv (harvested)")
        plt.plot(env.log_energy["Net"], label="Net energy")
        plt.legend()
        plt.title("Energy Terms per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Energy (J)")
        plt.savefig("Energy Terms per Episode.png")
        plt.show()

        plt.figure(figsize=(8, 5))
        plt.plot(episode_loc_count, label="local")
        plt.plot(episode_tx_count, label="Edge")
        plt.plot(episode_drop_count, label="Drop")
        plt.legend()
        plt.title("Loc Edge count")
        plt.xlabel("Episode")
        plt.ylabel("Count")
        plt.savefig("Loc Edge count.png")
        plt.show()

        plt.figure()
        plt.plot(env.log_energy["Z"], label="Z_energy (virtual queue)")
        plt.title("Virtual Queue Stability")
        plt.xlabel("Episode")
        plt.ylabel("Z value")
        plt.legend()
        plt.savefig("Virtual Queue Stability.png")
        plt.show()

        plt.figure()
        plt.plot(aoi_average_list, label="AoI average")
        plt.title("AoI Trend")
        plt.xlabel("Episode")
        plt.ylabel("AoI")
        plt.legend()
        plt.savefig("aoi_average.png")
        plt.show()

        plt.figure()
        plt.plot(episode_DRQN_reward, label='reward')
        plt.plot(episode_AoI, label='AoI')
        plt.plot(episode_dz, label='dz')
        plt.plot(env.log_energy["Z"], label="Z_energy (virtual queue)")
        plt.title("reward_record")
        plt.legend()
        plt.ylabel("Value")
        plt.xlabel("Episode")
        plt.savefig("reward_record.png")
        plt.show()

        plt.figure()
        plt.plot(episode_dz, label='dz')
        plt.title("reward_dz")
        plt.legend()
        plt.ylabel("Value")
        plt.xlabel("Episode")
        plt.savefig("reward_dz.png")
        plt.show()

        plt.figure()
        plt.plot(episode_AoI, label='AoI')
        plt.title("reward_AoI")
        plt.legend()
        plt.ylabel("Value")
        plt.xlabel("Episode")
        plt.savefig("reward_AoI.png")
        plt.show()

        plt.figure()
        plt.plot(episode_w_reward, label='reward')
        plt.plot(episode_AoI, label='AoI')
        plt.plot(episode_dz, label='dz')
        # plt.plot(env.log_energy["Z"], label="Z_energy (virtual queue)")
        plt.title("wait_reward_record")
        plt.legend()
        plt.ylabel("Value")
        plt.xlabel("Episode")
        plt.savefig("wait_reward_record.png")
        plt.show()

        plt.figure()
        plt.plot(episode_w_ave_time, label='wait_time')
        plt.title("ave_wait_time_record")
        plt.legend()
        plt.ylabel("time(sec)")
        plt.xlabel("Episode")
        plt.savefig("ave_wait_time_record.png")
        plt.show()

        plt.figure()
        plt.plot(episode_run_ave_time, label='run_time')
        plt.title("ave_run_time_record")
        plt.legend()
        plt.ylabel("time(sec)")
        plt.xlabel("Episode")
        plt.savefig("ave_run_time_record.png")
        plt.show()

# 添加设备检查
def check_cuda():
    if torch.cuda.is_available():
        print(f"Found {torch.cuda.device_count()} CUDA devices")
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
        return True
    else:
        print("No CUDA devices available, using CPU")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MEC-DRL')
    parser.add_argument('--mode', type=str, default='Waiting', help='choose a model: Offload_Only, Waiting')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate of DRQN')
    parser.add_argument('--comp_iot', type=float, default=2.5, help='Computing capacity of mobile device')
    parser.add_argument('--comp_edge', type=float, default=41.8, help='Computing capacity of edge device')
    parser.add_argument('--comp_cap_edge', type=float, nargs='+', default=[3, 8],
                        help='Computing capacity of edge device')
    parser.add_argument('--comp_density', type=float, default=0.297, help='Computing capacity of edge device')
    parser.add_argument('--num_iot', type=int, default=20, help='The number of mobile devices')
    parser.add_argument('--num_edge', type=int, default=2, help='The number of edge devices')
    parser.add_argument('--num_time', type=float, default=300, help='Time per episode')
    parser.add_argument('--num_episode', type=int, default=501, help='number of episode')
    parser.add_argument('--drop_coefficient', type=float, default=1.5, help='number of episode')
    parser.add_argument('--task_size', type=float, default=30, help='Task size (M)')
    parser.add_argument('--gamma', type=float, default=5, help='gamma for fractional')
    parser.add_argument('--folder', type=str, default='standard_Final', help='The folder name of the process')
    parser.add_argument('--subfolder', type=str, default='test_1125AoI_1(lam20_0.1_0.1_0)eng35', help='The sub-folder name of the process')
    parser.add_argument('--iot_step', type=int, default=0, help='step per iot')
    parser.add_argument('--wait_time', type=float, default=0, help='Fixed waiting time')
    parser.add_argument('--action_range', type=float, default=3, help='Waiting action range')
    parser.add_argument('--FRACTION', type=int, default=1, help='Have wait_rewardal AoI or not')

    parser.add_argument('--D3QN_NOISE', type=int, default=1, help='Have D3QN noise or not')
    parser.add_argument('--DDPG_NOISE', type=int, default=0, help='Have DDPG noise or not')
    parser.add_argument('--cuda', type=str, default='cuda:0', help='Using GPU')
    parser.add_argument('--STATE_STILL', type=int, default=0, help='STILL STATE?')
    parser.add_argument('--d3qn_step', type=int, default=10, help='D3QN Step')
    parser.add_argument('--WAIT_EXPLORE', type=int, default=300, help='Wait expoloer or not')
    parser.add_argument('--OFFLOAD_EXPLORE', type=int, default=0, help='Wait expoloer or not')
    parser.add_argument('--FULL_NOISE', type=int, default=1, help='Full noise or not')
    parser.add_argument("--hidden_width", type=int, default=64,
                        help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--use_tanh", type=float, default=True, help="Trick 10: tanh activation function")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--lr_a_o", type=float, default=0.10, help="Learning rate of actor")
    parser.add_argument("--lr_c_o", type=float, default=0.15, help="Learning rate of critic")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--wait_batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=16, help="Minibatch size")
    parser.add_argument("--rnn_input_size", type=int, default=8, help="Input size for GRU")
    parser.add_argument("--rnn_hidden_dim", type=int, default=8, help="Hidden size for GRU")
    parser.add_argument("--evaluate_freq", type=float, default=5e2,
                        help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--wait_lr_decay", type=float, default=10, help="lr decay episodes for updating policy")
    parser.add_argument("--pid", type=int, default=0, help="pid")
    parser.add_argument("--LOGNORMAL", type=int, default=0, help="Using lognormal as processing time or not")
    parser.add_argument("--LOG_VARIANCE", type=float, default=1, help="Lognormal variance")
    parser.add_argument("--random_seed", type=int, default=42, help="random_seed")
    parser.add_argument("--user_position", type=float, nargs='+',
                        default=[[20, 30], [50, 80], [10, 70], [60, 10], [90, 60], [10, 10], [20, 20], [30, 30],
                                 [40, 40], [50, 50], [60, 60], [70, 70], [80, 80], [90, 90], [100, 100], [110, 110],
                                 [120, 120], [130, 130], [140, 140], [150, 150]], help="user position")
    parser.add_argument("--server_positions", type=float, nargs='+', default=[[30, 40], [70, 50]],
                        help="server positions")
    parser.add_argument("--bandwidth", type=float, default=10e6, help="Bandwidth (Hz)")
    parser.add_argument("--model_training", type=int, default=1, help="Model training or not")
    parser.add_argument('--TRANSMIT', type=int, default=0, help='Have transmit or not')
    args = parser.parse_args()
    # 載入參數
    args.pid = os.getpid()
    # 根据CUDA可用性设置device
    if check_cuda():
        # args.cuda = args.cuda  # 保持原来的CUDA设备选择
        args.cuda = 'cuda'
    else:
        args.cuda = 'cpu'  # 如果没有CUDA设备，强制使用CPU

    args.state_dim = args.num_edge  # state 的維度，這裡設定成 edge 的數量（因為每個 state 可能就是描述各個 edge 的狀態）
    args.action_dim = args.num_edge + 1  # action 的數量，比 edge 多 1
    print(args)
    # GENERATE ENVIRONMENT
    setup_seed(args.random_seed)  # 固定隨機種子
    if args.TRANSMIT:  # 依照 --TRANSMIT 決定要用哪個環境類別 transmit預設為0
        from mec_env_test_transmit import Offload_Transmit  # 作者沒給

        env = Offload_Transmit(args)
    else:
        env = Offload(args)
    ob_shape = list([env.n_features])  # 把「環境 state 的特徵維度」存成一個 list for神經學習
    # GENERATE MULTIPLE CLASSES FOR RL
    RL = list()  # DRQN/D3QN 的 agent（處理 offload 決策）
    RL_wait = list()  # PPO 的 agent（處理等待時間決策，只有在 mode=Waiting 才會真的加東西）

    for i in range(args.num_iot):  # 有幾台 IoT，就建立幾個 DRQN agent 丟到 RL[i]
        RL.append(DRQN(args))  # DRQN 負責「要不要 offload、丟到哪一台 edge」
        if args.mode == 'Waiting':  # 如果是 Waiting 模式，再幫每台 IoT 多生一個 PPO（有 GRU），放到 RL_wait[i]，它負責「等多久」
            from RNN_PPO import PPO

            RL_wait.append(
                PPO(args=args, state_dim=args.num_edge, action_dim=1, lr_actor=0.0003, lr_critic=0.001, gamma=0.99,
                    K_epochs=10, eps_clip=0.2, has_continuous_action_space=True))
            WAIT = True  # 這個旗標讓後面主回圈分流：走 Waiting 分支（PPO）或 Offload 分支（DRQN）。
        elif args.mode == 'Offload_Only':
            WAIT = False

    print(f"Using device: {args.cuda}")

    # TRAIN THE SYSTEM
    train(RL, RL_wait, args)

    print('Training Finished')
# task_iot[][] > [剩餘時間, iot_id, action, start_time]