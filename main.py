import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# =====================================================================
# 1. Flappy Bird 物理环境 v3.0 (支持提前看到即将到来的水管 + 历史动作记忆)
# =====================================================================
class FlappyBirdChampionEnv:
    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed)
        self.gravity = -0.0040     # 重力加速度
        self.flap_vel = 0.024      # 起跳初速度
        self.pipe_speed = 0.020    # 水管移动速度
        self.gap_half = 0.068      # 水管通道半宽
        self.reset()

    def reset(self):
        self.gap_center = 0.50
        self.next_gap_center = 0.50 + self.rng.uniform(-0.05, 0.05)
        self.bird_y = 0.50
        self.bird_vel = 0.0
        self.pipe_x = 0.60
        self.score = 0
        self.steps = 0
        self.crash_type = "None"
        # 记录过去 3 帧模型自己发出过的动作历史 H_t = [a_{t-1}, a_{t-2}, a_{t-3}]
        self.act_history = deque([0, 0, 0], maxlen=3)
        return self._get_obs()

    def _get_target_center(self):
        # 当快飞出当前水管(pipe_x < 0.08)时, 视线平滑切换到下一根水管中心
        if self.pipe_x < 0.08:
            alpha = np.clip(self.pipe_x / 0.08, 0.0, 1.0)
            return alpha * self.gap_center + (1.0 - alpha) * self.next_gap_center
        return self.gap_center

    def _get_obs(self):
        target_c = self._get_target_center()
        dy = self.bird_y - target_c
        # 5维完整状态: [高度差 dy*10, 速度 vy*40, 前1帧动作, 前2帧动作, 前3帧动作]
        # 就像 Gamer 论文里的 H_t 一样, 让模型知道自己刚刚有没有按过空格!
        return np.array([
            dy * 10.0,
            self.bird_vel * 40.0,
            float(self.act_history[0]),
            float(self.act_history[1]),
            float(self.act_history[2])
        ], dtype=np.float32)

    def step(self, issued_action, executed_action):
        """
        issued_action: 当前帧大脑发出的动作 (存入短期记忆 H_t)
        executed_action: 经过网络/推理延迟后实际到达游戏引擎的动作
        """
        self.steps += 1
        self.act_history.appendleft(issued_action)

        if executed_action == 1:
            self.bird_vel = self.flap_vel
        else:
            self.bird_vel += self.gravity

        self.bird_y += self.bird_vel
        self.pipe_x -= self.pipe_speed

        upper_wall = self.gap_center + self.gap_half
        lower_wall = self.gap_center - self.gap_half

        done = False
        if 0.0 <= self.pipe_x <= 0.42:
            if self.bird_y > upper_wall:
                done = True; self.crash_type = "Upper"
            elif self.bird_y < lower_wall:
                done = True; self.crash_type = "Lower"

        if self.bird_y > 0.95:
            done = True; self.crash_type = "Upper"
        elif self.bird_y < 0.05:
            done = True; self.crash_type = "Lower"

        if self.pipe_x < 0.0 and not done:
            self.score += 1
            self.pipe_x = 0.70
            self.gap_center = self.next_gap_center
            self.next_gap_center = np.clip(self.gap_center + self.rng.uniform(-0.05, 0.05), 0.35, 0.65)

        if self.score >= 50 or self.steps >= 2500:
            done = True

        return self._get_obs(), done


# =====================================================================
# 2. 确定性物理预演: 采集高玩在不同云游戏延迟下的完美提前量轨迹
# =====================================================================
def simulate_future_state(bird_y, bird_vel, target_center, future_actions, gravity, flap_vel):
    y, v = bird_y, bird_vel
    for act in future_actions:
        if act == 1:
            v = flap_vel
        else:
            v += gravity
        y += v
    return y - target_center, v


def collect_delayed_human_data(delay_frames, num_episodes=100):
    env = FlappyBirdChampionEnv(seed=100 + delay_frames)
    X_obs, Y_act = [], []
    target_trigger_dy = -0.044

    for _ in range(num_episodes):
        obs = env.reset()
        action_queue = deque([0] * delay_frames, maxlen=max(1, delay_frames))

        while True:
            target_c = env._get_target_center()
            pred_dy, pred_vy = simulate_future_state(
                env.bird_y, env.bird_vel, target_c,
                list(action_queue), env.gravity, env.flap_vel
            )
            # 当预判未来帧在往下掉且低于起跳线时, 触发提前量按空格
            noise = env.rng.normal(0, 0.0012)
            if (pred_dy < target_trigger_dy + noise) and (pred_vy <= 0.004):
                human_action = 1
            else:
                human_action = 0

            X_obs.append(obs.copy())
            Y_act.append(human_action)

            if delay_frames == 0:
                exec_action = human_action
            else:
                exec_action = action_queue[0]
                action_queue.append(human_action)

            obs, done = env.step(issued_action=human_action, executed_action=exec_action)
            if done:
                break

    return np.array(X_obs), np.array(Y_act)


# =====================================================================
# 3. 搭载 Adam 优化器与历史动作记忆的 VLA 策略网络 (ChampionVLA v3.0)
# =====================================================================
class ChampionVLA:
    def __init__(self, in_dim=5, hidden_dim=64, seed=42):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, np.sqrt(2.0 / in_dim), (in_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.normal(0, np.sqrt(2.0 / hidden_dim), (hidden_dim, 1))
        self.b2 = np.zeros(1)

    def forward(self, X):
        self.h = np.tanh(X @ self.W1 + self.b1)
        logits = self.h @ self.W2 + self.b2
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -15, 15)))

    def train(self, X, Y, epochs=80, batch_size=256, lr=0.015):
        Y = Y.reshape(-1, 1)
        n = len(X)
        pos_ratio = np.mean(Y) + 1e-5
        pos_weight = (1.0 - pos_ratio) / pos_ratio

        # 初始化 Adam 优化器的一阶动量 m 和二阶动量 v
        params = [self.W1, self.b1, self.W2, self.b2]
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        t_step = 0

        for ep in range(epochs):
            idx = np.random.permutation(n)
            for i in range(0, n, batch_size):
                t_step += 1
                xb, yb = X[idx[i:i+batch_size]], Y[idx[i:i+batch_size]]
                prob = self.forward(xb)

                weights = np.where(yb == 1, pos_weight, 1.0)
                d_logits = weights * (prob - yb) / len(xb)

                dW2 = self.h.T @ d_logits
                db2 = np.sum(d_logits, axis=0)
                dh = (d_logits @ self.W2.T) * (1.0 - self.h ** 2)
                dW1 = xb.T @ dh
                db1 = np.sum(dh, axis=0)

                grads = [dW1, db1, dW2, db2]
                for k in range(4):
                    m[k] = 0.9 * m[k] + 0.1 * grads[k]
                    v[k] = 0.999 * v[k] + 0.001 * (grads[k] ** 2)
                    m_hat = m[k] / (1.0 - 0.9 ** t_step)
                    v_hat = v[k] / (1.0 - 0.999 ** t_step)
                    params[k] -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)

    def predict(self, obs):
        prob = self.forward(obs.reshape(1, -1))[0, 0]
        return 1 if prob > 0.5 else 0


# =====================================================================
# 4. 评测 v3.0 模型
# =====================================================================
def evaluate_model(model, model_infer_delay_frames, num_test_eps=60):
    env = FlappyBirdChampionEnv(seed=2026)
    scores, upper_crashes, lower_crashes = [], 0, 0

    for _ in range(num_test_eps):
        obs = env.reset()
        infer_queue = deque([0] * model_infer_delay_frames, maxlen=max(1, model_infer_delay_frames))

        while True:
            # 无需任何外部硬编码抑制， 模型自己看着 obs 里的过去 3 帧动作历史 H_t 就能决定按不按!
            act = model.predict(obs)

            if model_infer_delay_frames == 0:
                exec_act = act
            else:
                exec_act = infer_queue[0]
                infer_queue.append(act)

            obs, done = env.step(issued_action=act, executed_action=exec_act)
            if done:
                scores.append(env.score)
                if env.crash_type == "Upper": upper_crashes += 1
                elif env.crash_type == "Lower": lower_crashes += 1
                break

    return np.mean(scores), upper_crashes, lower_crashes


if __name__ == "__main__":
    delays = [0, 1, 2, 3]
    delay_names = ["0ms (TP)", "25ms (RT)", "50ms (RT)", "75ms (RT)"]
    score_matrix = np.zeros((4, 4))

    print("运行 v3.0 (+历史动作记忆 H_t + Adam 优化器 + 前瞻视野)...\n")
    header = f"{'训练数据延迟 \\ 测试推理延迟':<22} | " + " | ".join([f"{name:<15}" for name in delay_names])
    print(header)
    print("-" * len(header))

    for i, d_data in enumerate(delays):
        X, Y = collect_delayed_human_data(d_data, num_episodes=100)
        model = ChampionVLA(in_dim=5, hidden_dim=64, seed=42)
        model.train(X, Y, epochs=80, lr=0.015)

        row_str = f"Data RTT = {delay_names[i]:<11} | "
        for j, d_infer in enumerate(delays):
            avg_score, up_c, low_c = evaluate_model(model, d_infer, num_test_eps=60)
            score_matrix[i, j] = avg_score
            cell = f"{avg_score:4.1f} (U{up_c:02d}/L{low_c:02d})"
            row_str += f"{cell:<15} | "
        print(row_str)

    plt.figure(figsize=(8, 6), dpi=150)
    plt.imshow(score_matrix, cmap="YlOrRd", vmin=0, vmax=50)
    plt.colorbar(label="Average Pipes Passed (Max = 50)")
    plt.xticks(range(4), delay_names)
    plt.yticks(range(4), [f"Data {n}" for n in delay_names])
    plt.xlabel("Deployment Inference Latency (tau_model) [TP vs. RT]")
    plt.ylabel("Cloud Gaming Data Latency (tau_data)")
    plt.title("Latency-Matched VLA Law (v3.0 with Action History H_t)")
    for i in range(4):
        for j in range(4):
            plt.text(j, i, f"{score_matrix[i, j]:.1f}", ha="center", va="center", color="black", fontweight="bold")
    plt.tight_layout()
    plt.savefig("latency_matching_heatmap_v3.png")
    print("\n实验完成")
