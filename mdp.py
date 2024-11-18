import gymnasium as gym
import matplotlib.pyplot as plt


LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

HOLE_REWARD = -1
GOAL_REWARD = 1000
DEFALT_REWARD = 1
LEARNING_RATE = 0.99

ENV_NAME = "FrozenLake-v1"


class MDP:
    def __init__(self, grid):
        self.n_rows = len(grid)
        self.n_cols = len(grid[0])
        self.grid = grid
        self.m = "".join(grid)
        self.value_function = [0] * self.n_rows * self.n_cols
        self.actions = [LEFT, RIGHT, UP, DOWN]

    def get_next_state(self, state, action):
        # Check if we are at leftmost column.
        if action == LEFT:
            return (
                state - 1
                if state % self.n_cols != 0 and self.m[state - 1] != "H"
                else state
            )
        # Check if we are at rightmost column.
        elif action == RIGHT:
            return (
                state + 1
                if state % self.n_cols != self.n_cols - 1 and self.m[state + 1] != "H"
                else state
            )
        # Check if we are at the topmost row.
        elif action == UP:
            return state - self.n_cols if state // self.n_cols != 0 else state
        # Check if we are at the bottom row.
        elif action == DOWN:
            return (
                state + self.n_cols
                if state // self.n_cols != self.n_rows - 1
                and self.m[state + self.n_cols] != "H"
                else state
            )

        return state

    def get_reward(self, state):
        if self.m[state] == "H":
            return -HOLE_REWARD
        elif self.m[state] == "G":
            return GOAL_REWARD
        else:
            return DEFALT_REWARD

    def run_iterative_policy_evaluation(self, steps=1000):
        max_deltas = []
        for step in range(steps):
            new_value_function = [0.0] * self.n_rows * self.n_cols
            max_delta = 0
            for state in range(len(self.m)):
                new_value = 0.0
                for action in self.actions:
                    next_state = self.get_next_state(state, action)
                    reward = self.get_reward(next_state)
                    new_value += (
                        1
                        / len(self.actions)
                        * (reward + LEARNING_RATE * self.value_function[next_state])
                    )
                new_value_function[state] = new_value
                delta = abs(new_value_function[state] - self.value_function[state])
                max_delta = (
                    delta
                    if not max_delta
                    else max(
                        max_delta,
                        abs(new_value_function[state] - self.value_function[state]),
                    )
                )
            max_deltas.append((step, max_delta))
            self.value_function = new_value_function
        self.plot_value_function_deltas(max_deltas)

    def run(self, episodes=1, video_folder=None):
        if video_folder:
            tmp_env = gym.make(
                ENV_NAME, desc=self.grid, is_slippery=False, render_mode="rgb_array"
            )
            env = gym.wrappers.RecordVideo(
                env=tmp_env,
                video_folder=video_folder,
                name_prefix="run",
                episode_trigger=lambda e: True,
            )
        else:
            env = gym.make(
                ENV_NAME, desc=self.grid, is_slippery=False, render_mode="human"
            )

        for _ in range(episodes):
            state = env.reset()[0]
            while True:
                max_value_action = None
                max_action_value = None
                for action in self.actions:
                    next_state = self.get_next_state(state, action)
                    if next_state == state:
                        continue
                    reward = self.get_reward(next_state)
                    value = (
                        1
                        / len(self.actions)
                        * (reward + LEARNING_RATE * self.value_function[next_state])
                    )
                    if not max_action_value or value > max_action_value:
                        max_action_value = value
                        max_value_action = action

                observation, _, terminated, truncated, _ = env.step(max_value_action)
                state = observation
                if terminated or truncated:
                    break
        env.close()

    def plot_value_function_deltas(self, deltas):
        steps, max_deltas = zip(*deltas)
        plt.plot(steps, max_deltas, linestyle="-", markersize=4)
        plt.xlabel("Step Number")
        plt.ylabel("Max Delta")
        plt.title(
            f"Max Delta by Step in Policy Evaluation ({self.n_rows} x {self.n_cols})"
        )
        plt.grid(True)
        plt.savefig(f"./assets/max_delta_plot_{self.n_rows}_{self.n_cols}.png")
