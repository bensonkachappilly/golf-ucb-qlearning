# Golf Club Selection Using Bandit Algorithms and Q-Learning

## Problem Statement
Selecting the optimal golf club in uncertain conditions can be framed as a sequential decision-making problem under uncertainty.  
This project models a simplified golf scenario as a multi-armed bandit / short-horizon Markov Decision Process (MDP), where an agent must choose between different clubs to reach a terminal hole state while maximizing cumulative reward.

The objective is to:
- Establish a baseline bandit solution using the Upper Confidence Bound (UCB) algorithm
- Improve performance by incorporating a Q-function update that accounts for expected future rewards


## Dataset
This project uses a synthetic, environment-defined dataset rather than an external dataset.

- States: 8 discrete golf markers (positions on the course)
- Actions: 3 golf clubs  
  - Putter (1 step)  
  - Iron (2 steps)  
  - Driver (3 steps)
- Rewards: Predefined reward values assigned to each marker
- Terminal States: Markers 6 and 7

The environment includes stochasticity through a random swing variation in the range `[-1, 1]`.


## Approach

### Baseline: Upper Confidence Bound (UCB)
- Treats each state–action pair as a bandit arm
- Balances exploration and exploitation using a confidence bound
- Updates action values using an incremental mean estimate

### Improved Method: Q-Function with UCB Exploration
- Extends the bandit approach by incorporating a Q-learning style update
- Uses a discounted future reward estimate:
  \[
  Q(s,a) \leftarrow Q(s,a) + \alpha \left(r + \gamma \max_a Q(s',a) - Q(s,a)\right)
  \]
- Retains UCB-based exploration during training

A Tkinter-based GUI animates the learned policy by visualizing the agent’s path over the golf course.


## Results

- The Q-function approach demonstrates more consistent path selection toward high-reward terminal states compared to the pure UCB baseline.
- Lower learning rates (`α = 0.1`) resulted in improved convergence stability.
- The learned policy successfully avoids low-reward intermediate states in most rollouts.

Outputs include:
- Learned Q-value table
- Total cumulative reward for a rollout
- Animated visualization of the chosen path


## Tech Stack
- Python  
- NumPy  
- Tkinter (GUI & animation)  
- Pillow (image handling)

## How to Run

1. Install dependencies:
```bash
conda env create -f environment.yml
conda activate rlgolf
```

2. Ensure that golf_map.png is present is in the current working directory

3. Run the program 
```bash
python program.py
```
To switch between algorithms, edit def_main() by placing commenting out the algorithm:
e.g. 
Q = UCB()
#Q\ = Q_function()

## Limitations
-The “swing” randomness can occasionally overshoot terminal states; the code clips movement to state 7.
-The rollout policy after training is greedy: argmax(Q[state]).

## Future Improvements 
-Log and plot reward convergence over training iterations
-Run multiple episodes and report average performance
-Add alternative exploration strategies (ε-greedy, softmax)


