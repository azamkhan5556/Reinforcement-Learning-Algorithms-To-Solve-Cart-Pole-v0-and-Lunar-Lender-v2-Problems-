# README.md

# MuJoCo Installation Guide and Reinforcement Learning Tasks

This repository provides an installation guide for MuJoCo in a conda virtual environment and instructions for implementing DDPG algorithms to solve the "Pendulum-v0" and "HalfCheetah" tasks. 

## Installation Guide for MuJoCo

### Verified under Ubuntu 20.04

#### Step 1: Install Anaconda

```bash
wget https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh
sudo chmod +x Anaconda3-2023.09-0-Linux-x86_64.sh
./Anaconda3-2023.09-0-Linux-x86_64.sh
```

#### Step 2: Install Git

```bash
sudo apt install git
```

#### Step 3: Install the MuJoCo Library

1. Download the MuJoCo library from [MuJoCo Download](https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz)
2. Create a hidden folder:
    ```bash
    mkdir /home/username/.mujoco
    ```
3. Extract the library to the `.mujoco` folder.
4. Add the following lines to your `.bashrc` file:
    ```bash
    export LD_LIBRARY_PATH=/home/user_name/.mujoco/mujoco210/bin
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
    export PATH="$LD_LIBRARY_PATH:$PATH"
    export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so
    ```
5. Source the `.bashrc` file:
    ```bash
    source ~/.bashrc
    ```
6. Test the library installation:
    ```bash
    cd ~/.mujoco/mujoco210/bin
    ./simulate ../model/humanoid.xml
    ```
    If you encounter the error `ERROR: ld.so: object '/usr/lib/x86_64-linux-gnu/libGLEW.so' from LD_PRELOAD cannot be preloaded (cannot open shared object file): ignored.`, you can solve it by installing the packages `libglew1.5` and `libglew-dev`.

#### Step 4: Install the `mujoco-py` Package

```bash
conda create --name py38 python=3.8
conda activate py38
sudo apt update
sudo apt-get install patchelf python3-dev build-essential libssl-dev libffi-dev libxml2-dev libxslt1-dev zlib1g-dev libglew1.5 libglew-dev python3-pip
git clone https://github.com/openai/mujoco-py
cd mujoco-py
pip install -r requirements.txt
pip install -e . --no-cache
```

#### Step 5: Reboot Your Machine / Reopen a Terminal

#### Step 6: Run the Following Commands

```bash
conda activate py38
sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3
```

#### Step 7: Install D4RL

```bash
git clone https://github.com/Farama-Foundation/d4rl.git
cd d4rl
pip install -e .
```

#### Step 8: Run the Sanity Check Code on D4RL Repo

```python
import gym
import d4rl

# Create the environment
env = gym.make('maze2d-umaze-v1')

# d4rl abides by the OpenAI gym interface
env.reset()
env.step(env.action_space.sample())

# Each task is associated with a dataset
dataset = env.get_dataset()
print(dataset['observations'])

# Alternatively, use d4rl.qlearning_dataset which also adds next_observations.
dataset = d4rl.qlearning_dataset(env)
```

### Installation Guide for MuJoCo on Mac M1/M2

#### Step 1: Install Anaconda

```bash
wget https://repo.anaconda.com/archive/Anaconda3-2024.02-1-MacOSX-arm64.sh
sudo chmod +x Anaconda3-2024.02-1-MacOSX-arm64.sh
./Anaconda3-2024.02-1-MacOSX-arm64.sh
```

#### Step 2: Install Git

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install git
```

#### Step 3: Install the MuJoCo Library

Reference: [MuJoCo Issues](https://github.com/openai/mujoco-py/issues/682)

```bash
brew install glfw
mkdir -p $HOME/.mujoco/mujoco210
ln -sf /Applications/MuJoCo.app/Contents/Frameworks/MuJoCo.framework/Versions/Current/Headers/ $HOME/.mujoco/mujoco210/include
mkdir -p $HOME/.mujoco/mujoco210/bin
ln -sf /Applications/MuJoCo.app/Contents/Frameworks/MuJoCo.framework/Versions/Current/libmujoco.2.1.1.dylib $HOME/.mujoco/mujoco210/bin/libmujoco210.dylib
sudo ln -sf /Applications/MuJoCo.app/Contents/Frameworks/MuJoCo.framework/Versions/Current/libmujoco.2.1.1.dylib /usr/local/lib/
ln -s /Applications/MuJoCo.app/Contents/Frameworks/MuJoCo.framework/Versions/Current/libmujoco.2.1.1.dylib $HOME/.mujoco/mjpro200/bin/
ln -sf /opt/homebrew/lib/libglfw.3.dylib $HOME/.mujoco/mujoco210/bin
rm -rf /opt/homebrew/Caskroom/miniforge/base/lib/python3.9/site-packages/mujoco_py
export CC=/opt/homebrew/bin/gcc-11
```

#### Step 4: Install the `mujoco-py` Package

```bash
conda create --name py38 python=3.8
conda activate py38
pip install mujoco-py
```

#### Step 5: Install D4RL

```bash
git clone https://github.com/Farama-Foundation/d4rl.git
cd d4rl
pip install -e .
```

#### Step 6: Run the Sanity Check Code on D4RL Repo

```python
import gym
import d4rl

# Create the environment
env = gym.make('maze2d-umaze-v1')

# d4rl abides by the OpenAI gym interface
env.reset()
env.step(env.action_space.sample())

# Each task is associated with a dataset
dataset = env.get_dataset()
print(dataset['observations'])

# Alternatively, use d4rl.qlearning_dataset which also adds next_observations.
dataset = d4rl.qlearning_dataset(env)
```

## Reinforcement Learning Tasks

### Task (a): Solving the "Pendulum-v0" Problem with DDPG

The goal is to solve the "Pendulum-v0" environment using the DDPG algorithm.

1. Read through `ddpg.py`.
2. Implement the member functions of the classes `Actor`, `Critic`, and `DDPG` as well as the function `train`.
3. Summarize your results, including snapshots of Tensorboard or Weight & Bias records.
4. Document all hyperparameters (e.g., learning rates, NN architecture, and batch size).

### Task (b): Adapting DDPG to Solve the "HalfCheetah" Locomotion Task in MuJoCo

Based on the code for Task (a), adapt your DDPG algorithm to solve the "HalfCheetah" locomotion task.

1. Save your code in a file named `ddpg_cheetah.py`.
2. Add comments to your code for better readability.
3. Summarize your results and document all hyperparameters of your experiments.

(Note: As "HalfCheetah" is more challenging than "Pendulum", it might require tuning of hyperparameters and more training time.)

---

Feel free to reach out if you encounter any issues during the installation or implementation process. Enjoy your reinforcement learning experiments!
