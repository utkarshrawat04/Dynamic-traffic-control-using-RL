# Dynamic Traffic Control Using Reinforcement Learning

## 📋 Project Overview

This project implements and compares multiple **Reinforcement Learning (RL) algorithms** for dynamic traffic light control in urban environments using **SUMO (Simulation of Urban MObility)**. The system learns optimal traffic light phasing strategies to minimize vehicle waiting times and reduce traffic congestion.

### 👥 Team Members
- **Aditi Pawar** (I004)
- **Shreyas Sawant** (I059) 
- **Utkarsh Rawat** (I069)
- **B.Tech Artificial Intelligence**

### 🔗 Reference
- [YouTube Tutorial Reference](https://www.youtube.com/watch?v=NOPn9sE0AdY)

## 🚀 Quick Start

### Prerequisites
```bash
# Install required Python packages
pip install torch numpy matplotlib traci sumolib
```

### Basic Usage
```python
# Train and compare all algorithms
best_algo, results, models = compare_all_algorithms_with_baseline(
    model_name="traffic_comparison", 
    epochs=100, 
    steps=1000
)

# Test the best model
test_results = test_trained_model_with_normal_comparison(
    algorithm="A2C",
    model_name="traffic_comparison_best",
    episodes=5,
    steps=1000
)
```

## 🛠️ Complete Setup Instructions

### 1. SUMO Installation
1. Download and install SUMO from [official website](https://www.eclipse.org/sumo/)
2. Set environment variable `SUMO_HOME` to your installation directory
3. Add SUMO tools to Python path (automatically handled in code)

### 2. Map Setup

#### Step 1: Export OSM File
- Open **OpenStreetMap**
- Select your desired region
- Export as **map.osm**

#### Step 2: Copy Typemap File
```bash
# Copy from SUMO installation directory
cp "C:\Program Files (x86)\Eclipse\Sumo\data\typemap\osmNetconvert.typ.xml" .
```

#### Step 3: Generate Network File
```bash
netconvert --osm-files map.osm -o test.net.xml -t osmNetconvert.typ.xml --xml-validation never
```

#### Step 4: Import Additional Polygons
- Visit [SUMO Wiki - OSM Import](https://sumo.dlr.de/wiki/Networks/Import/OpenStreetMap)
- Copy the provided polygon code
- Save as **typemap.xml** in working folder

#### Step 5: Convert Polygons
```bash
polyconvert --net-file test.net.xml --osm-files map.osm --type-file typemap.xml -o map.poly.xml --xml-validation never
```

#### Step 6: Copy Random Trips Script
```bash
# Copy from SUMO tools directory
cp "C:\Program Files (x86)\Eclipse\Sumo\tools\randomTrips.py" .
```

#### Step 7: Generate Random Trips
```bash
python randomTrips.py -n test.net.xml -r map.rou.xml -e 1000 -l --validate
```

#### Step 8: Create Simulation Configuration
Create **map.sumo.cfg** with:
```xml
<configuration>
    <input>
        <net-file value="test.net.xml"/>
        <route-files value="map.rou.xml"/>
        <additional-files value="map.poly.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="10000"/>
    </time>
</configuration>
```

## 🧠 RL Algorithms Implemented

### 1. 🎯 DQN (Deep Q-Network)
**Type:** Value-based  
**Approach:** Learns Q-function using neural networks  
**Key Features:**
- Experience replay for stability
- Fixed target network
- ε-greedy exploration
- Best for discrete action spaces

### 2. 🚀 PPO (Proximal Policy Optimization)  
**Type:** Policy-based  
**Approach:** Direct policy optimization with clipping  
**Key Features:**
- Actor-critic architecture
- Clipped objective for stability
- Generalized Advantage Estimation (GAE)
- Excellent for continuous control

### 3. ⚡ A2C (Advantage Actor-Critic)
**Type:** Policy-based  
**Approach:** Advantage-based policy updates  
**Key Features:**
- Single network with actor-critic heads
- Simpler than PPO
- Good balance of performance and complexity

### 4. 📊 SARSA (State-Action-Reward-State-Action)
**Type:** Tabular, On-policy  
**Approach:** Traditional Q-learning variant  
**Key Features:**
- Q-table based
- On-policy updates
- Simple but limited scalability

## 📈 Algorithm Comparison

| Method | Type | Scalability | Stability | Training Time | Best Use Case |
|--------|------|-------------|-----------|---------------|---------------|
| **DQN** | Value-based | Good | Medium | Medium | Discrete actions |
| **PPO** | Policy-based | Excellent | High | High | Complex environments |
| **A2C** | Policy-based | Good | Medium | Medium | Balanced performance |
| **SARSA** | Tabular | Poor | Low | Low | Small state spaces |

## 💻 Code Architecture

### Core Components

#### 1. Neural Network Models
```python
# DQN Model
class DQNModel(nn.Module)
# PPO Model  
class PPOModel(nn.Module)
# A2C Model
class A2CModel(nn.Module)
```

#### 2. RL Agents
```python
# DQN Agent
class DQNAgent
# PPO Agent  
class PPOAgent
# A2C Agent
class A2CAgent
# SARSA Agent
class SARSAAgent
```

#### 3. Training Functions
```python
def run_single_algorithm()  # Train individual algorithm
def compare_all_algorithms_with_baseline()  # Comprehensive comparison
def test_trained_model()  # Model evaluation
```

### Key Features

- **Multi-junction Support**: Handles multiple traffic lights simultaneously
- **Modular Design**: Easy to add new RL algorithms
- **Comprehensive Logging**: Training progress and performance metrics
- **Model Persistence**: Save/load trained models
- **Visualization**: Training curves and performance comparisons

## 🎯 State, Action, and Reward Design

### State Representation
```python
# Vehicle counts per lane at each junction
state = [vehicles_lane1, vehicles_lane2, ..., vehicles_laneN]
```

### Action Space
```python
# Two possible actions per junction
actions = [0, 1]  # Different traffic light phases
```

### Reward Function
```python
# Negative waiting time (penalize congestion)
reward = -waiting_time / 1000.0
```

## 📊 Performance Metrics

- **Total Waiting Time**: Sum of waiting times across all vehicles
- **Training Stability**: Standard deviation of performance
- **Convergence Speed**: Episodes to reach optimal performance
- **Generalization**: Performance across different traffic conditions

## 🏆 Results Summary

Based on comprehensive testing, the algorithms performed as follows:

### 🥇 Winner: A2C (Advantage Actor-Critic)
**Why A2C Outperformed:**
- **Balance of Simplicity and Power**: More efficient than PPO for this task
- **Effective Credit Assignment**: Advantage function helped identify optimal actions
- **Lower Training Overhead**: Faster convergence than DQN and PPO
- **Consistency**: Stable performance with low variance

### Performance Ranking:
1. **A2C** - Best overall performance and stability
2. **PPO** - Strong results but higher variability  
3. **SARSA** - Surprisingly good baseline, lacks scalability
4. **DQN** - Struggled with state space complexity
5. **Normal Traffic** - Worst performance, demonstrates RL value

## 🚦 Testing and Evaluation

### Testing a Single Model
```python
results = test_trained_model(
    algorithm="A2C",
    model_name="traffic_comparison_best",
    episodes=5,
    steps=1000,
    use_gui=False
)
```

### Comparative Testing
```python
comparison = test_trained_model_with_normal_comparison(
    algorithm="A2C", 
    model_name="traffic_comparison_best",
    episodes=5,
    steps=1000
)
```

## 📁 Project Structure

```
traffic-rl-project/
├── models/                 # Trained model files
│   ├── dqn/
│   ├── ppo/
│   ├── a2c/
│   └── sarsa/
├── plots/                  # Performance visualizations
├── results/               # Evaluation results
├── map.osm               # OpenStreetMap data
├── test.net.xml          # SUMO network file
├── map.rou.xml           # Vehicle routes
├── map.poly.xml          # Additional polygons
├── map.sumo.cfg          # Simulation configuration
└── Trafic_Simulation.py         # Main code file
```

## 🔧 Software Requirements

### Essential Dependencies
```bash
# Core Python packages
torch>=1.9.0
numpy>=1.21.0
matplotlib>=3.3.0
jupyter>=1.0.0

# SUMO integration
traci
sumolib

# Optional for advanced features
ipywidgets  # For Jupyter interactive controls
```

### SUMO Requirements
- **SUMO Version**: 1.15.0 or later
- **Python**: 3.7+
- **Operating System**: Windows/Linux/macOS

## 🎮 Usage Examples

### Training Individual Algorithm
```python
# Train DQN
dqn_results = run_single_algorithm(
    algorithm="DQN",
    model_name="my_dqn_model",
    epochs=50,
    steps=2000
)

# Train PPO  
ppo_results = run_single_algorithm(
    algorithm="PPO", 
    model_name="my_ppo_model",
    epochs=100,
    steps=2000
)
```

### Comprehensive Comparison
```python
# Compare all algorithms
best_algo, results, models = compare_all_algorithms_with_baseline(
    model_name="full_comparison",
    epochs=100,
    steps=5000
)
```

## 📈 Expected Outcomes

- **20-40% reduction** in waiting times compared to fixed-time traffic lights
- **Better traffic flow** during peak hours
- **Adaptive behavior** to changing traffic patterns
- **Scalable solution** for complex urban networks

## 🔮 Future Enhancements

1. **Multi-agent RL** for coordinated intersection control
2. **Transfer learning** between different city layouts
3. **Real-time adaptation** to accident scenarios
4. **Pedestrian integration** in reward function
5. **Emergency vehicle priority** handling


