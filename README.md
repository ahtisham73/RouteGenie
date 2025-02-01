# 🚖🧞‍♂️ **RouteGenie – Intelligent Path Planning with RL**  
### 🤖🛤️🏁 Autonomous decision-making for smarter urban mobility  of Taxi/Cab

![Python](https://img.shields.io/badge/Python-3.10.12-3776AB?style=for-the-badge&logo=python&logoColor=white)  ![Reinforcement Learning](https://img.shields.io/badge/Reinforcement%20Learning-Q%20Learning-FF6F00?style=for-the-badge)  ![Gymnasium](https://img.shields.io/badge/OpenAI_Gymnasium-Taxi--v3-0081A7?style=for-the-badge)  
![Matplotlib](https://img.shields.io/badge/Matplotlib-Data_Visualization-11557C?style=for-the-badge)  ![NumPy](https://img.shields.io/badge/NumPy-Scientific_Computing-013243?style=for-the-badge) ![Seaborn](https://img.shields.io/badge/Seaborn-Statistical_Visualization-3776AB?style=for-the-badge)  ![Pygame](https://img.shields.io/badge/Pygame-Interactive_Simulation-FFCC00?style=for-the-badge)  

---

## 📜 **Introduction**  

🚀 **RouteGenie** is an advanced **Q-Learning-based reinforcement learning project** designed to solve the **Taxi-v3 environment**. The objective is to train an AI agent to efficiently **pick up and drop off passengers** while **maximizing cumulative rewards** in a grid-based environment.  

### **🔹 Key Features:**  
✔️ **Reinforcement Learning** using Q-Learning.  
✔️ **Intelligent Decision-Making** with reward optimization.  
✔️ **Dynamic Environment Handling** (Randomized Passenger & Taxi Locations).  
✔️ **Performance Evaluation** via animations & heatmaps.  

---

## 📂 **Project Structure**  

The directory is named **"RouteGenie"** and contains the following files:  

📁 **RouteGenie/**  
📄 **LICENSE** – Project License (MIT)  
🖼️ **Q-learning-Algorithm.jpg** – Visual representation of the Q-learning process.  
📜 **README.md** – Project documentation.  
📄 **Requirement.txt** – List of dependencies.  
📘 **RouteGenie.ipynb** – Jupyter Notebook with full implementation.  
📄 **RouteGenie.pdf** – Complete project report.  
🎥 **Test 2.0sec.mp4** – Video showing model performance.  
🎞️ **Test.gif** – GIF animation of agent’s learning behaviour.  

---

## 🛠️ **Installation & Setup**  

### **1️⃣ Clone the Repository**  
```
git clone https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPO_NAME.git
cd RouteGenie
```

### **2️⃣ Install Dependencies**
```
pip install -r Requirement.txt
```

### 🚀 How It Works
RouteGenie employs a Q-learning algorithm where the agent interacts with the environment, learns from rewards, and updates its decision-making strategy over time.

🔹 Core Algorithm (Q-Learning)

- 🌀 Algorithm Model
![Algoritham](https://github.com/ahtisham73/RouteGenie/blob/0f9fcc67599ed0a4812880a8efa4ba8a587fbb60/Q-learning-Algoritham.jpg)  

- 📌 Algorithm Code
```python
# Initialize Environment & Q-Table
env = gym.make('Taxi-v3', render_mode="rgb_array")
q_table = np.zeros((env.observation_space.n, env.action_space.n))

# Q-Update Rule
def update_q_table(state, action, reward, next_state, alpha=0.1, gamma=0.99):
    q_table[state, action] = (1 - alpha) * q_table[state, action] + \
                              alpha * (reward + gamma * np.max(q_table[next_state, :]))

```

🔹 Training Loop with Exploration & Exploitation
```python
for episode in range(10000):
    state, _ = env.reset()
    done = False

    while not done:
        action = np.argmax(q_table[state, :]) if np.random.uniform(0, 1) > 0.1 else env.action_space.sample()
        next_state, reward, terminated, truncated, _ = env.step(action)
        update_q_table(state, action, reward, next_state)
        state = next_state
        done = terminated or truncated
```

📊 Results & Performance

The trained Q-learning agent successfully navigates the environment with optimal efficiency.

Performance improvements were observed over 10,000 training episodes.

The model successfully adapts to different passenger and taxi locations.

- 📌 Performance Demo🎥

![Demo](https://github.com/ahtisham73/RouteGenie/raw/22d169ab33fba18e7ce6038bfdfc4687eadad2bf/Test.gif)



----

🏁 Conclusion

RouteGenie demonstrates how reinforcement learning can be effectively applied to autonomous decision-making. The agent successfully learns to optimize routes, and pick up, and drop off passengers while adapting to dynamic conditions. This project lays the foundation for autonomous vehicle navigation, robotic path planning, and intelligent urban mobility solutions.

---

🔗 Acknowledgments

🔹 OpenAI Gymnasium – Simulation environment 🏙️

🔹 Matplotlib & Seaborn – Visual analytics 📊

🔹 NumPy & Pandas – Data processing 📈

🔹 Pygame – Interactive environment setup 🎮


---
📜 License

This project is licensed under the MIT License. Feel free to use, modify, and distribute!

---

🚀 Contributions are welcome! 
Submit a PR or open an issue to collaborate. Let’s build smarter




