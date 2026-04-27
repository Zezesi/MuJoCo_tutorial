# MuJoCo_tutorial
This tutorial will guide you to set up MuJoCo and run your python projects on your Macbook(Mine is M3).

---

### 1️⃣ Create a directory for your python project
```sh
mkdir MuJoCo_py_project
```

### 2️⃣ Change to the diectory
```sh
cd MuJoCo_py_project
```

### 3️⃣ Install your python package
```sh
python3 -m pip install --user -U pip
```

### 4️⃣ Create a virtual environment
```sh
python3.13 -m venv mujoco_env
```
**Note:** Remember to use the right python version to create the virtual environment.

### 5️⃣ Activate the virtual environment
```sh
source mujoco_env/bin/activate
```

### 6️⃣ Install the required modules
```sh
pip install mujoco numpy simple_pid
```

### 7️⃣ git clone this repository
```sh
git clone https://github.com/Zezesi/MuJoCo_tutorial.git
```

### 8️⃣ Run the script
```sh
mjpython x2.py
```
**Note:** You could also download other models from MuJoCo model gallery and implement your own control algorithms.

### Screenshot of the project
<img width="1512" height="982" alt="Screenshot 2026-04-26 at 14 32 38" src="https://github.com/user-attachments/assets/5f1f73d1-b3f7-46b3-bf64-923a98aeaf63" />

---
# MuJoCo Gymnasium API
### 1️⃣ Humanoid
<img width="1512" height="982" alt="Screenshot 2026-04-27 at 00 55 43" src="https://github.com/user-attachments/assets/c95231f7-cdd1-4552-95ab-572044dcb369" />

### 2️⃣ Reacher
<img width="1512" height="982" alt="Screenshot 2026-04-27 at 00 56 58" src="https://github.com/user-attachments/assets/ef7c8db7-731f-4656-916e-4b77f323ff91" />
---
# To Do
Implement control algorithms(MPC, RL) on more complex models.

