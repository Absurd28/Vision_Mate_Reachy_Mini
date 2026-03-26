# VisionMate AI: Reachy Mini Assistive Simulation

VisionMate AI is a modular, AI-powered robotics simulation environment for the **Reachy Mini** robot. It integrates real-time physics (MuJoCo), Computer Vision (OpenCV), and Natural Language Processing (DistilBERT) to execute complex assistive tasks via voice commands.

## 🚀 Features
- **Voice Control**: Real-time speech recognition with phonetic correction.
- **AI Intent Engine**: Fine-tuned DistilBERT model for classifying robotic tasks.
- **High-Fidelity Physics**: Stabilized MuJoCo simulation using the official Reachy Mini model.
- **Modern Dashboard**: CustomTkinter-based GUI with a live webcam feed.
- **Sequential Tasking**: Multi-step state machine for complex maneuvers like fetching objects.

---

## 🛠️ Installation & Requirements

### **1. Prerequisites**
- **Python 3.10+** (3.11 or 3.12 recommended)
- **Windows Long Path Support**: Enabled in Registry.
- **Microphone & Webcam**: Required for voice and vision features.

### **2. Install Dependencies**
Open your terminal and run:
```powershell
pip install -r requirements.txt
```

### **3. Prepare the AI Brain**
Since model weights are large, you must generate the data and train the model locally once:
```powershell
# Generate synthetic training data
python generate_data.py

# Fine-tune the DistilBERT model (Takes ~5-10 mins on CPU)
python train_nlp.py
```

---

## 🖥️ How to Run
Once the model is trained, launch the main application:
```powershell
python main.py
```
1. Click **"Start System"** on the dashboard.
2. Wait for the **MuJoCo 3D Viewer** to open.
3. Say **"Reachy, fetch the bottle"** or **"Reachy, move your arm"**.

---

## 📁 Project Structure
- `main.py`: The system orchestrator and entry point.
- `robot.py`: MuJoCo physics engine and task state machine.
- `voice.py`: Speech-to-text listener with phonetic correction.
- `nlp_inference.py`: DistilBERT-based intent classification.
- `ui.py`: CustomTkinter dashboard layout.
- `vision.py`: OpenCV webcam and frame processing.

---

## 📅 Roadmap
- [ ] Implement Gaze-to-3D coordinate mapping.
- [ ] Fine-tune gripper collision physics.
- [ ] Add "Active Gaze" head tracking.

## 🤝 Contributing
Updates are added daily. Feel free to fork and submit pull requests!
