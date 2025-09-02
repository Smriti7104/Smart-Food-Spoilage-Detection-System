# 🥗 Smart Food Spoilage Detection System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![RaspberryPi](https://img.shields.io/badge/Hardware-Raspberry%20Pi-red)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-success)

A **real-time AI-powered system** that detects whether stored food is **fresh or spoiled** using **gas sensors** and **camera analysis**. It displays the results on a **mobile app**, showing detected gases and spoilage status, helping to reduce food waste and ensure food safety.

---

## 📌 Features
- 📷 **Camera-based detection**: Identifies spoilage signs via image classification.
- 🧪 **Gas sensor monitoring**: Detects harmful gases like ammonia and hydrogen sulfide.
- 📱 **Mobile app integration**: Displays spoilage status and detected gases in real time.
- ⏱️ **Real-time updates**: Continuous monitoring with instant notifications.
- 🌍 **Food waste reduction**: Early detection to prevent unnecessary waste.

---

## ⚙️ System Architecture

The system integrates **hardware + software + deep learning**:
1. **Raspberry Pi** – Central controller.
2. **Gas Sensors** – Detect gases emitted by food.
3. **Camera Module** – Captures images of food.
4. **Deep Learning Model** – Classifies food as fresh or spoiled.
5. **Mobile App** – Displays results to the user.

---

## 🔧 Hardware Setup

### Components Used
- Raspberry Pi 4
- Gas Sensor (MQ-series)
- Pi Camera Module
- Breadboard & jumper wires

### Prototype Images

#### Raspberry Pi + Gas Sensor Setup
![Hardware Setup](file-4hLdf9STrrYs1AjfJHcoiR)

#### Sensor + Camera Connected to Laptop
![Laptop Integration](file-Eu4nxaZ5TCt8XjbQJkH1aM)

#### Mobile App Displaying Prediction
![App Output](file-4Q5pBCKVzu3YpuipSRJHDG)

#### Prototype with Food Samples
![Prototype Demo](file-JGyTmiG6bLBexAkAcghhqR)

---

## 🤖 Deep Learning Model

We trained a **Convolutional Neural Network (CNN)** to classify food as **fresh** or **rotten**.

### Training Results
- **Accuracy Graph**
![Accuracy Graph](file-8a7zosbMvf4DX8UDqHzBcB)

- **Loss Graph**
![Loss Graph](file-8MeqQUV2E4k6rTnyyE4tR2)

### Sample Predictions
- Fresh Food Detection:
![Fresh Prediction](file-87jgAzMBH9i2Q2T43HtZxq)

- Rotten Food Detection:
![Rotten Prediction](file-BFuYWC6w8sR4V9aBuHRJrU)

---

## 📱 Mobile App
The app provides **real-time predictions**, displaying:
- 📷 Food image captured
- ✅ Fresh / ❌ Rotten status
- 🧪 Detected gases with concentration levels

![App Display](file-4Q5pBCKVzu3YpuipSRJHDG)

---

## 🛠️ Tech Stack
- **Hardware**: Raspberry Pi 4, MQ-series Gas Sensors, Pi Camera Module, Breadboard
- **Programming Languages**: Python, JavaScript
- **Frameworks & Libraries**: TensorFlow/Keras, OpenCV, Flask/React Native (for app)
- **Database**: SQLite/Firebase (for storing logs)
- **Version Control**: Git & GitHub
- **Tools**: Jupyter Notebook, VS Code

---

## 🚀 Installation & Usage

### Prerequisites
- Raspberry Pi 4 (with Raspbian OS)
- Python 3.8+
- Installed libraries: `tensorflow`, `keras`, `opencv`, `flask`, `requests`

### Steps
1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/smart-food-spoilage-detection.git
   cd smart-food-spoilage-detection
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Deep Learning Model Training (Optional)**
   ```bash
   jupyter notebook DL_MODEL.ipynb
   ```

4. **Start the Backend Server**
   ```bash
   python app/server.py
   ```

5. **Launch Mobile App**
   - Navigate to the `app/` folder
   - Run using Expo/React Native CLI
   ```bash
   npm install
   npm start
   ```

6. **Connect Hardware**
   - Attach MQ sensor + Pi Camera to Raspberry Pi.
   - Place food items and run detection.

---

## 🛠️ How It Works
1. **Food placed in fridge** → System starts monitoring.
2. **Gas sensors** detect emission levels.
3. **Camera** captures periodic food images.
4. **Deep Learning Model** processes data.
5. **Mobile App** displays freshness status + gases detected.

---

## 🚀 Future Scope
- IoT cloud integration for remote monitoring.
- Expanding dataset for more food types.
- Integration with smart refrigerators.
- Push notifications for expiry alerts.

---

## 📂 Repository Contents
- `DL_MODEL.ipynb` → Deep Learning Model code
- `hardware/` → Raspberry Pi + sensor integration
- `app/` → Mobile application code
- `results/` → Accuracy, Loss graphs & predictions

---

## 💡 Conclusion
The **Smart Food Spoilage Detection System** combines **AI + IoT** to monitor food freshness, detect spoilage early, and reduce waste. This project demonstrates the integration of **hardware sensors, deep learning, and mobile applications** into a practical solution for households.

---

🔥 *Contributions, feedback, and improvements are welcome!*
