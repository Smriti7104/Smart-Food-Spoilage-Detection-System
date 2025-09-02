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
![WhatsApp Image 2025-09-02 at 18 00 26_bda49d55](https://github.com/user-attachments/assets/83423845-ffaa-471f-8fba-ffc688c95dc0)



#### Sensor + Camera Connected to Laptop
![WhatsApp Image 2025-09-02 at 17 49 53_1bf73ad5](https://github.com/user-attachments/assets/f7a3c349-3c01-4c90-864e-150eb1de5c83)


#### Mobile App Displaying Prediction
![WhatsApp Image 2025-09-02 at 17 49 52_1ad2253a](https://github.com/user-attachments/assets/5fb7b0f6-1143-4c81-a95f-d4e26be85de8)


#### Prototype with Food Samples
![WhatsApp Image 2025-07-30 at 13 08 36_a48f9737](https://github.com/user-attachments/assets/f8195202-653b-4094-b604-bfeb41214b2b)


---

## 🤖 Deep Learning Model

We trained a **Convolutional Neural Network (CNN)** to classify food as **fresh** or **rotten**.

### Training Results
- **Accuracy Graph**
![Accuracy Graph]<img width="361" height="282" alt="Screenshot 2025-07-03 232344" src="https://github.com/user-attachments/assets/9729ff85-69f4-411a-9a8a-5152984b465e" />


- **Loss Graph**
<img width="357" height="272" alt="Screenshot 2025-07-03 232422" src="https://github.com/user-attachments/assets/df0776dc-2107-49ce-888f-a8162e8c0752" />


### Sample Predictions
- Fresh Food Detection:
<img width="303" height="286" alt="Screenshot 2025-07-03 232942" src="https://github.com/user-attachments/assets/530a263f-5f1a-4164-b119-4040ea547d43" />


- Rotten Food Detection:
<img width="365" height="367" alt="Screenshot 2025-07-03 233041" src="https://github.com/user-attachments/assets/d6737f0a-2e56-4bca-83af-9d5e9a38c3a4" />


---

## 📱 Mobile App
The app provides **real-time predictions**, displaying:
- 📷 Food image captured
- ✅ Fresh / ❌ Rotten status
- 🧪 Detected gases with concentration levels

![App Display]![IMG-20250902-WA0022 1](https://github.com/user-attachments/assets/1b983424-fc12-433b-8c3e-76dacea16dea)


---

## 🛠️ Tech Stack
- **Hardware**: Raspberry Pi 4, MQ-series Gas Sensors, Pi Camera Module, Breadboard
- **Programming Languages**: Python, JavaScript
- **Frameworks & Libraries**: TensorFlow/Keras, OpenCV, Flask/React Native (for app)
- **Database**: SQLite/Firebase (for storing logs)
- **Version Control**: Git & GitHub
- **Tools**: Jupyter Notebook, VS Code

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
