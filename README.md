# Smart Face Recognition Door Lock System

An integrated, IoT-enabled access control system combining a PC-based computer vision backend, an ESP32 microcontroller for hardware actuation, and a mobile application built on Kodular for remote management and monitoring.

## 📂 Repository Structure

```
├── esp32
│   └── esp32.ino             # Arduino sketch for ESP32 hardware and relay control
├── face_recognition_pc
│   ├── access.db             # SQLite database for access logs and user records
│   ├── dataset               # Training images and face data for authorized users
│   ├── door.py               # Python script for face recognition and server logic
│   ├── labels.json           # Label mappings for recognized users
│   └── mdp.py                # Model training / processing utilities
└── kodular_android_app
    └── smart_lock_app.aia    # Kodular project source / exported app assets

```

## 🚀 Key System Components

### 1. ESP32 Microcontroller (`/esp32`)

* **File:** `esp32.ino`

* **Functionality:** Connects to your local Wi-Fi network, listens for commands, and triggers the physical relay module to open the door lock upon successful authentication.

### 2. PC Face Recognition Backend (`/face_recognition_pc`)

* **Core Scripts:** `door.py`, `mdp.py`

* **Data & Config:** `access.db` (SQLite for logging timestamps and access attempts) and `labels.json` (user ID mapping).

* **Functionality:** Captures video streams, executes the facial recognition pipeline, verifies authorized individuals, logs entries, and communicates with the ESP32 hardware.

### 3. Kodular Android App (`/kodular_android_app`)

* **File:** `smart_lock_app.aia`

* **Functionality:** A companion mobile interface allowing users to monitor access activity logs, view system status, and manually send remote unlock commands from their Android device.

## 🛠️ Getting Started & Prerequisites

### Prerequisites

* **Hardware:**

  * ESP32 Development Board (with camera module if needed)

  * Relay module and electronic door lock setup

  * PC with a webcam for running the face recognition model

* **Software:**

  * Arduino IDE (configured with ESP32 board support)

  * Python 3.x (with OpenCV, SQLite, and required machine learning libraries)

  * Kodular Creator (to import and modify `smart_lock_app.aia`)

### Installation & Setup

1. **Clone the Repository:**

   ```
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   
   ```

2. **Configure and Flash the ESP32:**

   * Open `esp32/esp32.ino` in the Arduino IDE.

   * Update your Wi-Fi SSID, password, and target server/IP configurations.

   * Select your ESP32 board and port, then upload the sketch.

3. **Set Up the PC Face Recognition Backend:**

   * Navigate to the backend directory:

     ```
     cd face_recognition_pc
     
     ```

   * Install the required Python dependencies (e.g., OpenCV, NumPy):

     ```
     pip install opencv-python numpy
     
     ```

   * Run the main Python script:

     ```
     python door.py
     
     ```

4. **Deploy the Mobile App:**

   * Import the `smart_lock_app.aia` file into [Kodular Creator](https://www.kodular.io/) to view or edit blocks, and build the APK for your Android device.

## 👥 Contributors

I thank my team mates who helped to shape the project to what it is now, for testing, and improving this project:

* **[Aryan Parashar](https://github.com/aryanparashar06-cmd)**
* **Anston Jedidah**

## 🏆 Special Recognition

* Multidisciplinary Project - Awarded S Grade (97/100) for successful project demonstration and completion.
* [LinkedIn Post for Project Demonstration](https://lnkd.in/p/gcfPY4mp)

## 📝 License

This project is open-source and available under the [MIT License](LICENSE).
