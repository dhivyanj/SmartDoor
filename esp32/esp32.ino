#include "BluetoothSerial.h"
#include <Keypad.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

// --- PIN DEFINITIONS ---
#define LED_BLUE 15
#define LED_GREEN 2
#define LED_RED 4
#define BTN_SCAN 5     
#define LOCK_CONTROL 13  
#define BUZZER_PIN 21
#define RXD2 16
#define TXD2 17

BluetoothSerial SerialBT;

// Keypad Configuration
const byte ROWS = 4;
const byte COLS = 3;
char keys[ROWS][COLS] = {{'1','2','3'}, {'4','5','6'}, {'7','8','9'}, {'*','0','#'}};
byte rowPins[ROWS] = {12, 14, 27, 26}; 
byte colPins[COLS] = {25, 33, 32}; 

Keypad keypad = Keypad(makeKeymap(keys), rowPins, colPins, ROWS, COLS);
Adafruit_SSD1306 display(128, 64, &Wire, -1);

// Logic Variables
String masterPIN = "5555";
String inputPIN = "";
unsigned long lockTimer = 0;
bool isLocked = true;

// --- Helper Functions ---

void updateUI(String header, String body) {
  display.clearDisplay();
  display.setTextColor(WHITE);
  display.setTextSize(1);
  display.setCursor(0, 0);
  display.print(header);
  display.setTextSize(2);
  display.setCursor(0, 30);
  display.print(body);
  display.display();
}

void beep(int ms) {
  digitalWrite(BUZZER_PIN, HIGH);
  delay(ms);
  digitalWrite(BUZZER_PIN, LOW);
}

void handleUnlock(String reason) {
  isLocked = false;
  lockTimer = millis(); 
  updateUI("GRANTED", reason);
  digitalWrite(LED_BLUE, LOW); 
  digitalWrite(LED_GREEN, HIGH);
  digitalWrite(LOCK_CONTROL, HIGH); 
  beep(100); delay(50); beep(100);
}

void handleRelock() {
  isLocked = true;
  digitalWrite(LOCK_CONTROL, LOW);  
  digitalWrite(LED_GREEN, LOW); 
  digitalWrite(LED_BLUE, HIGH);
  updateUI("READY", "Secure");
}

void deniedAction(String reason) {
  updateUI("DENIED", reason);
  digitalWrite(LED_RED, HIGH);
  for(int i=0; i<3; i++) { 
    digitalWrite(BUZZER_PIN, HIGH); delay(100); 
    digitalWrite(BUZZER_PIN, LOW); delay(100); 
  }
  delay(1000);
  digitalWrite(LED_RED, LOW);
  digitalWrite(LED_BLUE, HIGH);
  updateUI("READY", "Secure");
}

// --- Setup ---

void setup() {
  Serial.begin(115200); 
  Serial2.begin(115200, SERIAL_8N1, RXD2, TXD2); 
  
  pinMode(LED_BLUE, OUTPUT); pinMode(LED_GREEN, OUTPUT); pinMode(LED_RED, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(LOCK_CONTROL, OUTPUT); 
  digitalWrite(LOCK_CONTROL, LOW); 
  pinMode(BTN_SCAN, INPUT_PULLUP);

  Wire.begin(23, 22, 100000);
  if(!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) Serial.println("OLED Fail");

  SerialBT.begin("SmartLock-Classic"); 
  updateUI("READY", "BT Classic");
  digitalWrite(LED_BLUE, HIGH);
  beep(200);
}

// --- Main Loop ---

void loop() {
  // 1. Auto-Relock Timer
  if (!isLocked && (millis() - lockTimer >= 5000)) {
    handleRelock();
  }

  // 2. Bluetooth Logic - Fixed to respond to '1' string
  if (SerialBT.available()) {
    char incomingChar = SerialBT.read(); 
    if (incomingChar == '1') {
      handleUnlock("APP UNLOCK");
    }
  }

  // 3. Face Scan from ESP32-CAM
  if (Serial2.available() > 0) {
    String camMsg = Serial2.readStringUntil('\n');
    camMsg.trim();
    if (camMsg == "MATCH_OK") handleUnlock("FACE MATCH");
    else if (camMsg == "MATCH_FAIL") deniedAction("UNKNOWN FACE");
  }

  // 4. Keypad Logic
  char key = keypad.getKey();
  if (key) {
    beep(50);
    if (key == '#') {
      if (inputPIN == masterPIN) {
        handleUnlock("PIN SUCCESS");
      } else {
        deniedAction("WRONG PIN");
      }
      inputPIN = "";
    } else if (key == '*') {
      inputPIN = "";
      updateUI("READY", "Secure");
    } else {
      inputPIN += key;
      String mask = "";
      for(int i=0; i<inputPIN.length(); i++) mask += "*";
      updateUI("LOCKED", mask);
    }
  }

  // 5. Manual Scan Button
  if (digitalRead(BTN_SCAN) == LOW) {
    updateUI("SCANNING", "Wait...");
    Serial2.println("SCAN");
    delay(400); 
  }
}