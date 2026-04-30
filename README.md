# LD7182 AIoT Health Monitor — ESP32-S3

## Project Overview
AI-powered health monitoring system using ESP32-S3, MAX30102, DHT22, TFLite Micro, and Firebase.

## Repository Structure
```
LD7182_Code/
├── src/
│   └── main.cpp              # Main ESP32-S3 firmware (C++/Arduino)
├── model/
│   └── model.h               # TFLite INT8 model header (replace with Edge Impulse export)
├── scripts/
│   └── generate_dataset.py   # Synthetic dataset generator for Edge Impulse
├── platformio.ini            # PlatformIO build configuration
└── README.md                 # This file
```

## Setup Instructions
See Appendix B of the technical report for full setup instructions.

### Quick Start
1. Install PlatformIO IDE in VS Code
2. Run `python scripts/generate_dataset.py` to generate training data
3. Upload CSV to Edge Impulse and train the model
4. Export INT8 TFLite model and paste into `model/model.h`
5. Update WiFi/Firebase credentials in `src/main.cpp`
6. Build and flash with PlatformIO

## Hardware
- ESP32-S3-DevKitC-1 N8R8
- MAX30102 Pulse Oximeter (I2C: SDA→GPIO8, SCL→GPIO9)
- DHT22 Temperature Sensor (GPIO4)
- SSD1306 0.96" OLED Display (I2C shared bus, addr 0x3C)

## Dependencies
See `platformio.ini` for library dependencies.

## Module: LD7182 – AI for IoT | Northumbria University
