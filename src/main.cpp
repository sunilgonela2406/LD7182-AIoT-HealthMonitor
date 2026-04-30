#include <Arduino.h>
#include "MAX30105.h"
#include "spo2_algorithm.h"
#include "DHT.h"
#include <Wire.h>
#include <Adafruit_SSD1306.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "../model/model.h"

// ── WiFi & Firebase credentials ──────────────────────────────────
const char* WIFI_SSID     = "YOUR_SSID";
const char* WIFI_PASSWORD = "YOUR_PASSWORD";
const char* FIREBASE_URL  = "https://your-project.firebaseio.com/alerts.json";
const char* FIREBASE_AUTH = "YOUR_DATABASE_SECRET";

// ── Pin definitions ───────────────────────────────────────────────
#define DHT_PIN     4
#define DHT_TYPE    DHT22
#define OLED_SDA    8
#define OLED_SCL    9
#define OLED_ADDR   0x3C

// ── TFLite configuration ──────────────────────────────────────────
constexpr int TENSOR_ARENA_SIZE = 32 * 1024;
uint8_t tensor_arena[TENSOR_ARENA_SIZE];

// ── Global objects ────────────────────────────────────────────────
MAX30105 particleSensor;
DHT dht(DHT_PIN, DHT_TYPE);
Adafruit_SSD1306 display(128, 64, &Wire, -1);

tflite::AllOpsResolver resolver;
const tflite::Model* model;
tflite::MicroInterpreter* interpreter;
TfLiteTensor* input;
TfLiteTensor* output;

// ── Normalisation parameters (from training set) ──────────────────
const float SPO2_MEAN = 96.2f, SPO2_STD = 2.1f;
const float HR_MEAN   = 78.5f, HR_STD   = 15.3f;
const float TEMP_MEAN = 36.8f, TEMP_STD = 0.8f;

const char* CLASS_LABELS[] = { "NORMAL", "CAUTION", "ALERT" };
int last_class = -1;

// ── Forward declarations ──────────────────────────────────────────
void initTFLite();
int runInference(float spo2, float hr, float temp);
void sendToFirebase(int cls, float spo2, float hr, float temp);
void updateDisplay(float spo2, float hr, float temp, int cls);

void setup() {
  Serial.begin(115200);
  Wire.begin(OLED_SDA, OLED_SCL);

  display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR);
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  display.println("AIoT Health Monitor");
  display.display();

  dht.begin();

  if (!particleSensor.begin(Wire, I2C_SPEED_FAST)) {
    Serial.println("ERROR: MAX30102 not found — check wiring");
    while(1);
  }
  particleSensor.setup();
  particleSensor.setPulseAmplitudeRed(0x0A);
  particleSensor.setPulseAmplitudeGreen(0);

  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) { delay(500); Serial.print("."); }
  Serial.println(" Connected: " + WiFi.localIP().toString());

  initTFLite();
  Serial.println("System ready. Starting monitoring loop...");
}

void loop() {
  // Collect 100 samples from MAX30102
  uint32_t irBuffer[100], redBuffer[100];
  int32_t spo2; int8_t spo2Valid;
  int32_t heartRate; int8_t hrValid;

  for (int i = 0; i < 100; i++) {
    while (!particleSensor.available()) particleSensor.check();
    redBuffer[i] = particleSensor.getRed();
    irBuffer[i]  = particleSensor.getIR();
    particleSensor.nextSample();
  }
  maxim_heart_rate_and_oxygen_saturation(irBuffer, 100, redBuffer,
    &spo2, &spo2Valid, &heartRate, &hrValid);

  float temperature = dht.readTemperature();

  if (!spo2Valid || !hrValid || isnan(temperature)) {
    Serial.println("WARN: Invalid reading — ensure finger on sensor");
    delay(2000);
    return;
  }

  int cls = runInference((float)spo2, (float)heartRate, temperature);
  Serial.printf("[%lu ms] SpO2: %d%%, HR: %d bpm, Temp: %.1fC -> %s\n",
    millis(), spo2, heartRate, temperature, CLASS_LABELS[cls]);

  updateDisplay((float)spo2, (float)heartRate, temperature, cls);

  // GDPR data minimisation: only upload on state change
  if (cls != last_class) {
    sendToFirebase(cls, spo2, heartRate, temperature);
    last_class = cls;
  }

  delay(5000);
}

void initTFLite() {
  model = tflite::GetModel(health_model_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.printf("ERROR: TFLite schema version mismatch (%d vs %d)\n",
      model->version(), TFLITE_SCHEMA_VERSION);
    while(1);
  }
  static tflite::MicroInterpreter static_interpreter(
    model, resolver, tensor_arena, TENSOR_ARENA_SIZE);
  interpreter = &static_interpreter;
  TfLiteStatus status = interpreter->AllocateTensors();
  if (status != kTfLiteOk) {
    Serial.println("ERROR: AllocateTensors failed — increase TENSOR_ARENA_SIZE");
    while(1);
  }
  input  = interpreter->input(0);
  output = interpreter->output(0);
  Serial.printf("TFLite loaded. Model: %d bytes, Arena: %d bytes\n",
    (int)sizeof(health_model_tflite), TENSOR_ARENA_SIZE);
}

int runInference(float spo2, float hr, float temp) {
  float f0 = (spo2 - SPO2_MEAN) / SPO2_STD;
  float f1 = (hr   - HR_MEAN)   / HR_STD;
  float f2 = (temp - TEMP_MEAN) / TEMP_STD;

  float in_scale = input->params.scale;
  int32_t in_zp  = input->params.zero_point;
  input->data.int8[0] = (int8_t)((f0 / in_scale) + in_zp);
  input->data.int8[1] = (int8_t)((f1 / in_scale) + in_zp);
  input->data.int8[2] = (int8_t)((f2 / in_scale) + in_zp);

  unsigned long t0 = millis();
  interpreter->Invoke();
  Serial.printf("  Inference: %lu ms\n", millis() - t0);

  float out_scale = output->params.scale;
  int32_t out_zp  = output->params.zero_point;
  int best = 0; float best_val = -999;
  for (int i = 0; i < 3; i++) {
    float val = (output->data.int8[i] - out_zp) * out_scale;
    Serial.printf("  P(%s) = %.3f\n", CLASS_LABELS[i], val);
    if (val > best_val) { best_val = val; best = i; }
  }
  return best;
}

void sendToFirebase(int cls, float spo2, float hr, float temp) {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WARN: WiFi disconnected, skipping Firebase upload");
    return;
  }
  HTTPClient http;
  http.begin(FIREBASE_URL);
  http.addHeader("Content-Type", "application/json");
  // Only send classified state — not raw health values (GDPR Art. 5(1)(c))
  String payload = String("{\"alert_class\":\"") + CLASS_LABELS[cls] +
    "\",\"timestamp\":" + String(millis()) + "}";
  int code = http.PUT(payload);
  Serial.printf("Firebase PUT -> HTTP %d\n", code);
  http.end();
}

void updateDisplay(float spo2, float hr, float temp, int cls) {
  display.clearDisplay();
  display.setCursor(0, 0);
  display.printf("SpO2: %.0f%%\n", spo2);
  display.printf("HR:   %.0f bpm\n", hr);
  display.printf("Temp: %.1fC\n", temp);
  display.setTextSize(1);
  display.printf(">> %s <<\n", CLASS_LABELS[cls]);
  display.display();
}
