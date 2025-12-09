#include <WiFi.h>

// ---------- CONFIGURE THESE ----------
const char* ssid = "DIR-825-8A4B";
const char* password = "123456789";
const char* server_ip = "192.168.0.127";   // your laptop’s IP address
const uint16_t port = 5050;
// -------------------------------------

// ===== Input pins =====
#define ONOFF_SWITCH_PIN 15
#define PUSH1_PIN        4
#define PUSH2_PIN        5

// ===== Output LED pins =====
#define LED_PUSH_IND     13     // lights when push buttons pressed
#define LED_CONN_STATUS  12    // ON = Wi-Fi+server connected
#define LED_ON_STATE     14    // ON/OFF;-switch = ON
#define LED_OFF_STATE    27    // ON/OFF-switch = OFF

WiFiClient client;

// Remember last states to send messages only on changes
int last_onoff = HIGH;
int last_p1 = HIGH;
int last_p2 = HIGH;

// ----------------------------
void setSwitchLEDs(int onoff) {
  // Always reflect switch condition
  if (onoff == LOW) {           // switch ON
    digitalWrite(LED_ON_STATE, HIGH);
    digitalWrite(LED_OFF_STATE, LOW);
  } else {                      // switch OFF
    digitalWrite(LED_ON_STATE, LOW);
    digitalWrite(LED_OFF_STATE, HIGH);
  }
}
// ----------------------------
void connectWiFi() {
  Serial.print("Connecting to Wi-Fi");
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(400);
    Serial.print(".");
  }
  Serial.println("\n✅ Wi-Fi connected");
}
// ----------------------------
void connectServer() {
  while (!client.connected()) {
    Serial.print("Connecting to server...");
    if (client.connect(server_ip, port)) {
      Serial.println("✅ Connected to server!");
      digitalWrite(LED_CONN_STATUS, HIGH);
    } else {
      Serial.println("❌ Failed, retrying...");
      digitalWrite(LED_CONN_STATUS, LOW);
      delay(1000);
    }
  }
}
// ----------------------------
void setup() {
  Serial.begin(115200);

  pinMode(ONOFF_SWITCH_PIN, INPUT_PULLUP);
  pinMode(PUSH1_PIN, INPUT_PULLUP);
  pinMode(PUSH2_PIN, INPUT_PULLUP);

  pinMode(LED_PUSH_IND, OUTPUT);
  pinMode(LED_CONN_STATUS, OUTPUT);
  pinMode(LED_ON_STATE, OUTPUT);
  pinMode(LED_OFF_STATE, OUTPUT);

  digitalWrite(LED_CONN_STATUS, LOW);
  digitalWrite(LED_PUSH_IND, LOW);
  digitalWrite(LED_ON_STATE, LOW);
  digitalWrite(LED_OFF_STATE, LOW);

  connectWiFi();
  connectServer();
}
// ----------------------------
void loop() {
  // --- check Wi-Fi status ---
  if (WiFi.status() != WL_CONNECTED) {
    digitalWrite(LED_CONN_STATUS, LOW);
    connectWiFi();
  }

  // --- check server connection ---
  if (!client.connected()) {
    digitalWrite(LED_CONN_STATUS, LOW);
    connectServer();
  } else {
    digitalWrite(LED_CONN_STATUS, HIGH);
  }

  // --- read inputs ---
  int onoff = digitalRead(ONOFF_SWITCH_PIN);
  int p1 = digitalRead(PUSH1_PIN);
  int p2 = digitalRead(PUSH2_PIN);

  // --- Always update switch LEDs ---
  setSwitchLEDs(onoff);

  // --- Send ON/OFF switch change only once per change ---
  if (onoff != last_onoff) {
    String msg = "ONOFF_SWITCH=" + String(onoff == LOW ? "ON" : "OFF") + "\n";
    if (client.connected()) {
      client.print(msg);
      Serial.print("Sent: "); Serial.println(msg);
    }
    last_onoff = onoff;
  }

  // --- push buttons (send once per press) ---
  if (p1 == LOW && last_p1 == HIGH) {
    if (client.connected()) client.print("PUSH1=PRESSED\n");
    Serial.println("Sent: PUSH1=PRESSED");
  }
  if (p2 == LOW && last_p2 == HIGH) {
    if (client.connected()) client.print("PUSH2=PRESSED\n");
    Serial.println("Sent: PUSH2=PRESSED");
  }

  // --- LED behaviour ---
  // LED 2 turns ON while any push button is pressed
  if (p1 == LOW || p2 == LOW)
    digitalWrite(LED_PUSH_IND, HIGH);
  else
    digitalWrite(LED_PUSH_IND, LOW);

  // update previous states
  last_p1 = p1;
  last_p2 = p2;

  delay(40);  // debounce
}