// W11 (XIAO ESP32S3 Sense clone) camera -> JPEG frames over native USB serial.
//
// Every frame on the wire, all integers little-endian:
//
//   offset  size  field
//   0       4     magic      0xA5 0x5A 0xCA 0xFE
//   4       4     length     uint32, number of JPEG bytes
//   8       4     timestamp  uint32, millis() when the frame was grabbed
//   12      len   jpeg       JPEG data (starts with FF D8)
//   12+len  4     crc32      uint32, CRC-32 over length + timestamp + jpeg
//                            (same as Python zlib.crc32(buf[4:12+len]))
//
// Nothing else is written to Serial. The ROM bootloader may still print text
// after a reset, and a frame can be cut short if the host stops reading, so
// the reader should scan for the magic, check that length <= MAX_FRAME_BYTES,
// and drop any frame whose CRC does not match.

#include <Arduino.h>
#include "esp_camera.h"
#include "esp_log.h"
#include "camera_pins.h"

// ---- Settings ---------------------------------------------------------------
static const framesize_t FRAME_SIZE = FRAMESIZE_VGA;  // FRAMESIZE_QVGA for 320x240
static const int JPEG_QUALITY = 12;                   // 0-63, lower is better quality
static const uint32_t MIN_FRAME_INTERVAL_MS = 0;      // 0 = as fast as possible; 100 = 10 fps cap
static const uint32_t MAX_FRAME_BYTES = 512 * 1024;   // frames larger than this are skipped
static const int MAX_CAPTURE_FAILURES = 20;           // restart after this many in a row
static const uint32_t SERIAL_TX_BUFFER = 16 * 1024;

static const uint8_t MAGIC[4] = {0xA5, 0x5A, 0xCA, 0xFE};

// ---- CRC-32 (IEEE 802.3, matches zlib) -------------------------------------
static uint32_t crcTable[256];

static void crcInit() {
  for (uint32_t i = 0; i < 256; i++) {
    uint32_t c = i;
    for (int k = 0; k < 8; k++) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
    crcTable[i] = c;
  }
}

// Start with crc = 0; feed the previous result back in to continue.
static uint32_t crcUpdate(uint32_t crc, const uint8_t *data, size_t len) {
  crc = ~crc;
  while (len--) crc = crcTable[(crc ^ *data++) & 0xFF] ^ (crc >> 8);
  return ~crc;
}

// ---- Serial output ----------------------------------------------------------
static void putU32(uint8_t *p, uint32_t v) {
  p[0] = v;
  p[1] = v >> 8;
  p[2] = v >> 16;
  p[3] = v >> 24;
}

// Returns false if the host stopped reading partway through.
static bool writeAll(const uint8_t *data, size_t len) {
  while (len > 0) {
    size_t n = Serial.write(data, len);
    if (n == 0) return false;
    data += n;
    len -= n;
  }
  return true;
}

// Returns false if the frame was cut short; the reader drops it on the CRC.
static bool sendFrame(const camera_fb_t *fb, uint32_t timestamp) {
  uint8_t header[12];
  memcpy(header, MAGIC, 4);
  putU32(header + 4, fb->len);
  putU32(header + 8, timestamp);

  uint32_t crc = crcUpdate(0, header + 4, 8);
  crc = crcUpdate(crc, fb->buf, fb->len);
  uint8_t trailer[4];
  putU32(trailer, crc);

  return writeAll(header, sizeof(header)) && writeAll(fb->buf, fb->len) &&
         writeAll(trailer, sizeof(trailer));
}

// ---- Camera -----------------------------------------------------------------
static bool cameraInit() {
  camera_config_t config = {};
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size = FRAME_SIZE;
  config.jpeg_quality = JPEG_QUALITY;
  config.fb_count = 2;
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.grab_mode = CAMERA_GRAB_LATEST;  // always hand out the newest frame

  if (esp_camera_init(&config) != ESP_OK) return false;

  sensor_t *s = esp_camera_sensor_get();
  if (s && s->id.PID == OV3660_PID) {
    // Same corrections CameraWebServer applies to the OV3660.
    s->set_vflip(s, 1);
    s->set_brightness(s, 1);
    s->set_saturation(s, -2);
  }
  return true;
}

// ---- Main -------------------------------------------------------------------
void setup() {
  // Keep log text off the binary stream.
  esp_log_level_set("*", ESP_LOG_NONE);

  Serial.setTxBufferSize(SERIAL_TX_BUFFER);  // must come before begin()
  Serial.begin(115200);                      // baud is ignored on native USB
  Serial.setDebugOutput(false);

  crcInit();

  if (!psramFound() || !cameraInit()) {
    // Without PSRAM or a camera there is nothing useful to do; retry from reset.
    delay(2000);
    ESP.restart();
  }
}

void loop() {
  static uint32_t lastFrameMs = 0;
  static int failures = 0;

  // Don't capture while no host has the port open.
  if (!Serial) {
    delay(50);
    return;
  }

  if (MIN_FRAME_INTERVAL_MS > 0) {
    uint32_t elapsed = millis() - lastFrameMs;
    if (elapsed < MIN_FRAME_INTERVAL_MS) delay(MIN_FRAME_INTERVAL_MS - elapsed);
  }

  camera_fb_t *fb = esp_camera_fb_get();
  lastFrameMs = millis();
  if (!fb) {
    if (++failures >= MAX_CAPTURE_FAILURES) ESP.restart();
    delay(10);
    return;
  }
  failures = 0;

  if (fb->format == PIXFORMAT_JPEG && fb->len > 0 && fb->len <= MAX_FRAME_BYTES) {
    sendFrame(fb, lastFrameMs);
  }
  esp_camera_fb_return(fb);
}
