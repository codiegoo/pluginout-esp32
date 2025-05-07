idf_component_register(
  SRCS "pluginout.cpp" "esp32-dht11.c"
  INCLUDE_DIRS "."
  PRIV_REQUIRES json esp-tflite-micro 
  REQUIRES esp_http_client esp_https_server spiffs nvs_flash esp_wifi driver
)
