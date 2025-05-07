// pluginout.cpp - Proyecto de control por voz en ESP32-S3 con ESP-IDF

#include <cmath>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_system.h"
#include "esp_spiffs.h"
#include "esp_wifi.h"
#include "nvs_flash.h"
#include "nvs.h"
#include "esp_event.h"
#include "esp_tls.h"
#include "esp_http_client.h"
#include "esp_https_server.h"
#include "driver/gpio.h"
#include "driver/i2s_std.h"
#include "cJSON.h"
#include "esp_netif.h"
#include "esp_err.h"
#include "esp_crt_bundle.h"
#include "esp_http_server.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "esp32-dht11.h"
#include <dirent.h>   // Para opendir, readdir, closedir
#include <sys/types.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

static constexpr const char *TAG_PLUGIN = "plugin";
static const char *TAG = "modelo";  




// Definiciones de hardware
#define LED_GPIO GPIO_NUM_10
#define DHT_GPIO GPIO_NUM_8
#define I2S_WS  GPIO_NUM_16
#define I2S_SD  GPIO_NUM_15
#define I2S_SCK GPIO_NUM_17


// Configuraciones de WiFi y otros
#define WIFI_CONNECT_TIMEOUT_MS 10000
#define HTTP_TIMEOUT_MS 5000
#define SCAN_TIMEOUT_MS 10000

static EventGroupHandle_t wifi_event_group;
static std::string ultimo_comando_aplicado = "";
const int WIFI_CONNECTED_BIT = BIT0;
const int WIFI_FAIL_BIT = BIT1;


esp_err_t guardar_get_handler(httpd_req_t *req);
esp_err_t root_get_handler(httpd_req_t *req);
esp_err_t redes_get_handler(httpd_req_t *req);
esp_err_t ip_get_handler(httpd_req_t *req);




// Tamaño del tensor y arena
constexpr int kTensorArenaSize = 1024 * 1500;
static uint8_t *tensor_arena = nullptr;


const tflite::Model* model = nullptr;   
uint8_t* modelo_data = nullptr;
size_t modelo_size = 0;

// Variables TensorFlow Lite
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;

httpd_handle_t server = NULL;
char redes_json[2048];

dht11_t dht = {
  .dht11_pin = DHT_GPIO,
  .temperature = 0.0f,
  .humidity = 0.0f
};



esp_err_t read_dht(float* temp, float* hum) {
    if (dht11_read(&dht, 2) == 0) {
        *temp = dht.temperature;
        *hum = dht.humidity;
        return ESP_OK;
    } else {
        ESP_LOGW(TAG, "❌ Error leyendo DHT11");
        return ESP_FAIL;
    }
}


i2s_chan_handle_t rx_channel;

void setupI2S() {
    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
    ESP_ERROR_CHECK(i2s_new_channel(&chan_cfg, &rx_channel, NULL));

    i2s_std_config_t std_cfg = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(16000),
        .slot_cfg = I2S_STD_MSB_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO),
        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,
            .bclk = I2S_SCK,
            .ws = I2S_WS,
            .dout = I2S_GPIO_UNUSED,
            .din = I2S_SD,
            .invert_flags = {
                .mclk_inv = false,
                .bclk_inv = false,
                .ws_inv = false,
            }
        }
    };

    ESP_ERROR_CHECK(i2s_channel_init_std_mode(rx_channel, &std_cfg));
    ESP_ERROR_CHECK(i2s_channel_enable(rx_channel));
}





// ---------------TFLITE--------------------
const tflite::Model* load_model_from_spiffs(const char* model_path) {
    ESP_LOGI(TAG, "Cargando modelo desde: %s", model_path);
    
    FILE* file = fopen(model_path, "rb");
    if (!file) {
        ESP_LOGE(TAG, "No se pudo abrir el archivo del modelo");
        return nullptr;
    }

    fseek(file, 0, SEEK_END);
    size_t model_size = ftell(file);
    fseek(file, 0, SEEK_SET);

    uint8_t* model_data = (uint8_t*)malloc(model_size);
    if (!model_data) {
        ESP_LOGE(TAG, "Fallo al asignar memoria para el modelo");
        fclose(file);
        return nullptr;
    }

    if (fread(model_data, 1, model_size, file) != model_size) {
        ESP_LOGE(TAG, "Error al leer el modelo");
        free(model_data);
        fclose(file);
        return nullptr;
    }

    fclose(file);

    const tflite::Model* model = tflite::GetModel(model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Versión del modelo no soportada");
        free(model_data);
        return nullptr;
    }

    ESP_LOGI(TAG, "Modelo cargado correctamente. Tamaño: %d bytes", model_size);
    return model;
}

void init_tflite_interpreter() {
    // 8. Inicialización del modelo de voz
    ESP_LOGI(TAG, "Inicializando modelo de voz");

    // Cargar modelo desde SPIFFS
    model = load_model_from_spiffs("/spiffs/modelo_comandos.tflite");
    if (!model) {
        ESP_LOGE(TAG, "Error al cargar el modelo TFLite");
        return;
    }

    // Asignar memoria para el tensor arena
    tensor_arena = (uint8_t*) heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!tensor_arena) {
        ESP_LOGE(TAG, "Fallo al asignar memoria para tensor arena");
        return;
    }

    // Configurar el resolvedor de operaciones
    static tflite::MicroMutableOpResolver<5> resolver;
    resolver.AddFullyConnected();
    resolver.AddSoftmax();
    resolver.AddReshape();
    resolver.AddConv2D();
    resolver.AddMaxPool2D();

    // Configurar el intérprete
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize);
    
    interpreter = &static_interpreter;

    // Asignar tensores
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(TAG, "Fallo al asignar tensores");
        return;
    }

    // Obtener puntero al tensor de entrada
    input = interpreter->input(0);
    ESP_LOGI(TAG, "Intérprete TFLite inicializado correctamente");



    // 9. Bucle principal
    ESP_LOGI(TAG, "Iniciando bucle principal");

    while (true) {
        // Inferencia de voz
        for (int i = 0; i < input->bytes; ++i) {
            input->data.uint8[i] = 0;
        }

        if (interpreter->Invoke() != kTfLiteOk) {
            ESP_LOGE(TAG, "Error al invocar modelo");
        } else {
            TfLiteTensor* output = interpreter->output(0);
            uint8_t max_score = 0;
            int command_index = -1;
            for (int i = 0; i < output->dims->data[1]; ++i) {
                if (output->data.uint8[i] > max_score) {
                    max_score = output->data.uint8[i];
                    command_index = i;
                }
            }

            if (command_index == 1) gpio_set_level(LED_GPIO, 1);
            else if (command_index == 2) gpio_set_level(LED_GPIO, 0);
        }

        vTaskDelay(pdMS_TO_TICKS(10));
    }
    
}

// Preprocesamiento del audio (ajusta según las necesidades de tu modelo)
void preprocess_audio(int16_t* audio_data, float* output, size_t length) {
    for (size_t i = 0; i < length; i++) {
        // Normalización simple (-1.0 a 1.0)
        output[i] = static_cast<float>(audio_data[i]) / 32768.0f;
    }
}

// Realizar predicción con el modelo
int predict_command(int16_t* audio_buffer, size_t buffer_size) {
    if (!interpreter || !input) {
        ESP_LOGE(TAG, "Intérprete no inicializado");
        return -1;
    }

    // Preprocesar audio y copiar al tensor de entrada
    float* input_data = input->data.f;
    preprocess_audio(audio_buffer, input_data, buffer_size);

    // Ejecutar inferencia
    if (interpreter->Invoke() != kTfLiteOk) {
        ESP_LOGE(TAG, "Error en la inferencia");
        return -1;
    }

    // Obtener resultados
    TfLiteTensor* output = interpreter->output(0);
    float* output_data = output->data.f;
    int predicted_class = 0;
    float max_score = output_data[0];

    // Encontrar la clase con mayor probabilidad
    for (int i = 1; i < output->dims->data[1]; i++) {
        if (output_data[i] > max_score) {
            max_score = output_data[i];
            predicted_class = i;
        }
    }

    ESP_LOGI(TAG, "Predicción: clase %d con score %.2f", predicted_class, max_score);
    return predicted_class;
}


void capturarAudio() {
    size_t bytes_leidos = 0;
    int16_t buffer[1024];

    ESP_ERROR_CHECK(i2s_channel_read(rx_channel, buffer, sizeof(buffer), &bytes_leidos, portMAX_DELAY));

    if (bytes_leidos == 0) {
        ESP_LOGE(TAG, "⚠️ No se ha recibido audio. Revisa conexiones.");
        return;
    }

    // Verifica si todos los datos son ceros
    bool es_silencio = true;
    for (size_t i = 0; i < sizeof(buffer) / sizeof(buffer[0]); i++) {
        if (buffer[i] != 0) {
            es_silencio = false;
            break;
        }
    }

    if (es_silencio) {
        ESP_LOGW(TAG, "⚠️ El micrófono está capturando solo silencio.");
    } else {
        ESP_LOGI(TAG, "✅ Audio capturado correctamente. Bytes leídos: %d", bytes_leidos);
    }
}



// -----------------Redes----------------





esp_err_t root_get_handler(httpd_req_t *req) {
    ESP_LOGI(TAG, "Intentando abrir /spiffs/config.html");
    FILE* f = fopen("/spiffs/config.html", "r");
    
    if (!f) {
        ESP_LOGE(TAG, "Error al abrir archivo HTML");
        httpd_resp_send_404(req);
        return ESP_FAIL;
    }
    
    char line[256];
    httpd_resp_set_type(req, "text/html");
    size_t bytes_sent = 0;
    
    while (fgets(line, sizeof(line), f)) {
        int ret = httpd_resp_sendstr_chunk(req, line);
        if (ret != ESP_OK) {
            ESP_LOGE(TAG, "Error al enviar chunk: %d", ret);
            fclose(f);
            return ESP_FAIL;
        }
        bytes_sent += strlen(line);
    };
    
    ESP_LOGI(TAG, "Archivo HTML enviado (%d bytes)", bytes_sent);
    httpd_resp_sendstr_chunk(req, NULL);
    fclose(f);
    return ESP_OK;
}

// Handler separado para la IP
esp_err_t ip_get_handler(httpd_req_t *req) {
    esp_netif_ip_info_t ip_info;
    esp_netif_t* netif = esp_netif_get_handle_from_ifkey("WIFI_AP_DEF");
    
    if (netif) {
        esp_netif_get_ip_info(netif, &ip_info);
        char ip_str[16];
        snprintf(ip_str, sizeof(ip_str), IPSTR, IP2STR(&ip_info.ip));
        httpd_resp_sendstr(req, ip_str);
    } else {
        httpd_resp_sendstr(req, "192.168.4.1"); // Fallback
    }
    return ESP_OK;
}

esp_err_t redes_get_handler(httpd_req_t *req) {
    httpd_resp_set_type(req, "application/json");
    httpd_resp_sendstr(req, redes_json);
    return ESP_OK;
}

void wifi_event_handler(void* arg, esp_event_base_t event_base, 
                        int32_t event_id, void* event_data) {
    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED) {
        ESP_LOGW(TAG, "WiFi desconectado, intentando reconectar...");
        esp_wifi_connect();
        xEventGroupClearBits(wifi_event_group, WIFI_CONNECTED_BIT);
        xEventGroupSetBits(wifi_event_group, WIFI_FAIL_BIT);
    } else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t* event = (ip_event_got_ip_t*) event_data;
        ESP_LOGI(TAG, "Conectado con IP: " IPSTR, IP2STR(&event->ip_info.ip));
        xEventGroupSetBits(wifi_event_group, WIFI_CONNECTED_BIT);
    }
}



void start_web_server() {
    httpd_config_t config = HTTPD_DEFAULT_CONFIG();
    httpd_start(&server, &config);

    httpd_uri_t root_uri = { .uri = "/", .method = HTTP_GET, .handler = root_get_handler, .user_ctx = NULL };
    httpd_register_uri_handler(server, &root_uri);

    httpd_uri_t redes_uri = { .uri = "/redes", .method = HTTP_GET, .handler = redes_get_handler, .user_ctx = NULL };
    httpd_register_uri_handler(server, &redes_uri);

    httpd_uri_t guardar_uri = { .uri = "/guardar", .method = HTTP_GET, .handler = guardar_get_handler, .user_ctx = NULL };
    httpd_register_uri_handler(server, &guardar_uri);

    httpd_uri_t ip_uri = {
        .uri = "/ip",
        .method = HTTP_GET,
        .handler = ip_get_handler,
        .user_ctx = NULL
    };
    httpd_register_uri_handler(server, &ip_uri);
}


bool conectar_a_wifi_guardado() {
    wifi_config_t sta_config = {};
    nvs_handle_t nvs;
    
    if (nvs_open("wifi", NVS_READONLY, &nvs) != ESP_OK) {
        ESP_LOGI(TAG, "No hay configuración WiFi guardada");
        return false;
    }

    size_t lenS = sizeof(sta_config.sta.ssid);
    size_t lenP = sizeof(sta_config.sta.password);
    
    if (nvs_get_str(nvs, "ssid", (char*)sta_config.sta.ssid, &lenS) != ESP_OK ||
        nvs_get_str(nvs, "pass", (char*)sta_config.sta.password, &lenP) != ESP_OK) {
        nvs_close(nvs);
        ESP_LOGI(TAG, "Credenciales WiFi incompletas");
        return false;
    }
    nvs_close(nvs);

    ESP_LOGI(TAG, "Intentando conectar a red guardada: %s", sta_config.sta.ssid);

    wifi_event_group = xEventGroupCreate();
    ESP_ERROR_CHECK(esp_event_handler_instance_register(WIFI_EVENT, ESP_EVENT_ANY_ID, &wifi_event_handler, NULL, NULL));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(IP_EVENT, IP_EVENT_STA_GOT_IP, &wifi_event_handler, NULL, NULL));

    esp_netif_create_default_wifi_sta();
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &sta_config));
    ESP_ERROR_CHECK(esp_wifi_start());
    ESP_ERROR_CHECK(esp_wifi_connect());

    EventBits_t bits = xEventGroupWaitBits(wifi_event_group, 
        WIFI_CONNECTED_BIT | WIFI_FAIL_BIT,
        pdFALSE, pdFALSE, 
        pdMS_TO_TICKS(WIFI_CONNECT_TIMEOUT_MS));

    if (bits & WIFI_CONNECTED_BIT) {
        ESP_LOGI(TAG, "Conectado exitosamente a WiFi");
        return true;
    } else {
        ESP_LOGW(TAG, "Falló la conexión a WiFi");
        esp_wifi_disconnect();
        esp_wifi_stop();
        return false;
    }
}

void iniciar_modo_ap() {
    ESP_LOGI(TAG, "Iniciando modo AP para configuración");

    wifi_pmf_config_t pmf_config = {
        .capable = true,
        .required = false
    };

    wifi_config_t ap_config = {};
    strcpy((char *)ap_config.ap.ssid, "PluginOut");  // Usamos strcpy para copiar el SSID
    ap_config.ap.ssid_len = strlen("PluginOut");
    ap_config.ap.channel = 11;
    ap_config.ap.authmode = WIFI_AUTH_OPEN;
    ap_config.ap.ssid_hidden = 0;
    ap_config.ap.max_connection = 4;
    ap_config.ap.beacon_interval = 100;
    ap_config.ap.pairwise_cipher = WIFI_CIPHER_TYPE_NONE;
    ap_config.ap.ftm_responder = false;
    ap_config.ap.pmf_cfg = pmf_config;
    ap_config.ap.sae_pwe_h2e = WPA3_SAE_PWE_BOTH;
    ap_config.ap.transition_disable = false;

    esp_netif_create_default_wifi_ap();
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_AP));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_AP, &ap_config));
    ESP_ERROR_CHECK(esp_wifi_start());

    start_web_server();
}


esp_err_t guardar_get_handler(httpd_req_t *req) {
    char ssid[32], pass[64], nombre[32], query[128];
    
    // Obtener la cadena de consulta de la URL
    if (httpd_req_get_url_query_str(req, query, sizeof(query)) == ESP_OK) {

        // Intentar obtener los valores de los parámetros de la query
        if (httpd_query_key_value(query, "ssid", ssid, sizeof(ssid)) == ESP_OK &&
            httpd_query_key_value(query, "pass", pass, sizeof(pass)) == ESP_OK &&
            httpd_query_key_value(query, "nombre", nombre, sizeof(nombre)) == ESP_OK) {

            // Abrir NVS para guardar los valores
            nvs_handle_t nvs;
            nvs_open("wifi", NVS_READWRITE, &nvs);
            nvs_set_str(nvs, "ssid", ssid);
            nvs_set_str(nvs, "pass", pass);
            nvs_set_str(nvs, "nombre", nombre);
            nvs_commit(nvs);
            nvs_close(nvs);

            // Responder con mensaje de éxito y reiniciar
            httpd_resp_sendstr(req, "Guardado, reiniciando...");
            vTaskDelay(pdMS_TO_TICKS(3000));
            esp_restart();
            return ESP_OK;
        }

        // Si no se pudieron obtener los parámetros correctamente
        httpd_resp_send_500(req); // Enviar error interno del servidor
        return ESP_FAIL;
    }

    // Si no se recibió la query
    httpd_resp_send_500(req); // Enviar error interno del servidor
    return ESP_FAIL;  // Asegurarse de retornar un valor de error
}

void escanear_redes() {
    wifi_mode_t current_mode;
    esp_wifi_get_mode(&current_mode);

    bool modo_temporal = false;

    if (current_mode == WIFI_MODE_AP) {
        ESP_LOGW(TAG, " Cambiando temporalmente a modo APSTA para escanear");
        ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_APSTA));
        modo_temporal = true;
    }

    wifi_scan_config_t scan_config = {};
    scan_config.ssid = nullptr;               // No filtra por SSID
    scan_config.bssid = nullptr;              // No filtra por BSSID
    scan_config.channel = 0;                  // Escanea TODOS los canales
    scan_config.show_hidden = true;           // Muestra redes ocultas
    scan_config.scan_type = WIFI_SCAN_TYPE_ACTIVE;
    scan_config.scan_time.active.min = 100;   // Tiempo mínimo de escaneo
    scan_config.scan_time.active.max = 300;   // Tiempo máximo de escaneo


    wifi_ap_record_t ap_info[20];
    uint16_t num = 20;

    ESP_LOGI(TAG, "🔍 Iniciando escaneo de redes...");

    esp_err_t scan_err = esp_wifi_scan_start(&scan_config, true);
    if (scan_err == ESP_OK) {
        if (esp_wifi_scan_get_ap_records(&num, ap_info) == ESP_OK) {
            strcpy(redes_json, "[");
            for (int i = 0; i < num; i++) {
                if (strlen((char*)ap_info[i].ssid) == 0) continue;
                if (strlen(redes_json) > 1) strcat(redes_json, ",");
                strcat(redes_json, "\"");
                strcat(redes_json, (char *)ap_info[i].ssid);
                strcat(redes_json, "\"");
            }
            strcat(redes_json, "]");
            ESP_LOGI(TAG, " Redes encontradas: %s", redes_json);
        } else {
            ESP_LOGW(TAG, "No se pudieron obtener registros de AP.");
            strcpy(redes_json, "[]");
        }
    } else {
        ESP_LOGE(TAG, " Error al iniciar escaneo WiFi.");
        strcpy(redes_json, "[]");
    }

    if (modo_temporal) {
        ESP_LOGW(TAG, " Restaurando modo AP");
        ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_AP));
    }
}



void tarea_escanear_redes(void* pvParameters) {
    escanear_redes();
    vTaskDelete(NULL);  // Termina la tarea después del escaneo
}






void obtener_nombre_dispositivo(char* buffer, size_t buffer_size) {
    nvs_handle_t nvs;
    strncpy(buffer, "sensorSala", buffer_size); // Valor por defecto
    
    if (nvs_open("wifi", NVS_READONLY, &nvs) == ESP_OK) {
        size_t len = buffer_size;
        esp_err_t err = nvs_get_str(nvs, "nombre", buffer, &len);
        if (err == ESP_OK) {
            ESP_LOGI(TAG, "Nombre obtenido: %s", buffer);  // Usamos buffer, no nombre
        } else {
            ESP_LOGE(TAG, "Error al obtener el nombre desde NVS");
            strncpy(buffer, "sensorSala", buffer_size);  // Valor por defecto en caso de error
        }
        nvs_close(nvs);
    } else {
        ESP_LOGE(TAG, "Error al abrir NVS");
    }
}

void url_encode(const char* src, char* dst, size_t dst_size) {
    const char *hex = "0123456789ABCDEF";
    size_t i = 0, j = 0;

    while (src[i] != '\0' && j < dst_size - 1) {
        char c = src[i];
        if ((c >= 'a' && c <= 'z') ||
            (c >= 'A' && c <= 'Z') ||
            (c >= '0' && c <= '9') ||
            c == '-' || c == '_' || c == '.' || c == '~') {
            // Carácter seguro, copiar tal cual
            dst[j++] = c;
        } else if (j + 3 < dst_size) {
            // Codificar como %XX
            dst[j++] = '%';
            dst[j++] = hex[(c >> 4) & 0xF];
            dst[j++] = hex[c & 0xF];
        } else {
            break; // No hay espacio suficiente
        }
        i++;
    }

    dst[j] = '\0';
}




void reportar_datos() {
    ESP_LOGI(TAG, "Entrando a reportar_datos...");

    wifi_mode_t modo;
    esp_wifi_get_mode(&modo);
    ESP_LOGI(TAG, "Modo WiFi actual: %d", modo);  // Agrega esto para ver el modo

    if (modo == WIFI_MODE_STA) {
        ESP_LOGI(TAG, "Modo STA detectado, procesando datos...");

        char nombre[64];
        obtener_nombre_dispositivo(nombre, sizeof(nombre));

        ESP_LOGI(TAG, "Nombre almacenado en NVS: %s", nombre);

        float temperatura, humedad;

        if (read_dht(&temperatura, &humedad) != ESP_OK) {
            ESP_LOGE(TAG, "Error leyendo sensor DHT");
            return;
        }

        char post_data[256];
        const char* tipo = "sensor";


        snprintf(post_data, sizeof(post_data),
            "{\"name\":\"%s\",\"temperature\":%.2f,\"humidity\":%.2f,\"type\":\"%s\"}",
            nombre, temperatura, humedad, tipo);


        ESP_LOGI(TAG, "Datos JSON: %s", post_data);

        esp_http_client_config_t config = {
            .url = "https://plugin-out.vercel.app/api/reportar",
            .crt_bundle_attach = esp_crt_bundle_attach,
        };    

        ESP_LOGI(TAG, "Configurando cliente HTTP para enviar datos...");
        esp_http_client_handle_t client = esp_http_client_init(&config);
        esp_http_client_set_method(client, HTTP_METHOD_POST);
        esp_http_client_set_header(client, "Content-Type", "application/json");
        esp_http_client_set_post_field(client, post_data, strlen(post_data));

        esp_err_t err = esp_http_client_perform(client);
        if (err == ESP_OK) {
            ESP_LOGI(TAG, "Datos reportados. Código: %d", esp_http_client_get_status_code(client));
        } else {
            ESP_LOGE(TAG, "Error HTTP POST: %s", esp_err_to_name(err));
        }

        esp_http_client_cleanup(client);
    } else {
        ESP_LOGW(TAG, "No estamos en modo STA, no se reportan datos.");
    }
}


void verificar_comando() {
    ESP_LOGI(TAG, "Entrando a verificar_comando...");

    wifi_mode_t modo;
    esp_wifi_get_mode(&modo);

    if (modo != WIFI_MODE_STA) {
        ESP_LOGW(TAG, "WiFi no está en modo STA, se omite verificación");
        return;
    }

    ESP_LOGI(TAG, "Modo STA detectado, consultando comando...");

    char nombre[64];
    obtener_nombre_dispositivo(nombre, sizeof(nombre));
    ESP_LOGI(TAG, "Nombre del dispositivo: %s", nombre);

    char nombre_codificado[128];
    url_encode(nombre, nombre_codificado, sizeof(nombre_codificado));

    char url[256];
    snprintf(url, sizeof(url), "https://plugin-out.vercel.app/api/comando?nombre=%s", nombre_codificado);
    ESP_LOGI(TAG, "URL construida correctamente: %s", url);

    esp_http_client_config_t config = {
        .url = url,
        .crt_bundle_attach = esp_crt_bundle_attach,
    };

    esp_http_client_handle_t client = esp_http_client_init(&config);
    if (client == NULL) {
        ESP_LOGE(TAG, "No se pudo inicializar el cliente HTTP");
        return;
    }

    esp_http_client_set_method(client, HTTP_METHOD_GET);
    esp_http_client_set_header(client, "Accept-Encoding", "identity");

    // Abrir la conexión (modo lectura: write_len = 0)
    esp_err_t err = esp_http_client_open(client, 0);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Error al abrir la conexión: %s", esp_err_to_name(err));
        esp_http_client_cleanup(client);
        return;
    }

    // Leer encabezados
    int64_t content_length = esp_http_client_fetch_headers(client);
    if (content_length < 0) {
        ESP_LOGE(TAG, "Error al obtener encabezados: %lld", content_length);
        esp_http_client_close(client);
        esp_http_client_cleanup(client);
        return;
    }

    ESP_LOGI(TAG, "Content-Length recibido: %lld", content_length);

    // Leer el cuerpo
    char buffer[512] = {0};
    int bytes_leidos = esp_http_client_read(client, buffer, sizeof(buffer) - 1);
    if (bytes_leidos >= 0) {
        buffer[bytes_leidos] = '\0';  // Asegurar terminación nula

        // Limpiar saltos de línea por si los hay
        for (int i = 0; i < bytes_leidos; i++) {
            if (buffer[i] == '\r' || buffer[i] == '\n') {
                buffer[i] = '\0';
                break;
            }
        }

        ESP_LOGI(TAG, "Comando recibido (texto plano): '%s'", buffer);

        if (strcmp(buffer, "encender") == 0) {
            ESP_LOGI(TAG, "Encendiendo LED...");
            gpio_set_level(LED_GPIO, 1);
        } else if (strcmp(buffer, "apagar") == 0) {
            ESP_LOGI(TAG, "Apagando LED...");
            gpio_set_level(LED_GPIO, 0);
        } else {
            ESP_LOGW(TAG, "Comando desconocido: %s", buffer);
        }
    } else {
        ESP_LOGE(TAG, "Error al leer cuerpo de la respuesta: %s", esp_err_to_name(bytes_leidos));
    }

    esp_http_client_close(client);
    esp_http_client_cleanup(client);
}



void task_reportar(void *pvParameters) {
    ESP_LOGI(TAG, " Stack disponible: %d bytes", uxTaskGetStackHighWaterMark(NULL));

    // Esperar hasta que haya conexión a WiFi con IP
    while (esp_netif_get_handle_from_ifkey("WIFI_STA_DEF") == NULL ||
            !esp_netif_is_netif_up(esp_netif_get_handle_from_ifkey("WIFI_STA_DEF"))) {
        ESP_LOGI(TAG, " Esperando conexión WiFi en task_reportar...");
        vTaskDelay(pdMS_TO_TICKS(1000));
    }

    ESP_LOGI(TAG, " WiFi conectado. Iniciando tarea reportar...");

    while (1) {
        reportar_datos();
        vTaskDelay(pdMS_TO_TICKS(1200));
    }
}

void task_verificar(void *pvParameters) {
    ESP_LOGI(TAG, " Stack disponible: %d bytes", uxTaskGetStackHighWaterMark(NULL));

    while (esp_netif_get_handle_from_ifkey("WIFI_STA_DEF") == NULL ||
            !esp_netif_is_netif_up(esp_netif_get_handle_from_ifkey("WIFI_STA_DEF"))) {
        ESP_LOGI(TAG, "⏳ Esperando conexión WiFi en task_verificar...");
        vTaskDelay(pdMS_TO_TICKS(1000));
    }

    ESP_LOGI(TAG, " WiFi conectado. Iniciando tarea verificar...");

    while (1) {
        verificar_comando();
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}


extern "C" void app_main() {
    // 1. Inicialización básica del sistema
    ESP_LOGI(TAG, "Inicializando NVS");
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ESP_ERROR_CHECK(nvs_flash_init());
    }

    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());

    // 2. Configuración inicial de NVS para WiFi
    nvs_handle_t nvs;
    if (nvs_open("wifi", NVS_READWRITE, &nvs) == ESP_OK) {
        size_t len = 0;
        if (nvs_get_str(nvs, "nombre", NULL, &len) != ESP_OK) {
            nvs_set_str(nvs, "nombre", "sensorSala");
            nvs_commit(nvs);
        }
        nvs_close(nvs);
    }

    // 3. Sistema de archivos SPIFFS
    ESP_LOGI(TAG, "Montando SPIFFS");
    esp_vfs_spiffs_conf_t spiffs_conf = {
        .base_path = "/spiffs",
        .partition_label = NULL,
        .max_files = 5,
        .format_if_mount_failed = true
    };
    ESP_ERROR_CHECK(esp_vfs_spiffs_register(&spiffs_conf));

    // Verificación de archivos SPIFFS
    DIR* dir = opendir("/spiffs");
    if (dir) {
        struct dirent* entry;
        while ((entry = readdir(dir)) != NULL) {
            ESP_LOGI(TAG, "Archivo encontrado: %s", entry->d_name);
        }
        closedir(dir);
    } else {
        ESP_LOGE(TAG, "No se pudo abrir directorio SPIFFS");
    }

    // 4. Configuración WiFi
    ESP_LOGI(TAG, "Inicializando WiFi");
    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));


     // 5. Intento de conexión a red guardada
    bool conectado = conectar_a_wifi_guardado();

    if (!conectado) {
        ESP_LOGI(TAG, "🛜 No se pudo conectar → Modo AP");
        iniciar_modo_ap();

        // 6. Escanear redes solo en modo AP
        ESP_LOGI(TAG, "Iniciando escaneo de redes...");
        xTaskCreate(tarea_escanear_redes, "escanear_redes", 4096, NULL, 5, NULL);
    }

    // 7. Configuración de hardware
    ESP_LOGI(TAG, "Configurando GPIO");
    gpio_config_t io_conf = {
        .pin_bit_mask = (1ULL << LED_GPIO) | (1ULL << DHT_GPIO),
        .mode = GPIO_MODE_OUTPUT,
        .pull_up_en = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE
    };
    ESP_ERROR_CHECK(gpio_config(&io_conf));



    if (conectado) {
        xTaskCreate(task_reportar, "Task Reportar", 8192, NULL, 1, NULL);
    } else {
        ESP_LOGI(TAG, "WiFi no conectado, no se inicia task_reportar");
    }

    xTaskCreate(task_verificar, "Task Verificar", 8192, NULL, 1, NULL);




}


