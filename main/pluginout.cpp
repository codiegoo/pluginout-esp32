// pluginout.cpp - Proyecto de control por voz en ESP32-S3 con ESP-IDF

// Librerías estándar y del sistema
#include <cmath>
#include <dirent.h>   // Manejo de archivos/directorios SPIFFS
#include <sys/types.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

// Librerías de FreeRTOS
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

// Logs y sistema
#include "esp_log.h"
#include "esp_system.h"

// Manejo de almacenamiento SPIFFS
#include "esp_spiffs.h"

// Manejo de WiFi y red
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_netif.h"

// Almacenamiento no volátil
#include "nvs_flash.h"
#include "nvs.h"

// TLS y HTTPS
#include "esp_tls.h"
#include "esp_crt_bundle.h"
#include "esp_http_client.h"
#include "esp_https_server.h"
#include "esp_http_server.h"

// Manejo de GPIO e I2S
#include "driver/gpio.h"
#include "driver/i2s_std.h"

// Manejo de JSON
#include "cJSON.h"

// TensorFlow Lite Micro
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Sensor DHT11 personalizado
#include "esp32-dht11.h"

// Logs para depuración
static constexpr const char *TAG_PLUGIN = "plugin";
static const char *TAG = "modelo";

// ==== DEFINICIONES DE HARDWARE ====
#define I2S_NUM I2S_NUM_0          // Canal I2S
#define SAMPLE_RATE 16000          // Frecuencia de muestreo de audio

#define LED_GPIO GPIO_NUM_10       // Pin del LED controlado
#define DHT_GPIO GPIO_NUM_11       // Pin del sensor DHT11

#define I2S_WS  GPIO_NUM_16        // WS/LRCLK
#define I2S_SD  GPIO_NUM_15        // SD/DATA del micrófono INMP441
#define I2S_SCK GPIO_NUM_17        // SCK/BCLK

// ==== CLASES DEL MODELO DE VOZ ====
#define COMANDO_APAGAR 0
#define COMANDO_ENCODER 1
#define COMANDO_HUMEDAD 2
#define PALABRA_CLAVE_PLUGIN 3
#define COMANDO_TEMPERATURA 4

// ==== CONFIGURACIONES DE TIEMPO ====
#define WIFI_CONNECT_TIMEOUT_MS 10000
#define HTTP_TIMEOUT_MS 5000
#define SCAN_TIMEOUT_MS 10000

// ==== MANEJO DE EVENTOS ====
static EventGroupHandle_t wifi_event_group;
const int WIFI_CONNECTED_BIT = BIT0;
const int WIFI_FAIL_BIT = BIT1;
static std::string ultimo_comando_aplicado = "";  // Guarda el último comando aplicado

// ==== PROTOTIPOS DE HANDLERS PARA EL SERVIDOR HTTP ====
esp_err_t guardar_get_handler(httpd_req_t *req);
esp_err_t root_get_handler(httpd_req_t *req);
esp_err_t redes_get_handler(httpd_req_t *req);
esp_err_t ip_get_handler(httpd_req_t *req);

// ==== CONFIGURACIÓN DE TENSORFLOW LITE MICRO ====
constexpr int kTensorArenaSize = 1024 * 1500;  // Memoria para el modelo
static uint8_t *tensor_arena = nullptr;

const tflite::Model* model = nullptr;   // Modelo cargado desde SPIFFS
uint8_t* modelo_data = nullptr;         // Datos crudos del modelo
size_t modelo_size = 0;

tflite::MicroInterpreter* interpreter = nullptr; // Intérprete del modelo
TfLiteTensor* input = nullptr;                   // Tensor de entrada

// ==== VARIABLES GENERALES ====
httpd_handle_t server = NULL;   // Servidor web HTTP
char redes_json[2048];          // Buffer para almacenar redes WiFi en JSON

// ==== SENSOR DHT11 ====
dht11_t dht = {
  .dht11_pin = DHT_GPIO,
  .temperature = 0.0f,
  .humidity = 0.0f
};

// ==== FUNCIÓN PARA LEER DHT11 ====
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

// ==== CONFIGURACIÓN DE I2S PARA INMP441 ====
static i2s_chan_handle_t rx_channel = NULL;

void setupI2S() {
    // Configura canal RX I2S (sólo recepción)
    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
    ESP_ERROR_CHECK(i2s_new_channel(&chan_cfg, NULL, &rx_channel)); // Solo canal de entrada

    i2s_std_config_t std_cfg = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(16000),  // 16 kHz
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(
            I2S_DATA_BIT_WIDTH_16BIT,
            I2S_SLOT_MODE_MONO
        ),
        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,     // MCLK no usado
            .bclk = I2S_SCK,             // BCK/SCK del micrófono
            .ws = I2S_WS,                // WS/LRCLK del micrófono
            .dout = I2S_GPIO_UNUSED,     // Salida no usada (solo entrada)
            .din = I2S_SD,               // Entrada de datos del micrófono
            .invert_flags = {
                .mclk_inv = false,
                .bclk_inv = false,
                .ws_inv = true           // INMP441 requiere invertir WS
            }
        }
    };

    ESP_ERROR_CHECK(i2s_channel_init_std_mode(rx_channel, &std_cfg));
    ESP_ERROR_CHECK(i2s_channel_enable(rx_channel));
}

// ---------------TFLITE--------------------

// Función para cargar el modelo TFLite desde SPIFFS
const tflite::Model* load_model_from_spiffs(const char* model_path) {
    ESP_LOGI(TAG, "Cargando modelo desde: %s", model_path);  // Log de inicio de carga del modelo
    
    FILE* file = fopen(model_path, "rb");  // Abre el archivo del modelo en modo binario
    if (!file) {  // Si no se puede abrir el archivo
        ESP_LOGE(TAG, "No se pudo abrir el archivo del modelo");  // Log de error
        return nullptr;  // Retorna un puntero nulo
    }

    fseek(file, 0, SEEK_END);  // Se mueve al final del archivo para obtener su tamaño
    size_t model_size = ftell(file);  // Obtiene el tamaño del archivo
    fseek(file, 0, SEEK_SET);  // Vuelve al principio del archivo

    uint8_t* model_data = (uint8_t*)malloc(model_size);  // Reserva memoria para almacenar el modelo
    if (!model_data) {  // Si no se puede asignar memoria
        ESP_LOGE(TAG, "Fallo al asignar memoria para el modelo");  // Log de error
        fclose(file);  // Cierra el archivo
        return nullptr;  // Retorna un puntero nulo
    }

    // Lee el contenido del archivo en el buffer de modelo
    if (fread(model_data, 1, model_size, file) != model_size) {
        ESP_LOGE(TAG, "Error al leer el modelo");  // Log de error
        free(model_data);  // Libera la memoria reservada
        fclose(file);  // Cierra el archivo
        return nullptr;  // Retorna un puntero nulo
    }

    fclose(file);  // Cierra el archivo después de leerlo

    // Obtiene el modelo TFLite desde los datos leídos
    const tflite::Model* model = tflite::GetModel(model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {  // Verifica que la versión del modelo sea compatible
        ESP_LOGE(TAG, "Versión del modelo no soportada");  // Log de error
        free(model_data);  // Libera la memoria reservada
        return nullptr;  // Retorna un puntero nulo
    }

    ESP_LOGI(TAG, "Modelo cargado correctamente. Tamaño: %d bytes", model_size);  // Log de éxito
    return model;  // Retorna el modelo cargado
}

// Función para inicializar el intérprete de TFLite
void init_tflite_interpreter() {
    ESP_LOGI(TAG, "Inicializando modelo de voz");  // Log de inicio de inicialización del modelo de voz

    // Carga el modelo desde SPIFFS
    model = load_model_from_spiffs("/spiffs/modelo_comandos.tflite");
    if (!model) {  // Si el modelo no se cargó correctamente
        ESP_LOGE(TAG, "Error al cargar el modelo TFLite");  // Log de error
        return;  // Sale de la función
    }

    // Asigna memoria para el tensor arena (memoria para tensores temporales)
    tensor_arena = (uint8_t*) heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!tensor_arena) {  // Si no se puede asignar memoria
        ESP_LOGE(TAG, "Fallo al asignar memoria para tensor arena");  // Log de error
        return;  // Sale de la función
    }

    // Configura el resuelve de operaciones de TFLite (opciones de operaciones disponibles)
    static tflite::MicroMutableOpResolver<5> resolver;
    resolver.AddFullyConnected();  // Añade la operación de capa completamente conectada
    resolver.AddSoftmax();  // Añade la operación softmax
    resolver.AddReshape();  // Añade la operación reshape
    resolver.AddConv2D();  // Añade la operación de convolución 2D
    resolver.AddMaxPool2D();  // Añade la operación de max pooling 2D

    // Configura el intérprete de TFLite
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize);
    
    interpreter = &static_interpreter;  // Asigna el intérprete

    // Asigna los tensores necesarios para la inferencia
    if (interpreter->AllocateTensors() != kTfLiteOk) {  // Si no se pueden asignar los tensores
        ESP_LOGE(TAG, "Fallo al asignar tensores");  // Log de error
        return;  // Sale de la función
    }

    input = interpreter->input(0);  // Obtiene el puntero al tensor de entrada
    ESP_LOGI(TAG, "Intérprete TFLite inicializado correctamente");  // Log de éxito
}

// Función para preprocesar los datos de audio (ajustado según el modelo)
void preprocess_audio(int16_t* audio_data, int8_t* output, size_t length) {
    for (size_t i = 0; i < length; i++) {
        float normalized = (float)audio_data[i] / 32768.0f;  // Normaliza el valor de audio a [-1.0, 1.0]
        output[i] = (int8_t)(normalized * 127.0f);            // Escala el valor normalizado a [-127, 127]
    }
}

// Función para realizar predicción con el modelo de TFLite
int predict_command(int16_t* audio_buffer, size_t buffer_size) {
    if (!interpreter || !input) {  // Verifica si el intérprete o la entrada no están inicializados
        ESP_LOGE(TAG, "Intérprete no inicializado");  // Log de error
        return -1;  // Retorna error
    }

    // Preprocesa el audio y lo copia al tensor de entrada
    int8_t* input_data = input->data.int8;
    preprocess_audio(audio_buffer, input_data, buffer_size);

    // Ejecuta la inferencia
    if (interpreter->Invoke() != kTfLiteOk) {  // Si la inferencia falla
        ESP_LOGE(TAG, "Error en la inferencia");  // Log de error
        return -1;  // Retorna error
    }

    // Obtiene los resultados de la predicción
    TfLiteTensor* output = interpreter->output(0);  // Puntero al tensor de salida
    int8_t* output_data = output->data.int8;  // Datos de la salida

    // Obtiene la escala y el punto cero del tensor de salida
    float scale = output->params.scale;
    int zero_point = output->params.zero_point;

    int predicted_class = 0;
    float max_score = (output_data[0] - zero_point) * scale;  // Calcula la puntuación máxima

    // Recorre todas las clases posibles para encontrar la clase con la puntuación más alta
    for (int i = 1; i < output->dims->data[1]; i++) {
        float score = (output_data[i] - zero_point) * scale;
        if (score > max_score) {
            max_score = score;
            predicted_class = i;
        }
    }

    ESP_LOGI(TAG, "Predicción: clase %d con score %.2f", predicted_class, max_score);  // Log del resultado de la predicción
    return predicted_class;  // Retorna la clase predicha
}

// Función para capturar el audio desde el micrófono
void capturarAudio() {
    size_t bytes_leidos = 0;
    int16_t buffer[1024];  // Buffer para almacenar los datos de audio capturados

    ESP_ERROR_CHECK(i2s_channel_read(rx_channel, buffer, sizeof(buffer), &bytes_leidos, portMAX_DELAY));  // Lee el audio del canal I2S

    if (bytes_leidos == 0) {  // Si no se ha leído audio
        ESP_LOGE(TAG, "⚠️ No se ha recibido audio. Revisa conexiones.");  // Log de advertencia
        return;  // Sale de la función
    }

    // Verifica si todos los datos leídos son ceros (silencio)
    bool es_silencio = true;
    for (size_t i = 0; i < bytes_leidos / sizeof(int16_t); i++) {
        if (buffer[i] != 0) {
            es_silencio = false;
            break;
        }
    }

    if (es_silencio) {  // Si se detecta solo silencio
        ESP_LOGW(TAG, "⚠️ El micrófono está capturando solo silencio.");  // Log de advertencia
        return;  // Sale de la función
    }

    ESP_LOGI(TAG, "✅ Audio capturado correctamente. Bytes leídos: %d", bytes_leidos);  // Log de éxito
    // Calcular el RMS (Raíz Cuadrada Media) del audio capturado
    float sum = 0;
    for (int i = 0; i < bytes_leidos / sizeof(int16_t); i++) {
        sum += buffer[i] * buffer[i];
    }
    float rms = sqrtf(sum / (bytes_leidos / sizeof(int16_t)));  // Calcula el RMS
    printf("🔊 RMS: %.2f\n", rms);  // Imprime el valor del RMS

    // Realiza la predicción para la palabra clave
    size_t num_muestras = bytes_leidos / sizeof(int16_t);
    int prediccion_palabra_clave = predict_command(buffer, num_muestras);

    // Si se detecta la palabra clave "plugin"
    if (prediccion_palabra_clave == PALABRA_CLAVE_PLUGIN) {
        ESP_LOGI(TAG, "Palabra clave 'plugin' detectada. Esperando comando...");  // Log de detección de palabra clave
        
        // Captura y procesa la siguiente palabra para el comando
        bytes_leidos = 0;  // Limpiar el buffer de audio
        ESP_ERROR_CHECK(i2s_channel_read(rx_channel, buffer, sizeof(buffer), &bytes_leidos, portMAX_DELAY));  // Lee el siguiente bloque de audio

        // Realiza la predicción para el comando
        int prediccion_comando = predict_command(buffer, num_muestras);

        // Acciona el LED según el comando
        if (prediccion_comando == COMANDO_ENCODER) {  // Comando "encender" detectado
            gpio_set_level(LED_GPIO, 1);  // Enciende el LED
            ESP_LOGI(TAG, "LED encendido");  // Log de encendido de LED
        }
        else if (prediccion_comando == COMANDO_APAGAR) {  // Comando "apagar" detectado
            gpio_set_level(LED_GPIO, 0);  // Apaga el LED
            ESP_LOGI(TAG, "LED apagado");  // Log de apagado de LED
        } else {
            ESP_LOGI(TAG, "Comando no reconocido");  // Log de comando no reconocido
        }
    } else {
        ESP_LOGI(TAG, "Palabra clave 'plugin' no detectada");  // Log si no se detectó la palabra clave
    }
}



// -----------------Redes----------------

// Maneja las solicitudes GET a la ruta raíz "/"
esp_err_t root_get_handler(httpd_req_t *req) {
    // Registro de intento de abrir el archivo config.html
    ESP_LOGI(TAG, "Intentando abrir /spiffs/config.html");
    
    // Intenta abrir el archivo HTML desde el sistema de archivos SPIFFS
    FILE* f = fopen("/spiffs/config.html", "r");
    
    // Si el archivo no se pudo abrir, enviar error 404
    if (!f) {
        ESP_LOGE(TAG, "Error al abrir archivo HTML");
        httpd_resp_send_404(req);  // Responde con código de error 404
        return ESP_FAIL;  // Retorna fallo
    }
    
    // Buffer para leer cada línea del archivo
    char line[256];
    // Establece el tipo de respuesta como "text/html"
    httpd_resp_set_type(req, "text/html");
    size_t bytes_sent = 0;  // Inicializa el contador de bytes enviados
    
    // Enviar el contenido del archivo línea por línea
    while (fgets(line, sizeof(line), f)) {
        int ret = httpd_resp_sendstr_chunk(req, line);  // Envía cada línea como un chunk
        if (ret != ESP_OK) {
            // Si ocurre un error enviando el chunk, registrar y devolver error
            ESP_LOGE(TAG, "Error al enviar chunk: %d", ret);
            fclose(f);  // Cierra el archivo
            return ESP_FAIL;  // Retorna fallo
        }
        bytes_sent += strlen(line);  // Acumula los bytes enviados
    }
    
    // Log del número de bytes enviados
    ESP_LOGI(TAG, "Archivo HTML enviado (%d bytes)", bytes_sent);
    
    // Finaliza el envío de chunks
    httpd_resp_sendstr_chunk(req, NULL);
    fclose(f);  // Cierra el archivo después de enviar
    return ESP_OK;  // Retorna éxito
}

// Handler para obtener la dirección IP del dispositivo
esp_err_t ip_get_handler(httpd_req_t *req) {
    esp_netif_ip_info_t ip_info;  // Estructura para almacenar la información IP
    esp_netif_t* netif = esp_netif_get_handle_from_ifkey("WIFI_AP_DEF");  // Obtiene el manejador de la interfaz WiFi
    
    if (netif) {
        // Si la interfaz WiFi existe, obtiene la IP
        esp_netif_get_ip_info(netif, &ip_info);
        char ip_str[16];
        snprintf(ip_str, sizeof(ip_str), IPSTR, IP2STR(&ip_info.ip));  // Convierte la IP a string
        httpd_resp_sendstr(req, ip_str);  // Envía la IP como respuesta
    } else {
        // Si no existe la interfaz, enviar IP de fallback
        httpd_resp_sendstr(req, "192.168.4.1");
    }
    return ESP_OK;  // Retorna éxito
}

// Handler para obtener la lista de redes WiFi disponibles en formato JSON
esp_err_t redes_get_handler(httpd_req_t *req) {
    httpd_resp_set_type(req, "application/json");  // Establece el tipo de respuesta como JSON
    httpd_resp_sendstr(req, redes_json);  // Envía la lista de redes en formato JSON
    return ESP_OK;  // Retorna éxito
}

// Manejador de eventos de WiFi
void wifi_event_handler(void* arg, esp_event_base_t event_base, 
                        int32_t event_id, void* event_data) {
    // Si el evento es de desconexión WiFi
    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED) {
        ESP_LOGW(TAG, "WiFi desconectado, intentando reconectar...");
        esp_wifi_connect();  // Intenta reconectar
        xEventGroupClearBits(wifi_event_group, WIFI_CONNECTED_BIT);  // Limpia el bit de conexión
        xEventGroupSetBits(wifi_event_group, WIFI_FAIL_BIT);  // Establece el bit de fallo
    } 
    // Si el evento es de obtención de IP
    else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t* event = (ip_event_got_ip_t*) event_data;  // Cast del evento para obtener IP
        ESP_LOGI(TAG, "Conectado con IP: " IPSTR, IP2STR(&event->ip_info.ip));  // Muestra la IP
        xEventGroupSetBits(wifi_event_group, WIFI_CONNECTED_BIT);  // Establece el bit de conexión exitosa
    }
}

// Inicializa el servidor web y registra los handlers
void start_web_server() {
    httpd_config_t config = HTTPD_DEFAULT_CONFIG();  // Configuración por defecto del servidor HTTP
    httpd_start(&server, &config);  // Inicia el servidor HTTP

    // Registra los handlers para las diferentes rutas
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

// Función para conectar a WiFi usando credenciales guardadas en NVS
bool conectar_a_wifi_guardado() {
    wifi_config_t sta_config = {};  // Configuración WiFi
    nvs_handle_t nvs;
    
    // Abre el espacio de almacenamiento NVS en modo solo lectura
    if (nvs_open("wifi", NVS_READONLY, &nvs) != ESP_OK) {
        ESP_LOGI(TAG, "No hay configuración WiFi guardada");
        return false;
    }

    // Lee el SSID y la contraseña guardados en NVS
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

    wifi_event_group = xEventGroupCreate();  // Crea el grupo de eventos de WiFi
    ESP_ERROR_CHECK(esp_event_handler_instance_register(WIFI_EVENT, ESP_EVENT_ANY_ID, &wifi_event_handler, NULL, NULL));  // Registra el handler de eventos WiFi
    ESP_ERROR_CHECK(esp_event_handler_instance_register(IP_EVENT, IP_EVENT_STA_GOT_IP, &wifi_event_handler, NULL, NULL));  // Registra el handler de eventos IP

    esp_netif_create_default_wifi_sta();  // Crea la interfaz WiFi
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));  // Establece el modo WiFi a STA (cliente)
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &sta_config));  // Configura la interfaz WiFi con los valores leídos
    ESP_ERROR_CHECK(esp_wifi_start());  // Inicia WiFi
    ESP_ERROR_CHECK(esp_wifi_connect());  // Conecta a la red

    // Espera hasta obtener una conexión o fallar
    EventBits_t bits = xEventGroupWaitBits(wifi_event_group, 
        WIFI_CONNECTED_BIT | WIFI_FAIL_BIT,
        pdFALSE, pdFALSE, 
        pdMS_TO_TICKS(WIFI_CONNECT_TIMEOUT_MS));

    if (bits & WIFI_CONNECTED_BIT) {
        ESP_LOGI(TAG, "Conectado exitosamente a WiFi");
        return true;  // Conexión exitosa
    } else {
        ESP_LOGW(TAG, "Falló la conexión a WiFi");
        esp_wifi_disconnect();  // Desconecta WiFi si falla
        esp_wifi_stop();  // Detiene WiFi
        return false;  // Conexión fallida
    }
}

// Configura y comienza el modo AP (punto de acceso)
void iniciar_modo_ap() {
    ESP_LOGI(TAG, "Iniciando modo AP para configuración");

    // Configuración de PMF (Protección de marco) para la red WiFi
    wifi_pmf_config_t pmf_config = {
        .capable = true,
        .required = false
    };

    wifi_config_t ap_config = {};  // Configuración para el AP
    strcpy((char *)ap_config.ap.ssid, "PluginOut");  // Establece el SSID del AP
    ap_config.ap.ssid_len = strlen("PluginOut");  // Longitud del SSID
    ap_config.ap.channel = 11;  // Canal WiFi
    ap_config.ap.authmode = WIFI_AUTH_OPEN;  // Autenticación abierta
    ap_config.ap.ssid_hidden = 0;  // Hace visible el SSID
    ap_config.ap.max_connection = 4;  // Número máximo de conexiones
    ap_config.ap.beacon_interval = 100;  // Intervalo de beacon
    ap_config.ap.pairwise_cipher = WIFI_CIPHER_TYPE_NONE;  // Sin cifrado
    ap_config.ap.ftm_responder = false;  // No responder a FTM (Prueba de distancia)
    ap_config.ap.pmf_cfg = pmf_config;  // Configuración de PMF
    ap_config.ap.sae_pwe_h2e = WPA3_SAE_PWE_BOTH;  // Configuración WPA3
    ap_config.ap.transition_disable = false;  // Habilita la transición de WPA3

    esp_netif_create_default_wifi_ap();  // Crea la interfaz de red en modo AP
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_AP));  // Establece el modo AP
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_AP, &ap_config));  // Configura el AP
    ESP_ERROR_CHECK(esp_wifi_start());  // Inicia el AP

    start_web_server();  // Inicia el servidor web
}

// Handler para guardar las configuraciones WiFi
esp_err_t guardar_get_handler(httpd_req_t *req) {
    char ssid[32], pass[64], nombre[32], query[128];
    
    // Obtiene la cadena de consulta (query string) de la URL
    if (httpd_req_get_url_query_str(req, query, sizeof(query)) == ESP_OK) {

        // Intenta obtener los parámetros de la query
        if (httpd_query_key_value(query, "ssid", ssid, sizeof(ssid)) == ESP_OK &&
            httpd_query_key_value(query, "pass", pass, sizeof(pass)) == ESP_OK &&
            httpd_query_key_value(query, "nombre", nombre, sizeof(nombre)) == ESP_OK) {

            // Abre NVS para guardar los valores
            nvs_handle_t nvs;
            nvs_open("wifi", NVS_READWRITE, &nvs);
            nvs_set_str(nvs, "ssid", ssid);  // Guarda el SSID
            nvs_set_str(nvs, "pass", pass);  // Guarda la contraseña
            nvs_set_str(nvs, "nombre", nombre);  // Guarda el nombre
            nvs_commit(nvs);  // Guarda los cambios
            nvs_close(nvs);  // Cierra el NVS

            // Responde con mensaje de éxito y reinicia el dispositivo
            httpd_resp_sendstr(req, "Guardado, reiniciando...");
            vTaskDelay(pdMS_TO_TICKS(3000));  // Espera 3 segundos
            esp_restart();  // Reinicia el dispositivo
            return ESP_OK;
        }

        // Si los parámetros no fueron obtenidos correctamente, enviar error 500
        httpd_resp_send_500(req);  // Enviar error 500
        return ESP_FAIL;  // Retornar fallo
    }

    // Si no hay query string, responder con error 400
    httpd_resp_send_400(req);  // Enviar error 400
    return ESP_FAIL;  // Retornar fallo
}


// Tarea que escanea las redes WiFi y luego elimina la tarea
void tarea_escanear_redes(void* pvParameters) {
    escanear_redes();  // Llama a la función que realiza el escaneo de redes
    vTaskDelete(NULL);  // Elimina la tarea después de que el escaneo ha terminado
}

// Obtiene el nombre del dispositivo desde el almacenamiento NVS o usa un valor por defecto
void obtener_nombre_dispositivo(char* buffer, size_t buffer_size) {
    nvs_handle_t nvs;
    strncpy(buffer, "sensorSala", buffer_size); // Asigna un valor por defecto al buffer
    
    // Intenta abrir el almacenamiento NVS en modo solo lectura
    if (nvs_open("wifi", NVS_READONLY, &nvs) == ESP_OK) {
        size_t len = buffer_size;
        esp_err_t err = nvs_get_str(nvs, "nombre", buffer, &len);  // Lee el nombre almacenado
        if (err == ESP_OK) {
            ESP_LOGI(TAG, "Nombre obtenido: %s", buffer);  // Imprime el nombre obtenido
        } else {
            ESP_LOGE(TAG, "Error al obtener el nombre desde NVS");
            strncpy(buffer, "sensorSala", buffer_size);  // Asigna valor por defecto en caso de error
        }
        nvs_close(nvs);  // Cierra el acceso a NVS
    } else {
        ESP_LOGE(TAG, "Error al abrir NVS");
    }
}

// Realiza la codificación de URL para garantizar que los caracteres sean seguros para URL
void url_encode(const char* src, char* dst, size_t dst_size) {
    const char *hex = "0123456789ABCDEF";  // Tabla de caracteres hexadecimales
    size_t i = 0, j = 0;

    while (src[i] != '\0' && j < dst_size - 1) {
        char c = src[i];
        if ((c >= 'a' && c <= 'z') ||
            (c >= 'A' && c <= 'Z') ||
            (c >= '0' && c <= '9') ||
            c == '-' || c == '_' || c == '.' || c == '~') {
            // Si el carácter es seguro, se copia tal cual
            dst[j++] = c;
        } else if (j + 3 < dst_size) {
            // Si el carácter no es seguro, se codifica como %XX
            dst[j++] = '%';
            dst[j++] = hex[(c >> 4) & 0xF];
            dst[j++] = hex[c & 0xF];
        } else {
            break; // Si no hay espacio suficiente, termina
        }
        i++;
    }

    dst[j] = '\0';  // Asegura la terminación de la cadena
}

// Reporta los datos de temperatura y humedad al servidor
void reportar_datos() {
    ESP_LOGI(TAG, "Entrando a reportar_datos...");

    wifi_mode_t modo;
    esp_wifi_get_mode(&modo);  // Obtiene el modo actual de WiFi
    ESP_LOGI(TAG, "Modo WiFi actual: %d", modo);  // Imprime el modo WiFi

    if (modo == WIFI_MODE_STA) {  // Si estamos en modo STA (conectado a una red)
        ESP_LOGI(TAG, "Modo STA detectado, procesando datos...");

        char nombre[64];
        obtener_nombre_dispositivo(nombre, sizeof(nombre));  // Obtiene el nombre del dispositivo

        ESP_LOGI(TAG, "Nombre almacenado en NVS: %s", nombre);

        float temperatura, humedad;
        if (read_dht(&temperatura, &humedad) != ESP_OK) {  // Lee los valores de temperatura y humedad
            ESP_LOGE(TAG, "Error leyendo sensor DHT");
            return;
        }

        char post_data[256];
        const char* tipo = "sensor";  // Tipo de dispositivo (sensor)

        // Crea un JSON con los datos obtenidos
        snprintf(post_data, sizeof(post_data),
            "{\"name\":\"%s\",\"temperature\":%.2f,\"humidity\":%.2f,\"type\":\"%s\"}",
            nombre, temperatura, humedad, tipo);

        ESP_LOGI(TAG, "Datos JSON: %s", post_data);

        esp_http_client_config_t config = {
            .url = "https://plugin-out.vercel.app/api/reportar",  // URL del servidor
            .crt_bundle_attach = esp_crt_bundle_attach,  // Certificado para HTTPS
        };

        ESP_LOGI(TAG, "Configurando cliente HTTP para enviar datos...");
        esp_http_client_handle_t client = esp_http_client_init(&config);
        esp_http_client_set_method(client, HTTP_METHOD_POST);  // Método POST
        esp_http_client_set_header(client, "Content-Type", "application/json");  // Establece el tipo de contenido
        esp_http_client_set_post_field(client, post_data, strlen(post_data));  // Agrega el cuerpo de la solicitud

        esp_err_t err = esp_http_client_perform(client);  // Realiza la solicitud HTTP
        if (err == ESP_OK) {
            ESP_LOGI(TAG, "Datos reportados. Código: %d", esp_http_client_get_status_code(client));  // Imprime el código de estado
        } else {
            ESP_LOGE(TAG, "Error HTTP POST: %s", esp_err_to_name(err));  // Imprime error en caso de fallo
        }

        esp_http_client_cleanup(client);  // Limpia el cliente HTTP
    } else {
        ESP_LOGW(TAG, "No estamos en modo STA, no se reportan datos.");  // Aviso si no estamos en modo STA
    }
}

// Verifica el comando recibido desde el servidor y actúa en consecuencia
void verificar_comando() {
    ESP_LOGI(TAG, "Entrando a verificar_comando...");

    wifi_mode_t modo;
    esp_wifi_get_mode(&modo);  // Obtiene el modo actual de WiFi

    if (modo != WIFI_MODE_STA) {  // Si no estamos en modo STA, omite la verificación
        ESP_LOGW(TAG, "WiFi no está en modo STA, se omite verificación");
        return;
    }

    ESP_LOGI(TAG, "Modo STA detectado, consultando comando...");

    char nombre[64];
    obtener_nombre_dispositivo(nombre, sizeof(nombre));  // Obtiene el nombre del dispositivo
    ESP_LOGI(TAG, "Nombre del dispositivo: %s", nombre);

    char nombre_codificado[128];
    url_encode(nombre, nombre_codificado, sizeof(nombre_codificado));  // Codifica el nombre

    char url[256];
    snprintf(url, sizeof(url), "https://plugin-out.vercel.app/api/comando?nombre=%s", nombre_codificado);  // Crea la URL de la solicitud
    ESP_LOGI(TAG, "URL construida correctamente: %s", url);

    esp_http_client_config_t config = {
        .url = url,  // Configura la URL de la solicitud
        .crt_bundle_attach = esp_crt_bundle_attach,  // Certificado para HTTPS
    };

    esp_http_client_handle_t client = esp_http_client_init(&config);
    if (client == NULL) {  // Verifica si se pudo inicializar el cliente
        ESP_LOGE(TAG, "No se pudo inicializar el cliente HTTP");
        return;
    }

    esp_http_client_set_method(client, HTTP_METHOD_GET);  // Establece el método GET
    esp_http_client_set_header(client, "Accept-Encoding", "identity");  // Establece encabezados

    // Abre la conexión para leer la respuesta del servidor
    esp_err_t err = esp_http_client_open(client, 0);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Error al abrir la conexión: %s", esp_err_to_name(err));  // Error al abrir conexión
        esp_http_client_cleanup(client);  // Limpia el cliente HTTP
        return;
    }

    // Lee los encabezados de la respuesta
    int64_t content_length = esp_http_client_fetch_headers(client);
    if (content_length < 0) {
        ESP_LOGE(TAG, "Error al obtener encabezados: %lld", content_length);  // Error al obtener encabezados
        esp_http_client_close(client);  // Cierra la conexión
        esp_http_client_cleanup(client);  // Limpia el cliente HTTP
        return;
    }

    ESP_LOGI(TAG, "Content-Length recibido: %lld", content_length);  // Imprime el tamaño del contenido

    // Lee el cuerpo de la respuesta
    char buffer[512] = {0};
    int bytes_leidos = esp_http_client_read(client, buffer, sizeof(buffer) - 1);
    if (bytes_leidos >= 0) {
        buffer[bytes_leidos] = '\0';  // Asegura la terminación nula

        // Elimina saltos de línea del buffer
        for (int i = 0; i < bytes_leidos; i++) {
            if (buffer[i] == '\r' || buffer[i] == '\n') {
                buffer[i] = '\0';
                break;
            }
        }

        ESP_LOGI(TAG, "Comando recibido (texto plano): '%s'", buffer);

        // Compara el comando recibido y realiza la acción correspondiente
        if (strcmp(buffer, "encender") == 0) {
            ESP_LOGI(TAG, "Encendiendo LED...");
            gpio_set_level(LED_GPIO, 1);  // Enciende el LED
        } else if (strcmp(buffer, "apagar") == 0) {
            ESP_LOGI(TAG, "Apagando LED...");
            gpio_set_level(LED_GPIO, 0);  // Apaga el LED
        } else {
            ESP_LOGW(TAG, "Comando desconocido: %s", buffer);  // Comando desconocido
        }
    } else {
        ESP_LOGE(TAG, "Error al leer cuerpo de la respuesta: %s", esp_err_to_name(bytes_leidos));  // Error al leer respuesta
    }

    esp_http_client_close(client);  // Cierra la conexión
    esp_http_client_cleanup(client);  // Limpia el cliente HTTP
}

void task_reportar(void *pvParameters) {
    // Imprime la cantidad de stack disponible
    ESP_LOGI(TAG, " Stack disponible: %d bytes", uxTaskGetStackHighWaterMark(NULL));

    // Espera hasta que el dispositivo esté conectado a la red WiFi
    while (esp_netif_get_handle_from_ifkey("WIFI_STA_DEF") == NULL ||
            !esp_netif_is_netif_up(esp_netif_get_handle_from_ifkey("WIFI_STA_DEF"))) {
        ESP_LOGI(TAG, " Esperando conexión WiFi en task_reportar...");
        vTaskDelay(pdMS_TO_TICKS(1000));  // Espera 1 segundo antes de volver a comprobar
    }

    // Una vez conectado a WiFi, imprime mensaje y comienza la tarea de reporte
    ESP_LOGI(TAG, " WiFi conectado. Iniciando tarea reportar...");

    // Tarea continua: ejecuta reportar_datos cada 1.2 segundos
    while (1) {
        reportar_datos();  // Función que se encarga de reportar los datos
        vTaskDelay(pdMS_TO_TICKS(1200));  // Espera 1.2 segundos
    }
}

void task_verificar(void *pvParameters) {
    // Imprime la cantidad de stack disponible
    ESP_LOGI(TAG, " Stack disponible: %d bytes", uxTaskGetStackHighWaterMark(NULL));

    // Espera hasta que el dispositivo esté conectado a la red WiFi
    while (esp_netif_get_handle_from_ifkey("WIFI_STA_DEF") == NULL ||
            !esp_netif_is_netif_up(esp_netif_get_handle_from_ifkey("WIFI_STA_DEF"))) {
        ESP_LOGI(TAG, "⏳ Esperando conexión WiFi en task_verificar...");
        vTaskDelay(pdMS_TO_TICKS(1000));  // Espera 1 segundo antes de volver a comprobar
    }

    // Una vez conectado a WiFi, imprime mensaje y comienza la tarea de verificación
    ESP_LOGI(TAG, " WiFi conectado. Iniciando tarea verificar...");

    // Tarea continua: ejecuta verificar_comando cada 1 segundo
    while (1) {
        verificar_comando();  // Función que se encarga de verificar comandos
        vTaskDelay(pdMS_TO_TICKS(1000));  // Espera 1 segundo
    }
}

void task_escuchar(void *pvParameters) {
    // Inicia la tarea de escucha por voz
    ESP_LOGI(TAG, "🎤 Iniciando tarea de escucha por voz...");
    while (1) {
        capturarAudio();  // Función que contiene la inferencia de voz
        vTaskDelay(pdMS_TO_TICKS(500)); // Espera 500ms entre detecciones
    }
}

void task_probar_microfono(void *param) {
    int16_t buffer[1024];  // Buffer para almacenar los datos del micrófono
    size_t bytes_read;
    
    // Tarea continua para probar el micrófono
    while (1) {
        i2s_channel_read(rx_channel, buffer, sizeof(buffer), &bytes_read, portMAX_DELAY);  // Lee datos del micrófono
        if (bytes_read > 0) {
            int max_val = 0;
            // Procesa los datos para encontrar el valor máximo de audio
            for (int i = 0; i < bytes_read / 2; i++) {
                buffer[i] = buffer[i] / 16;  // Reduce la amplitud del valor de audio
                int val = abs(buffer[i]);
                if (val > max_val) max_val = val;  // Guarda el máximo valor encontrado
            }
            ESP_LOGI("MIC_TEST", "Nivel de audio (normalizado): %d", max_val);  // Imprime el nivel de audio
            
            // Debug: Imprime las primeras 10 muestras de audio en formato hexadecimal
            ESP_LOG_BUFFER_HEX_LEVEL("Audio RAW", buffer, 20, ESP_LOG_INFO);
        }
        vTaskDelay(pdMS_TO_TICKS(500));  // Espera 500ms antes de la siguiente lectura
    }
}

extern "C" void app_main() {
    
    // 1. Inicialización básica del sistema
    ESP_LOGI(TAG, "Inicializando NVS");
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        // Si hay un error con el NVS, lo borra y lo vuelve a inicializar
        ESP_ERROR_CHECK(nvs_flash_erase());
        ESP_ERROR_CHECK(nvs_flash_init());
    }

    ESP_ERROR_CHECK(esp_netif_init());  // Inicializa la interfaz de red
    ESP_ERROR_CHECK(esp_event_loop_create_default());  // Crea el bucle de eventos predeterminado

    // 2. Configuración inicial de NVS para WiFi
    nvs_handle_t nvs;
    if (nvs_open("wifi", NVS_READWRITE, &nvs) == ESP_OK) {
        size_t len = 0;
        if (nvs_get_str(nvs, "nombre", NULL, &len) != ESP_OK) {
            // Si no existe el nombre de la red, lo configura
            nvs_set_str(nvs, "nombre", "sensorSala");
            nvs_commit(nvs);  // Guarda los cambios
        }
        nvs_close(nvs);  // Cierra el manejador NVS
    }

    // 3. Sistema de archivos SPIFFS
    ESP_LOGI(TAG, "Montando SPIFFS");
    esp_vfs_spiffs_conf_t spiffs_conf = {
        .base_path = "/spiffs",  // Ruta de montaje
        .partition_label = NULL,  // Partición predeterminada
        .max_files = 5,  // Número máximo de archivos
        .format_if_mount_failed = true  // Formatea si la montura falla
    };
    ESP_ERROR_CHECK(esp_vfs_spiffs_register(&spiffs_conf));  // Registra SPIFFS

    // Verificación de archivos SPIFFS
    DIR* dir = opendir("/spiffs");  // Abre el directorio SPIFFS
    if (dir) {
        struct dirent* entry;
        while ((entry = readdir(dir)) != NULL) {
            ESP_LOGI(TAG, "Archivo encontrado: %s", entry->d_name);  // Imprime los archivos encontrados
        }
        closedir(dir);  // Cierra el directorio
    } else {
        ESP_LOGE(TAG, "No se pudo abrir directorio SPIFFS");
    }

    // 4. Inicializa I2S para capturar datos del micrófono
    ESP_LOGI(TAG, "Inicializando I2S para el micrófono");
    setupI2S();
    init_tflite_interpreter();  // Inicializa el intérprete de TensorFlow Lite

    // 5. Configuración WiFi
    ESP_LOGI(TAG, "Inicializando WiFi");
    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();  // Configuración predeterminada de WiFi
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));  // Inicializa el WiFi

    // 6. Intento de conexión a red guardada
    bool conectado = conectar_a_wifi_guardado();  // Intenta conectarse a una red WiFi guardada

    if (!conectado) {
        ESP_LOGI(TAG, "🛜 No se pudo conectar → Modo AP");
        iniciar_modo_ap();  // Si no se conecta, inicia el modo AP

        // 7. Escanear redes WiFi solo en modo AP
        ESP_LOGI(TAG, "Iniciando escaneo de redes...");
        xTaskCreate(tarea_escanear_redes, "escanear_redes", 4096, NULL, 5, NULL);  // Crea la tarea de escaneo de redes
    }

    // 8. Configuración de GPIO
    ESP_LOGI(TAG, "Configurando GPIO");
    gpio_config_t io_conf = {
        .pin_bit_mask = (1ULL << LED_GPIO) | (1ULL << DHT_GPIO),  // Configura los pines GPIO
        .mode = GPIO_MODE_OUTPUT,
        .pull_up_en = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE
    };
    ESP_ERROR_CHECK(gpio_config(&io_conf));  // Configura los pines GPIO

    // 9. Crear las tareas
    xTaskCreate(task_escuchar, "Task Escuchar", 8192, NULL, 2, NULL);  // Tarea de escucha por voz

    if (conectado) {
        xTaskCreate(task_reportar, "Task Reportar", 8192, NULL, 1, NULL);  // Tarea de reporte de datos
    } else {
        ESP_LOGI(TAG, "WiFi no conectado, no se inicia task_reportar");
    }
    
    xTaskCreate(task_verificar, "Task Verificar", 8192, NULL, 1, NULL);  // Tarea de verificación de comandos
    xTaskCreate(task_probar_microfono, "Probar Microfono", 8192, NULL, 1, NULL);  // Tarea de prueba de micrófono
}
