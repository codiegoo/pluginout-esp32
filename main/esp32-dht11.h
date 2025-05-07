#ifndef _DHT_11
#define _DHT_11

#ifdef __cplusplus
extern "C" {
#endif

#include <driver/gpio.h>
#include <stdio.h>
#include <string.h>
#include <rom/ets_sys.h>
#include "esp_log.h"

/**
 * @brief Structure containing readings and info about the DHT11
 */
typedef struct {
    int dht11_pin;       // GPIO pin connected to DHT11
    float temperature;   // Last temperature reading
    float humidity;      // Last humidity reading
} dht11_t;

/**
 * @brief Wait on pin until it reaches the specified state
 * 
 * @param dht11 DHT11 sensor struct
 * @param state State to wait for (0 or 1)
 * @param timeout Timeout in microseconds
 * @return Time waited or -1 on timeout
 */
int wait_for_state(dht11_t dht11, int state, int timeout);

/**
 * @brief Holds the pin low for the specified duration
 * 
 * @param dht11 DHT11 sensor struct
 * @param hold_time_us Time to hold the pin low in microseconds
 */
void hold_low(dht11_t dht11, int hold_time_us);

/**
 * @brief Reads temperature and humidity values from the DHT11 sensor
 * 
 * @note This function is blocking; wait at least 2 seconds between reads.
 * 
 * @param dht11 Pointer to DHT11 sensor struct
 * @param connection_timeout Number of attempts before timeout
 * @return 0 on success, -1 on failure
 */
int dht11_read(dht11_t *dht11, int connection_timeout);

#ifdef __cplusplus
}
#endif

#endif // _DHT_11
