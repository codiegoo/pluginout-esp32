#include "esp32-dht11.h"

static const char* TAG = "DHT11";

#ifdef __cplusplus
extern "C" {
#endif

int wait_for_state(dht11_t dht11, int state, int timeout)
{
    gpio_set_direction(dht11.dht11_pin, GPIO_MODE_INPUT);
    int count = 0;

    while (gpio_get_level(dht11.dht11_pin) != state) {
        if (count >= timeout) return -1;
        count += 2;
        ets_delay_us(2);
    }

    return count;
}

void hold_low(dht11_t dht11, int hold_time_us)
{
    gpio_set_direction(dht11.dht11_pin, GPIO_MODE_OUTPUT);
    gpio_set_level(dht11.dht11_pin, 0);
    ets_delay_us(hold_time_us);
    gpio_set_level(dht11.dht11_pin, 1);
}

int dht11_read(dht11_t *dht11, int connection_timeout)
{
    int waited = 0;
    int one_duration = 0;
    int zero_duration = 0;
    int timeout_counter = 0;

    uint8_t received_data[5] = {0};

    while (timeout_counter < connection_timeout) {
        timeout_counter++;
        gpio_set_direction(dht11->dht11_pin, GPIO_MODE_INPUT);
        hold_low(*dht11, 18000);

        if ((waited = wait_for_state(*dht11, 0, 40)) == -1) {
            ESP_LOGE(TAG, "Failed at phase 1");
            ets_delay_us(20000);
            continue;
        }

        if ((waited = wait_for_state(*dht11, 1, 90)) == -1) {
            ESP_LOGE(TAG, "Failed at phase 2");
            ets_delay_us(20000);
            continue;
        }

        if ((waited = wait_for_state(*dht11, 0, 90)) == -1) {
            ESP_LOGE(TAG, "Failed at phase 3");
            ets_delay_us(20000);
            continue;
        }

        break;
    }

    if (timeout_counter == connection_timeout) return -1;

    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 8; j++) {
            zero_duration = wait_for_state(*dht11, 1, 58);
            one_duration = wait_for_state(*dht11, 0, 74);
            received_data[i] |= (one_duration > zero_duration) << (7 - j);
        }
    }

    int crc = (received_data[0] + received_data[1] +
               received_data[2] + received_data[3]) & 0xFF;

    if (crc == received_data[4]) {
        dht11->humidity = received_data[0] + received_data[1] / 10.0f;
        dht11->temperature = received_data[2] + received_data[3] / 10.0f;
        return 0;
    } else {
        ESP_LOGE(TAG, "Wrong checksum");
        return -1;
    }
}

#ifdef __cplusplus
}
#endif
