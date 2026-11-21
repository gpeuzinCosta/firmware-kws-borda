#include "botao.h"

#include "driver/gpio.h"
#include "driver/rtc_io.h"
#include "esp_log.h"
#include "esp_sleep.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "placa.h"

#define BOTAO_FIRME_MS      30      /* menos que isso é ruído do contato */
#define BOTAO_LONGO_MS      800     /* segurar isso no menu = OK */
#define BOTAO_AMOSTRA_MS    10

static const char *TAG = "botao";

static bool s_gravando;
static bool s_esperar_soltar;       /* ignora o botão até ele ser solto */
static int64_t s_apertado_desde;    /* 0 = solto */

static bool apertado(void)
{
    return gpio_get_level(BOTAO_PINO) == 0;
}

/* Olha o botão uma vez e devolve o evento, se houver. */
static botao_evento_t amostrar(void)
{
    int64_t agora = esp_timer_get_time();

    if (s_esperar_soltar) {
        if (!apertado()) {
            s_esperar_soltar = false;
            s_apertado_desde = 0;
        }
        return BOTAO_NADA;
    }
    if (apertado()) {
        if (s_apertado_desde == 0) {
            s_apertado_desde = agora;
            return BOTAO_NADA;
        }
        int64_t ms = (agora - s_apertado_desde) / 1000;
        if ((s_gravando && ms >= BOTAO_FIRME_MS) || ms >= BOTAO_LONGO_MS) {
            s_esperar_soltar = true;
            return BOTAO_OK;
        }
        return BOTAO_NADA;
    }
    if (s_apertado_desde != 0) {
        int64_t ms = (agora - s_apertado_desde) / 1000;
        s_apertado_desde = 0;
        if (ms >= BOTAO_FIRME_MS) {
            return BOTAO_MODO;
        }
    }
    return BOTAO_NADA;
}

botao_evento_t botao_ler_evento(uint32_t espera_ms)
{
    int64_t prazo = esp_timer_get_time() + espera_ms * 1000LL;
    while (true) {
        botao_evento_t ev = amostrar();
        if (ev != BOTAO_NADA || esp_timer_get_time() >= prazo) {
            return ev;
        }
        vTaskDelay(pdMS_TO_TICKS(BOTAO_AMOSTRA_MS));
    }
}

void botao_gravando(bool gravando)
{
    s_gravando = gravando;
}

bool botao_acordou_pelo_botao(void)
{
    return esp_sleep_get_wakeup_cause() == ESP_SLEEP_WAKEUP_EXT0;
}

esp_err_t botao_iniciar(void)
{
    /* Depois do deep sleep o pino ainda está com o RTC (ext0). */
    rtc_gpio_deinit(BOTAO_PINO);
    gpio_config_t cfg = {
        .pin_bit_mask = 1ULL << BOTAO_PINO,
        .mode = GPIO_MODE_INPUT,
        .pull_up_en = GPIO_PULLUP_ENABLE,
    };
    esp_err_t err = gpio_config(&cfg);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Falha ao configurar o GPIO%d: %s", BOTAO_PINO, esp_err_to_name(err));
        return err;
    }
    s_esperar_soltar = apertado();
    return ESP_OK;
}

void botao_dormir(void)
{
    /* Dormir com o botão apertado acordaria na hora. */
    while (apertado()) {
        vTaskDelay(pdMS_TO_TICKS(BOTAO_AMOSTRA_MS));
    }
    esp_sleep_enable_ext0_wakeup(BOTAO_PINO, 0);
    rtc_gpio_pullup_en(BOTAO_PINO);
    rtc_gpio_pulldown_dis(BOTAO_PINO);
    ESP_LOGW(TAG, "Deep sleep. Aperte o botao (GPIO%d) para ligar.", BOTAO_PINO);
    esp_deep_sleep_start();
}
