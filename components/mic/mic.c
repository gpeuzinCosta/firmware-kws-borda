/*
 * Captura do INMP441.
 *
 * O INMP441 manda 24 bits (MSB primeiro) dentro de um slot de 32 bits, no
 * formato I2S padrão (Philips). Com L/R no GND ele fala no canal esquerdo.
 * Lemos palavras de 32 bits e convertemos para 16 bits com deslocamento.
 */
#include "mic.h"

#include <string.h>
#include "driver/i2s_std.h"
#include "esp_check.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/stream_buffer.h"
#include "freertos/task.h"

static const char *TAG = "mic";

/* Deslocamento de 32 para 16 bits. 16 seria "sem ganho"; 14 dá +12 dB,
 * porque o INMP441 é baixinho em fala normal. Se o áudio saturar (clipar)
 * em voz alta perto do mic, aumente para 15 ou 16. */
#define MIC_DESLOCAMENTO   14

/* DMA: 6 descritores de 256 quadros = ~96 ms de folga no próprio hardware. */
#define MIC_DMA_DESC       6
#define MIC_DMA_QUADROS    256

/* Buffer entre a task de captura e quem lê: 16 KB = 0,5 s de áudio. No gravador são 64 KB, para aguentar
 * uma demora do cartão SD; aqui não há cartão, e a memória vai para a rede (firmware-kws-borda). */
#define MIC_BUFFER_BYTES   (16 * 1024)

/* Descarta o começo: o INMP441 leva alguns ms para estabilizar ao ligar. */
#define MIC_DESCARTE_MS    200

/* Passa-alta de 1ª ordem (corte ~25 Hz a 16 kHz): remove o deslocamento DC que o
 * INMP441 tem ao ligar e que leva segundos para sumir. Não afeta a voz. */
#define MIC_PASSA_ALTA     0.99f

static i2s_chan_handle_t s_rx;
static float s_filtro_x, s_filtro_y;
static StreamBufferHandle_t s_buffer;
static SemaphoreHandle_t s_task_fim;
static volatile bool s_rodando;
static volatile uint32_t s_perdidas;

static void task_captura(void *arg)
{
    static int32_t brutas[MIC_DMA_QUADROS];
    static int16_t pcm[MIC_DMA_QUADROS];
    size_t descartar = MIC_TAXA_HZ * MIC_DESCARTE_MS / 1000;

    while (s_rodando) {
        size_t lidos = 0;
        esp_err_t err = i2s_channel_read(s_rx, brutas, sizeof(brutas), &lidos, pdMS_TO_TICKS(200));
        if (err != ESP_OK) {
            if (err != ESP_ERR_TIMEOUT) {
                ESP_LOGE(TAG, "Erro lendo I2S: %s", esp_err_to_name(err));
            }
            continue;
        }
        size_t n = lidos / sizeof(int32_t);

        /* Passa-alta (tira o DC) e 32 -> 16 bits com saturação. Roda também no
         * trecho descartado, para o filtro já estar assentado quando gravar.
         * Os 24 bits úteis cabem exatos num float. */
        for (size_t i = 0; i < n; i++) {
            float x = (float)(brutas[i] >> 8);
            s_filtro_y = x - s_filtro_x + MIC_PASSA_ALTA * s_filtro_y;
            s_filtro_x = x;
            int32_t v = (int32_t)(s_filtro_y / (1 << (MIC_DESLOCAMENTO - 8)));
            if (v > INT16_MAX) v = INT16_MAX;
            if (v < INT16_MIN) v = INT16_MIN;
            pcm[i] = (int16_t)v;
        }

        if (descartar) {
            size_t d = n < descartar ? n : descartar;
            descartar -= d;
            continue;
        }

        /* Não bloqueia: se o buffer estiver cheio, conta a perda e segue. */
        size_t enviados = xStreamBufferSend(s_buffer, pcm, n * sizeof(int16_t), 0);
        if (enviados < n * sizeof(int16_t)) {
            s_perdidas += (n * sizeof(int16_t) - enviados) / sizeof(int16_t);
        }
    }
    xSemaphoreGive(s_task_fim);
    vTaskDelete(NULL);
}

esp_err_t mic_iniciar(void)
{
    if (s_rodando) {
        return ESP_OK;
    }

    /* Canal I2S como mestre (o ESP32 gera SCK e WS), só recepção. */
    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
    chan_cfg.dma_desc_num = MIC_DMA_DESC;
    chan_cfg.dma_frame_num = MIC_DMA_QUADROS;
    ESP_RETURN_ON_ERROR(i2s_new_channel(&chan_cfg, NULL, &s_rx), TAG, "criar canal I2S");

    /* Formato Philips, 16 kHz, slot de 32 bits, mono no canal esquerdo. */
    i2s_std_config_t std_cfg = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(MIC_TAXA_HZ),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_MONO),
        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,
            .bclk = MIC_PINO_SCK,
            .ws = MIC_PINO_WS,
            .dout = I2S_GPIO_UNUSED,
            .din = MIC_PINO_SD,
            .invert_flags = { 0 },
        },
    };
    std_cfg.slot_cfg.slot_mask = I2S_STD_SLOT_LEFT;

    esp_err_t err = i2s_channel_init_std_mode(s_rx, &std_cfg);
    if (err == ESP_OK) {
        err = i2s_channel_enable(s_rx);
    }
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Falha ao configurar o I2S: %s", esp_err_to_name(err));
        i2s_del_channel(s_rx);
        s_rx = NULL;
        return err;
    }

    s_buffer = xStreamBufferCreate(MIC_BUFFER_BYTES, sizeof(int16_t));
    s_task_fim = xSemaphoreCreateBinary();
    if (!s_buffer || !s_task_fim) {
        ESP_LOGE(TAG, "Sem memoria para o buffer de audio");
        mic_parar();
        return ESP_ERR_NO_MEM;
    }

    s_perdidas = 0;
    s_filtro_x = s_filtro_y = 0;
    s_rodando = true;
    /* Prioridade acima da task principal e no núcleo 1, longe do SD/console. */
    if (xTaskCreatePinnedToCore(task_captura, "mic_captura", 4096, NULL, 10, NULL, 1) != pdPASS) {
        s_rodando = false;
        mic_parar();
        return ESP_ERR_NO_MEM;
    }
    ESP_LOGI(TAG, "Captura iniciada: %d Hz, mono, slot 32 bits -> PCM 16 bits", MIC_TAXA_HZ);
    return ESP_OK;
}

size_t mic_ler(int16_t *destino, size_t max_amostras, uint32_t espera_ms)
{
    if (!s_buffer) {
        return 0;
    }
    size_t bytes = xStreamBufferReceive(s_buffer, destino, max_amostras * sizeof(int16_t),
                                        pdMS_TO_TICKS(espera_ms));
    return bytes / sizeof(int16_t);
}

uint32_t mic_amostras_perdidas(void)
{
    return s_perdidas;
}

void mic_parar(void)
{
    /* Pede para a task terminar e espera ela sair antes de apagar o I2S. */
    if (s_rodando) {
        s_rodando = false;
        xSemaphoreTake(s_task_fim, pdMS_TO_TICKS(1000));
    }
    if (s_rx) {
        i2s_channel_disable(s_rx);
        i2s_del_channel(s_rx);
        s_rx = NULL;
    }
    if (s_buffer) {
        vStreamBufferDelete(s_buffer);
        s_buffer = NULL;
    }
    if (s_task_fim) {
        vSemaphoreDelete(s_task_fim);
        s_task_fim = NULL;
    }
}
