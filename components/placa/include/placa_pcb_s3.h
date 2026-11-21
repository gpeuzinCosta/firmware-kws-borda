/*
 * PCB final: ESP32-S3-WROOM-1-N8R8 (alvo esp32s3), pinagem de docs/hardware.md.
 * AINDA NÃO TESTADA: confira no esquemático antes de usar.
 */
#pragma once

#include "sdkconfig.h"

#if !CONFIG_IDF_TARGET_ESP32S3
#error "placa_pcb_s3.h e para o alvo esp32s3: rode idf.py set-target esp32s3"
#endif

/* OLED SSD1306 (I2C) */
#define OLED_PINO_SDA       8
#define OLED_PINO_SCL       9

/* microSD nos pinos do IO_MUX do FSPI (SPI2_HOST), o que permite clock mais alto. */
#define SD_HOST             SPI2_HOST
#define SD_PINO_MISO        13
#define SD_PINO_MOSI        11
#define SD_PINO_SCK         12
#define SD_PINO_CS          10
#define SD_FREQ_KHZ         20000

/* INMP441 (I2S) */
#define MIC_PINO_SCK        5
#define MIC_PINO_WS         6
#define MIC_PINO_SD         7

/* LEDs, 1 kΩ em série */
#define LED_PINO_VERMELHO   1
#define LED_PINO_BRANCO     2

/* Botão do gravador, num pino RTC (GPIO0 a 21) para o ext0. Não é o GPIO0:
 * segurá-lo ao ligar entra no modo de gravar firmware. */
#define BOTAO_PINO          14

/* Ainda não usados pelo firmware (veja o checklist de docs/hardware.md). */
#define PERIF_PINO_ENERGIA  21      /* gate do AO3401, ativo baixo */
#define BATERIA_PINO_ADC    4       /* ADC1_CH3, divisor 1 MΩ / 1 MΩ */
#define CARGA_PINO_CHRG     17      /* CHRG do TP4056, ativo baixo */
#define USB_PINO_VBUS       18      /* divisor do VBUS */
