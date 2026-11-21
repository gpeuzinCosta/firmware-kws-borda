/*
 * Breadboard: ESP32-WROOM-32E DevKit (alvo esp32), alimentado pela USB.
 */
#pragma once

#include "sdkconfig.h"

#if !CONFIG_IDF_TARGET_ESP32
#error "placa_devkit_wroom32e.h e para o alvo esp32: rode idf.py set-target esp32"
#endif

/* OLED SSD1306 (I2C) */
#define OLED_PINO_SDA       21
#define OLED_PINO_SCL       22

/* microSD (SPI). SPI3_HOST (VSPI) é o barramento cujos pinos nativos são
 * 18/19/23/5. Na protoboard os fios longos não gostam de 20 MHz; 10 MHz sobra
 * para gravar áudio (32 KB/s). */
#define SD_HOST             SPI3_HOST
#define SD_PINO_MISO        19
#define SD_PINO_MOSI        23
#define SD_PINO_SCK         18
#define SD_PINO_CS          5       /* pino de boot, mas inofensivo aqui */
#define SD_FREQ_KHZ         10000

/* INMP441 (I2S) */
#define MIC_PINO_SCK        26
#define MIC_PINO_WS         25
#define MIC_PINO_SD         33

/* LEDs: -1 = sem LED. O DevKit não tem LED de usuário e a protoboard está sem
 * LEDs; o estado aparece no OLED. Para voltar a
 * usá-los: 27 (vermelho) e 32 (branco), 220 Ω em série, cátodo no GND. */
#define LED_PINO_VERMELHO   -1
#define LED_PINO_BRANCO     -1

/* Botão BOOT do DevKit (GPIO0, pull-up de 10k na placa). Também acorda o
 * ESP32 do deep sleep (ext0). */
#define BOTAO_PINO          0

/* O DevKit não mede a bateria: o OLED não mostra a carga. Para medir, um
 * divisor da bateria até um pino ADC1 (ex.: GPIO34) e o número do pino aqui. */
#define BATERIA_PINO_ADC    -1
