/*
 * Display OLED SSD1306 128x64 via I2C (driver esp_lcd do ESP-IDF).
 * Desenho num framebuffer em RAM; oled_atualizar() envia tudo ao display.
 * Texto só em ASCII (sem acentos): a fonte embutida é 5x7.
 */
#pragma once

#include <stdbool.h>
#include <stdint.h>
#include "esp_err.h"
#include "placa.h"           /* pinos */

#define OLED_ENDERECO   0x3C
#define OLED_LARGURA    128
#define OLED_ALTURA     64

/* Se o texto aparecer de cabeça para baixo no seu módulo, troque para 0. */
#define OLED_GIRAR_180  1

/* Cria o barramento I2C (se ainda não existir) e inicializa o painel. */
esp_err_t oled_iniciar(void);

/* Procura dispositivos no barramento I2C e imprime os endereços. Retorna quantos achou, ou -1 se o barramento estiver travado. */
int oled_scanner(void);

void oled_limpar(void);

/* Escreve texto com o canto superior esquerdo em (x, y) pixels. escala 1 = 6x8 px por letra. */
void oled_texto(int x, int y, const char *texto, int escala);

/* Retângulo com canto superior esquerdo em (x, y): só a borda ou cheio. */
void oled_retangulo(int x, int y, int largura, int altura, bool cheio);

/* Envia o framebuffer para o display. */
esp_err_t oled_atualizar(void);

/* Apaga o painel (modo sleep do SSD1306, ~10 µA). oled_iniciar() liga de novo. */
esp_err_t oled_desligar(void);
