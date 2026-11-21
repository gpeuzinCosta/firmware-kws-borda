/*
 * Microfone INMP441 via I2S (driver/i2s_std.h).
 * Uma task dedicada lê o I2S e coloca amostras de 16 bits num buffer
 * circular (stream buffer) de ~2 s. Quem grava no SD lê desse buffer,
 * então uma demora do cartão não faz perder áudio.
 */
#pragma once

#include <stddef.h>
#include <stdint.h>
#include "esp_err.h"
#include "placa.h"           /* pinos */

#define MIC_TAXA_HZ     16000

/* Liga o I2S e começa a captura em segundo plano. */
esp_err_t mic_iniciar(void);

/* Copia até max_amostras amostras de 16 bits. Espera até espera_ms por dados.
 * Retorna quantas amostras copiou (0 se nada chegou no prazo). */
size_t mic_ler(int16_t *destino, size_t max_amostras, uint32_t espera_ms);

/* Amostras descartadas porque o buffer encheu (o SD não acompanhou). */
uint32_t mic_amostras_perdidas(void);

/* Para a captura e libera o I2S. */
void mic_parar(void);
