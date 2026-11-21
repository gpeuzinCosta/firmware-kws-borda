/*
 * Botão único da placa (BOTAO_PINO em placa.h, ativo em 0) e deep sleep.
 *
 *   - no menu: clique curto = MODO; segurar BOTAO_LONGO_MS = OK;
 *   - gravando (botao_gravando(true)): qualquer clique = OK.
 * O mesmo botão acorda o ESP32 do deep sleep (ext0).
 */
#pragma once

#include <stdbool.h>
#include <stdint.h>
#include "esp_err.h"

typedef enum {
    BOTAO_NADA = 0,         /* nenhum evento no prazo */
    BOTAO_MODO,
    BOTAO_OK,
} botao_evento_t;

/* Configura o pino. Se o botão já estiver apertado (o clique que acordou o
 * ESP32), ele só conta de novo depois de solto. */
esp_err_t botao_iniciar(void);

/* Espera até espera_ms por um evento (0 = só olha o botão uma vez). */
botao_evento_t botao_ler_evento(uint32_t espera_ms);

/* Durante a gravação um clique curto já vale OK. */
void botao_gravando(bool gravando);

/* true se o ESP32 acabou de acordar do deep sleep pelo botão. */
bool botao_acordou_pelo_botao(void);

/* Espera o botão ser solto e entra em deep sleep. Não retorna: o próximo
 * clique reinicia o ESP32 do começo (app_main). */
void botao_dormir(void);
