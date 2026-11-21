/*
 * Demonstração do trabalho kws-borda no gravador: fala um comando, o ESP32 reconhece e mostra a palavra no OLED.
 *
 *   clique curto        ouve 2 s, reconhece e mostra a palavra, a confiança e os tempos (MFCC e rede)
 *   segurar 0,8 s       troca o modo: DS-CNN EN -> DS-CNN PT -> ResNet EN -> ResNet PT
 *
 * Tudo roda no próprio ESP32 (sem Wi-Fi, sem computador): MFCC, rede e head, em float32 (components/kws).
 * O monitor serial (idf.py monitor) mostra as 3 palavras mais prováveis de cada comando.
 * Fica sempre ligado pela USB: não usa deep sleep.
 */
#include <stdio.h>
#include <string.h>

#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_system.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "botao.h"
#include "kws.h"
#include "mic.h"
#include "oled.h"

static const char *TAG = "kws";

#define OUVIR_MS            2000        /* janela de gravação; a rede usa o 1 s com mais energia */
#define PULAR_MS            150         /* ignorado logo depois do clique: o som do botão e a fala apressada */
#define OUVIR_AMOSTRAS      (KWS_TAXA_HZ * OUVIR_MS / 1000)
#define SILENCIO_DBFS       (-50.0f)    /* trecho abaixo disso: "(silencio)", nem passa pela rede */
#define CONFIANCA_MINIMA    0.5f        /* abaixo disso a palavra aparece com "?" */

/* No OLED bicolor as linhas 0 a 15 são amarelas (cabeçalho) e o resto azul. */
#define Y_CABECALHO         4
#define LETRA_PX            6           /* largura de uma letra em escala 1 */

typedef struct {
    kws_modelo_t modelo;
    kws_idioma_t idioma;
} modo_t;

static const modo_t MODOS[] = {
    { KWS_DSCNN, KWS_EN }, { KWS_DSCNN, KWS_PT }, { KWS_RESNET, KWS_EN }, { KWS_RESNET, KWS_PT },
};
#define N_MODOS (sizeof(MODOS) / sizeof(MODOS[0]))

static int16_t s_audio[OUVIR_AMOSTRAS];
static float s_mfcc[KWS_N_COEF * KWS_N_FRAMES];

/* ------------------------------------------------------------------------------------------------------------ */
/* Telas                                                                                                        */
/* ------------------------------------------------------------------------------------------------------------ */

static void texto_centro(int y, const char *txt, int escala)
{
    int largura = (int)strlen(txt) * LETRA_PX * escala;
    int x = (OLED_LARGURA - largura) / 2;
    oled_texto(x < 0 ? 0 : x, y, txt, escala);
}

static void cabecalho(const modo_t *m)
{
    char txt[24];
    snprintf(txt, sizeof txt, "%s  %s", kws_nome_modelo(m->modelo), m->idioma == KWS_EN ? "ingles" : "portugues");
    oled_limpar();
    texto_centro(Y_CABECALHO, txt, 1);
}

static void tela_pronto(const modo_t *m)
{
    char txt[24];
    cabecalho(m);
    texto_centro(22, "Clique e fale", 1);
    snprintf(txt, sizeof txt, "%d palavras", kws_n_classes(m->modelo, m->idioma));
    texto_centro(36, txt, 1);
    texto_centro(54, "Segure: trocar modo", 1);
    oled_atualizar();
}

static void tela_ouvindo(const modo_t *m, int feito, int total)
{
    cabecalho(m);
    texto_centro(24, "Fale!", 2);
    oled_retangulo(4, 50, OLED_LARGURA - 8, 8, false);
    oled_retangulo(4, 50, (OLED_LARGURA - 8) * feito / total, 8, true);
    oled_atualizar();
}

static void tela_resultado(const modo_t *m, const kws_resultado_t *r, float mfcc_ms, float rede_ms)
{
    char txt[32];
    cabecalho(m);
    snprintf(txt, sizeof txt, "%s%s", r->palavra[0], r->prob[0] < CONFIANCA_MINIMA ? "?" : "");
    texto_centro(20, txt, strlen(txt) * LETRA_PX * 2 <= OLED_LARGURA ? 2 : 1);
    snprintf(txt, sizeof txt, "%d%%  (2o: %s)", (int)(r->prob[0] * 100.0f + 0.5f), r->palavra[1]);
    texto_centro(40, txt, 1);
    snprintf(txt, sizeof txt, "MFCC %.0fms rede %.0fms", mfcc_ms, rede_ms);
    texto_centro(54, txt, 1);
    oled_atualizar();
}

static void tela_mensagem(const modo_t *m, const char *grande, const char *pequena)
{
    cabecalho(m);
    texto_centro(24, grande, 2);
    if (pequena) texto_centro(50, pequena, 1);
    oled_atualizar();
}

/* ------------------------------------------------------------------------------------------------------------ */
/* Ouvir e reconhecer                                                                                           */
/* ------------------------------------------------------------------------------------------------------------ */

/* Ignora PULAR_MS e grava OUVIR_MS. Retorna quantas amostras chegaram. */
static size_t ouvir(const modo_t *m)
{
    int16_t lixo[256];
    while (mic_ler(lixo, 256, 0) > 0) { }         /* descarta o que acumulou enquanto esperava o clique */
    tela_ouvindo(m, 0, OUVIR_AMOSTRAS);
    for (size_t pular = KWS_TAXA_HZ * PULAR_MS / 1000; pular > 0; ) {
        size_t lidas = mic_ler(lixo, pular < 256 ? pular : 256, 500);
        if (lidas == 0) break;
        pular -= lidas;
    }

    size_t n = 0;
    int ultima_barra = -1;
    while (n < OUVIR_AMOSTRAS) {
        size_t lidas = mic_ler(s_audio + n, OUVIR_AMOSTRAS - n, 500);
        if (lidas == 0) break;                      /* o microfone parou */
        n += lidas;
        int barra = (int)(10 * n / OUVIR_AMOSTRAS);
        if (barra != ultima_barra) {
            tela_ouvindo(m, (int)n, OUVIR_AMOSTRAS);
            ultima_barra = barra;
        }
    }
    return n;
}

static void reconhecer(const modo_t *m)
{
    size_t n = ouvir(m);
    if (n < KWS_AMOSTRAS) {
        ESP_LOGE(TAG, "microfone: só %u amostras", (unsigned)n);
        tela_mensagem(m, "Erro mic", "confira o INMP441");
        return;
    }

    size_t ini = kws_escolher_trecho(s_audio, n);
    const int16_t *trecho = s_audio + ini;
    float nivel = kws_nivel_dbfs(trecho, KWS_AMOSTRAS);
    if (nivel < SILENCIO_DBFS) {
        ESP_LOGI(TAG, "nível %.1f dBFS: silêncio", nivel);
        tela_mensagem(m, "(silencio)", "fale mais perto");
        return;
    }

    int64_t t0 = esp_timer_get_time();
    kws_mfcc(trecho, s_mfcc);
    int64_t t1 = esp_timer_get_time();
    float logits[KWS_MAX_CLASSES];
    int nc = kws_logits(m->modelo, m->idioma, s_mfcc, logits);
    int64_t t2 = esp_timer_get_time();

    kws_resultado_t r;
    kws_top(m->idioma, logits, nc, &r);
    float mfcc_ms = (t1 - t0) / 1000.0f, rede_ms = (t2 - t1) / 1000.0f;

    ESP_LOGI(TAG, "%s %s | trecho %u ms, %.1f dBFS | MFCC %.1f ms, rede %.1f ms",
             kws_nome_modelo(m->modelo), m->idioma == KWS_EN ? "EN" : "PT",
             (unsigned)(ini * 1000 / KWS_TAXA_HZ), nivel, mfcc_ms, rede_ms);
    for (int i = 0; i < KWS_TOP; i++) {
        ESP_LOGI(TAG, "  %d. %-10s %5.1f%%", i + 1, r.palavra[i], r.prob[i] * 100.0f);
    }
    tela_resultado(m, &r, mfcc_ms, rede_ms);
}

/* ------------------------------------------------------------------------------------------------------------ */

static int proximo_modo(int atual)
{
    for (int i = 1; i <= (int)N_MODOS; i++) {
        int j = (atual + i) % N_MODOS;
        if (kws_n_classes(MODOS[j].modelo, MODOS[j].idioma) > 0) return j;   /* pula modo sem head exportada */
    }
    return atual;
}

void app_main(void)
{
    ESP_ERROR_CHECK(oled_iniciar());
    ESP_ERROR_CHECK(botao_iniciar());
    if (kws_iniciar() != 0) {
        ESP_LOGE(TAG, "sem memória para as ativações da rede");
        oled_limpar(); oled_texto(0, 24, "Sem memoria", 2); oled_atualizar();
        return;
    }
    ESP_ERROR_CHECK(mic_iniciar());

    for (int i = 0; i < (int)N_MODOS; i++) {
        ESP_LOGI(TAG, "modo %d: %s %s, %d palavras, %d parâmetros", i, kws_nome_modelo(MODOS[i].modelo),
                 MODOS[i].idioma == KWS_EN ? "EN" : "PT", kws_n_classes(MODOS[i].modelo, MODOS[i].idioma),
                 kws_n_parametros(MODOS[i].modelo, MODOS[i].idioma));
    }
    ESP_LOGI(TAG, "memória livre: %u bytes (maior bloco %u)", (unsigned)esp_get_free_heap_size(),
             (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_8BIT));

    int modo = 0;
    tela_pronto(&MODOS[modo]);
    while (true) {
        botao_evento_t ev = botao_ler_evento(1000);
        if (ev == BOTAO_MODO) {                   /* clique curto: ouvir */
            reconhecer(&MODOS[modo]);
        } else if (ev == BOTAO_OK) {              /* segurou: troca o modo */
            modo = proximo_modo(modo);
            tela_pronto(&MODOS[modo]);
        }
    }
}
