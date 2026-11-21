/*
 * Reconhecimento de comandos de voz (keyword spotting) do trabalho kws-borda, em C puro.
 *
 * Pipeline, idêntico ao do treino (PyTorch):
 *   áudio 16 kHz (int16) -> trecho de 1 s com mais energia -> MFCC 'hello_edge' (10 x 49)
 *   -> normalização por amostra -> rede (DS-CNN ou ResNet res15-narrow, float32)
 *   -> média global -> head (inglês: 35 palavras do GSC; português: head treinada no MSWC-pt).
 *
 * Não depende do ESP-IDF: o mesmo código compila no Mac (teste_host/) para comparar com o PyTorch.
 */
#pragma once

#include <stddef.h>
#include <stdint.h>

#define KWS_TAXA_HZ        16000
#define KWS_AMOSTRAS       16000      /* a rede vê sempre 1 s */
#define KWS_N_COEF         10
#define KWS_N_FRAMES       49
#define KWS_MAX_CLASSES    35
#define KWS_TOP            3

typedef enum { KWS_DSCNN = 0, KWS_RESNET, KWS_N_MODELOS } kws_modelo_t;
typedef enum { KWS_EN = 0, KWS_PT, KWS_N_IDIOMAS } kws_idioma_t;

typedef struct {
    int n_classes;
    int classe[KWS_TOP];               /* as KWS_TOP mais prováveis, da maior para a menor */
    float prob[KWS_TOP];               /* probabilidade (softmax), 0 a 1 */
    const char *palavra[KWS_TOP];      /* em ASCII */
} kws_resultado_t;

/* Aloca os buffers das ativações (cerca de 115 KB) e prepara a FFT. 0 = ok, -1 = sem memória. */
int kws_iniciar(void);

/* Nome curto do modelo ("DS-CNN", "ResNet") e quantas palavras a head do idioma tem (0 = não exportada). */
const char *kws_nome_modelo(kws_modelo_t modelo);
int kws_n_classes(kws_modelo_t modelo, kws_idioma_t idioma);
const char *kws_palavra(kws_idioma_t idioma, int classe);
/* Parâmetros da rede (convs) + da head do idioma. */
int kws_n_parametros(kws_modelo_t modelo, kws_idioma_t idioma);

/* Início do trecho de KWS_AMOSTRAS com mais energia (passos de 20 ms). n >= KWS_AMOSTRAS. */
size_t kws_escolher_trecho(const int16_t *audio, size_t n);

/* Nível RMS do trecho em dBFS (0 = escala cheia; silêncio da sala fica por volta de -60). */
float kws_nivel_dbfs(const int16_t *audio, size_t n);

/* MFCC normalizado de 1 s de áudio (KWS_AMOSTRAS amostras). saída: [KWS_N_COEF][KWS_N_FRAMES]. */
void kws_mfcc(const int16_t *audio, float *mfcc);

/* Rede + head: logits [kws_n_classes(modelo, idioma)]. Retorna o número de classes (0 = head ausente). */
int kws_logits(kws_modelo_t modelo, kws_idioma_t idioma, const float *mfcc, float *logits);

/* Softmax dos logits e as KWS_TOP palavras mais prováveis. */
void kws_top(kws_idioma_t idioma, const float *logits, int n_classes, kws_resultado_t *res);
