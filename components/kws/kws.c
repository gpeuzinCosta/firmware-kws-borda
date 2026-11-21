/*
 * As duas redes (mesma sequência de camadas do exportador, exportar.py:embedding_dobrado) e a API de kws.h.
 */
#include "kws.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "camadas.h"
#include "fft640.h"
#include "gerado/dscnn_pesos.h"
#include "gerado/resnet_pesos.h"
#include "gerado/rotulos.h"

_Static_assert(DSCNN_N_EN == KWS_N_EN && RESNET_N_EN == KWS_N_EN, "heads EN com tamanhos diferentes");
_Static_assert(KWS_N_EN <= KWS_MAX_CLASSES && KWS_N_PT <= KWS_MAX_CLASSES, "aumente KWS_MAX_CLASSES");

/* Maior ativação: ResNet, 19 canais x 10 x 49 (a DS-CNN chega a 48 x 25 x 5). */
#define ATIV_MAX (19 * KWS_N_COEF * KWS_N_FRAMES)
#define CANAIS_MAX 64

static float *s_a, *s_b, *s_c;

int kws_iniciar(void)
{
    fft640_iniciar();
    if (!s_a) {
        s_a = malloc(ATIV_MAX * sizeof(float));
        s_b = malloc(ATIV_MAX * sizeof(float));
        s_c = malloc(ATIV_MAX * sizeof(float));
    }
    return (s_a && s_b && s_c) ? 0 : -1;
}

/* Confere, uma camada por vez, que a saída cabe nos buffers. */
static int saida_cabe(const kws_conv_t *c, int h, int w)
{
    int ho = (h + c->pt + c->pb - c->dh * (c->kh - 1) - 1) / c->sh + 1;
    int wo = (w + c->pl + c->pr - c->dw * (c->kw - 1) - 1) / c->sw + 1;
    return c->cout * ho * wo <= ATIV_MAX;
}

/* DS-CNN: entrada (1, tempo 49, frequência 10); conv inicial e 4 x (depthwise, pointwise), todas com ReLU. */
static void embedding_dscnn(const float *mfcc, float *emb)
{
    float *x = s_a, *y = s_b;
    for (int k = 0; k < KWS_N_COEF; k++)                 /* transpõe [coef][tempo] -> [tempo][coef] */
        for (int t = 0; t < KWS_N_FRAMES; t++)
            x[t * KWS_N_COEF + k] = mfcc[k * KWS_N_FRAMES + t];

    int h = KWS_N_FRAMES, w = KWS_N_COEF;
    for (int i = 0; i < DSCNN_N_CONVS; i++) {
        if (!saida_cabe(&DSCNN_CONVS[i], h, w)) abort();
        kws_conv2d(&DSCNN_CONVS[i], x, h, w, y, &h, &w, true);
        float *t = x; x = y; y = t;
    }
    kws_media_global(x, DSCNN_CANAIS, h * w, emb);
}

/* ResNet res15-narrow: entrada (1, 10, 49); conv0, blocos residuais (conv, ReLU, conv, + atalho, ReLU) e a conv
 * extra quando sobra uma. */
static void embedding_resnet(const float *mfcc, float *emb)
{
    float *x = s_a, *y = s_b, *z = s_c;
    int h = KWS_N_COEF, w = KWS_N_FRAMES, h2, w2;

    memcpy(z, mfcc, sizeof(float) * KWS_N_COEF * KWS_N_FRAMES);
    kws_conv2d(&RESNET_CONVS[0], z, h, w, x, &h, &w, true);

    const kws_conv_t *resto = RESNET_CONVS + 1;
    const int n = RESNET_N_CONVS - 1;
    for (int b = 0; b < n / 2; b++) {
        if (!saida_cabe(&resto[2 * b], h, w)) abort();
        kws_conv2d(&resto[2 * b], x, h, w, y, &h2, &w2, true);
        kws_conv2d(&resto[2 * b + 1], y, h2, w2, z, &h2, &w2, false);
        if (h2 != h || w2 != w) abort();                 /* o atalho exige o mesmo formato */
        kws_soma_relu(z, x, RESNET_CANAIS * h * w);
        float *t = x; x = z; z = t;
    }
    if (n % 2 == 1) {
        kws_conv2d(&resto[n - 1], x, h, w, y, &h, &w, true);
        x = y;
    }
    kws_media_global(x, RESNET_CANAIS, h * w, emb);
}

const char *kws_nome_modelo(kws_modelo_t modelo)
{
    return modelo == KWS_DSCNN ? "DS-CNN" : "ResNet";
}

int kws_n_classes(kws_modelo_t modelo, kws_idioma_t idioma)
{
    if (idioma == KWS_EN) return KWS_N_EN;
    return modelo == KWS_DSCNN ? DSCNN_N_PT : RESNET_N_PT;
}

const char *kws_palavra(kws_idioma_t idioma, int classe)
{
    if (idioma == KWS_EN) return (classe >= 0 && classe < KWS_N_EN) ? KWS_ROTULOS_EN[classe] : "?";
    return (classe >= 0 && classe < KWS_N_PT) ? KWS_ROTULOS_PT[classe] : "?";
}

int kws_n_parametros(kws_modelo_t modelo, kws_idioma_t idioma)
{
    int canais = modelo == KWS_DSCNN ? DSCNN_CANAIS : RESNET_CANAIS;
    int rede = modelo == KWS_DSCNN ? DSCNN_PARAMS_REDE : RESNET_PARAMS_REDE;
    int n = kws_n_classes(modelo, idioma);
    return rede + n * (canais + 1);
}

int kws_logits(kws_modelo_t modelo, kws_idioma_t idioma, const float *mfcc, float *logits)
{
    int n = kws_n_classes(modelo, idioma);
    if (n == 0) return 0;

    float emb[CANAIS_MAX];
    if (modelo == KWS_DSCNN) {
        embedding_dscnn(mfcc, emb);
        if (idioma == KWS_EN) kws_linear(emb, DSCNN_CANAIS, dscnn_head_en_w, dscnn_head_en_b, n, logits);
        else                  kws_linear(emb, DSCNN_CANAIS, dscnn_head_pt_w, dscnn_head_pt_b, n, logits);
    } else {
        embedding_resnet(mfcc, emb);
        if (idioma == KWS_EN) kws_linear(emb, RESNET_CANAIS, resnet_head_en_w, resnet_head_en_b, n, logits);
        else                  kws_linear(emb, RESNET_CANAIS, resnet_head_pt_w, resnet_head_pt_b, n, logits);
    }
    return n;
}

void kws_top(kws_idioma_t idioma, const float *logits, int n, kws_resultado_t *res)
{
    float prob[KWS_MAX_CLASSES];
    float maximo = -INFINITY, soma = 0.0f;
    for (int i = 0; i < n; i++) if (logits[i] > maximo) maximo = logits[i];
    for (int i = 0; i < n; i++) { prob[i] = expf(logits[i] - maximo); soma += prob[i]; }
    for (int i = 0; i < n; i++) prob[i] /= soma;

    res->n_classes = n;
    for (int j = 0; j < KWS_TOP; j++) {
        int melhor = -1;
        for (int i = 0; i < n; i++) {
            int usado = 0;
            for (int q = 0; q < j; q++) usado |= (res->classe[q] == i);
            if (!usado && (melhor < 0 || prob[i] > prob[melhor])) melhor = i;
        }
        res->classe[j] = melhor;
        res->prob[j] = melhor >= 0 ? prob[melhor] : 0.0f;
        res->palavra[j] = melhor >= 0 ? kws_palavra(idioma, melhor) : "";
    }
}

size_t kws_escolher_trecho(const int16_t *audio, size_t n)
{
    if (n <= KWS_AMOSTRAS) return 0;
    const size_t passo = 320;                          /* 20 ms */
    size_t melhor = 0;
    double melhor_e = -1.0;
    for (size_t ini = 0; ini + KWS_AMOSTRAS <= n; ini += passo) {
        double e = 0.0;
        for (size_t i = ini; i < ini + KWS_AMOSTRAS; i++) e += (double)audio[i] * audio[i];
        if (e > melhor_e) { melhor_e = e; melhor = ini; }
    }
    return melhor;
}

float kws_nivel_dbfs(const int16_t *audio, size_t n)
{
    double e = 0.0;
    for (size_t i = 0; i < n; i++) e += (double)audio[i] * audio[i];
    double rms = sqrt(e / (n ? n : 1)) / 32768.0;
    return (float)(20.0 * log10(rms > 1e-9 ? rms : 1e-9));
}
