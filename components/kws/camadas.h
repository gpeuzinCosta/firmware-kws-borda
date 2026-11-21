/*
 * Camadas da rede em float32, com os tensores em ordem (canal, linha, coluna).
 * Os pesos vêm dos headers gerados (gerado/<modelo>_pesos.h), com o BatchNorm já dobrado na convolução.
 */
#pragma once

#include <stdbool.h>

typedef struct {
    int cin, cout;          /* canais de entrada e de saída */
    int kh, kw;             /* kernel (linhas, colunas) */
    int sh, sw;             /* stride */
    int dh, dw;             /* dilatação */
    int pt, pb, pl, pr;     /* padding com zeros: cima, baixo, esquerda, direita */
    int grupos;             /* 1 = conv comum; cin = cout = grupos: depthwise */
    const float *peso;      /* [cout][cin / grupos][kh][kw] */
    const float *bias;      /* [cout] (o BatchNorm dobrado) */
} kws_conv_t;

/* Convolução 2D (+ bias, + ReLU se relu). entrada [cin][h][w] -> saída [cout][*ho][*wo]. */
void kws_conv2d(const kws_conv_t *c, const float *entrada, int h, int w, float *saida, int *ho, int *wo,
                bool relu);

/* x = max(x + atalho, 0), elemento a elemento (fim do bloco residual). */
void kws_soma_relu(float *x, const float *atalho, int n);

/* Média global: [canais][n] -> [canais]. */
void kws_media_global(const float *entrada, int canais, int n, float *saida);

/* Linear: saida[j] = bias[j] + soma_i peso[j][i] * entrada[i]. */
void kws_linear(const float *entrada, int n_entrada, const float *peso, const float *bias, int n_saida,
                float *saida);
