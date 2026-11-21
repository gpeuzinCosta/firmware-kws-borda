#include "camadas.h"

#include <string.h>

/* Faixa de saídas o (0 <= o < n_saida) cuja entrada i = o * s - pad + desloc cai dentro de [0, n_entrada). */
static void faixa_valida(int n_entrada, int n_saida, int s, int pad, int desloc, int *ini, int *fim)
{
    int base = desloc - pad;                         /* i = o * s + base */
    int a = base >= 0 ? 0 : (-base + s - 1) / s;     /* menor o com i >= 0 */
    int lim = n_entrada - 1 - base;                  /* o * s <= lim */
    int b = lim < 0 ? -1 : lim / s;                  /* maior o com i < n_entrada */
    if (b > n_saida - 1) b = n_saida - 1;
    *ini = a;
    *fim = b;                                        /* vazio se fim < ini */
}

/*
 * Para cada canal de saída: começa no bias e, para cada (canal de entrada, ky, kx), soma peso * plano de entrada
 * deslocado. Assim o laço de dentro percorre uma linha contígua, sem testes de borda: as bordas (padding com
 * zeros) são resolvidas antes, pela faixa de linhas e colunas válidas.
 */
void kws_conv2d(const kws_conv_t *c, const float *entrada, int h, int w, float *saida, int *pho, int *pwo,
                bool relu)
{
    const int ho = (h + c->pt + c->pb - c->dh * (c->kh - 1) - 1) / c->sh + 1;
    const int wo = (w + c->pl + c->pr - c->dw * (c->kw - 1) - 1) / c->sw + 1;
    const int cin_g = c->cin / c->grupos;
    const int cout_g = c->cout / c->grupos;
    const int plano = ho * wo;

    for (int oc = 0; oc < c->cout; oc++) {
        float *out = saida + (size_t)oc * plano;
        const float b = c->bias[oc];
        for (int i = 0; i < plano; i++) out[i] = b;

        const int g = oc / cout_g;
        for (int icl = 0; icl < cin_g; icl++) {
            const float *in = entrada + (size_t)(g * cin_g + icl) * h * w;
            const float *pk = c->peso + ((size_t)oc * cin_g + icl) * c->kh * c->kw;
            for (int ky = 0; ky < c->kh; ky++) {
                int y0, y1;
                faixa_valida(h, ho, c->sh, c->pt, ky * c->dh, &y0, &y1);
                if (y1 < y0) continue;
                for (int kx = 0; kx < c->kw; kx++) {
                    int x0, x1;
                    faixa_valida(w, wo, c->sw, c->pl, kx * c->dw, &x0, &x1);
                    if (x1 < x0) continue;
                    const float p = pk[ky * c->kw + kx];
                    for (int oy = y0; oy <= y1; oy++) {
                        const int iy = oy * c->sh - c->pt + ky * c->dh;
                        const float *lin = in + (size_t)iy * w - c->pl + kx * c->dw;
                        float *o = out + (size_t)oy * wo;
                        if (c->sw == 1) {
                            for (int ox = x0; ox <= x1; ox++) o[ox] += p * lin[ox];
                        } else {
                            for (int ox = x0; ox <= x1; ox++) o[ox] += p * lin[ox * c->sw];
                        }
                    }
                }
            }
        }
        if (relu) {
            for (int i = 0; i < plano; i++) if (out[i] < 0.0f) out[i] = 0.0f;
        }
    }
    *pho = ho;
    *pwo = wo;
}

void kws_soma_relu(float *x, const float *atalho, int n)
{
    for (int i = 0; i < n; i++) {
        float v = x[i] + atalho[i];
        x[i] = v > 0.0f ? v : 0.0f;
    }
}

void kws_media_global(const float *entrada, int canais, int n, float *saida)
{
    for (int c = 0; c < canais; c++) {
        const float *p = entrada + (size_t)c * n;
        float s = 0.0f;
        for (int i = 0; i < n; i++) s += p[i];
        saida[c] = s / (float)n;
    }
}

void kws_linear(const float *entrada, int n_entrada, const float *peso, const float *bias, int n_saida,
                float *saida)
{
    for (int j = 0; j < n_saida; j++) {
        const float *pj = peso + (size_t)j * n_entrada;
        float s = bias[j];
        for (int i = 0; i < n_entrada; i++) s += pj[i] * entrada[i];
        saida[j] = s;
    }
}
