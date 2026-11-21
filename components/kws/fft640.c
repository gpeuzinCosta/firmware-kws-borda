#include "fft640.h"

#include <math.h>

#define N1 5            /* DFTs de 5 pontos */
#define N2 128          /* FFTs radix-2 de 128 pontos */

/* W^j = exp(-2 pi i j / 640), j = 0 .. 639 */
static float s_cos[FFT640_N], s_sin[FFT640_N];
static int s_pronto;

void fft640_iniciar(void)
{
    if (s_pronto) return;
    for (int j = 0; j < FFT640_N; j++) {
        double a = 2.0 * M_PI * j / FFT640_N;
        s_cos[j] = (float)cos(a);
        s_sin[j] = (float)-sin(a);
    }
    s_pronto = 1;
}

/* FFT complexa de 128 pontos, no lugar (radix-2, decimação no tempo). Twiddles: W_128^j = W_640^(5j). */
static void fft128(float *re, float *im)
{
    for (int i = 1, j = 0; i < N2; i++) {            /* permutação por inversão de bits */
        int bit = N2 >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j |= bit;
        if (i < j) {
            float t = re[i]; re[i] = re[j]; re[j] = t;
            t = im[i]; im[i] = im[j]; im[j] = t;
        }
    }
    for (int len = 2; len <= N2; len <<= 1) {
        int passo = (N2 / len) * N1;                 /* índice na tabela de 640 */
        for (int i = 0; i < N2; i += len) {
            for (int k = 0; k < len / 2; k++) {
                float wr = s_cos[k * passo], wi = s_sin[k * passo];
                int a = i + k, b = i + k + len / 2;
                float tr = re[b] * wr - im[b] * wi;
                float ti = re[b] * wi + im[b] * wr;
                re[b] = re[a] - tr; im[b] = im[a] - ti;
                re[a] += tr;        im[a] += ti;
            }
        }
    }
}

void fft640_potencia(const float *x, float *potencia, int n_bins)
{
    /* Y_n1[k2] = FFT128 de x[5 * n2 + n1] */
    static float yr[N1][N2], yi[N1][N2];
    for (int n1 = 0; n1 < N1; n1++) {
        for (int n2 = 0; n2 < N2; n2++) {
            yr[n1][n2] = x[N1 * n2 + n1];
            yi[n1][n2] = 0.0f;
        }
        fft128(yr[n1], yi[n1]);
    }

    /* X[k2 + 128 k1] = soma_n1 W_640^(n1 k2) W_5^(n1 k1) Y_n1[k2], e W_5^m = W_640^(128 m) */
    for (int k = 0; k < n_bins; k++) {
        int k2 = k % N2, k1 = k / N2;
        float sr = 0.0f, si = 0.0f;
        for (int n1 = 0; n1 < N1; n1++) {
            int j = (n1 * k2 + N2 * ((n1 * k1) % N1)) % FFT640_N;
            float wr = s_cos[j], wi = s_sin[j];
            sr += yr[n1][k2] * wr - yi[n1][k2] * wi;
            si += yr[n1][k2] * wi + yi[n1][k2] * wr;
        }
        potencia[k] = sr * sr + si * si;
    }
}
