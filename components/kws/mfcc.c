/*
 * MFCC 'hello_edge' idêntico ao do treino (dados/mfcc.py: torchaudio.transforms.MFCC) + normalização por amostra
 * (dados/speech_commands.py: normalizar). As tabelas (janela, filtros Mel, DCT) vêm do próprio torchaudio.
 *
 * Por janela de 40 ms (640 amostras, passo de 320, sem padding nas bordas -> 49 janelas):
 *   x * Hann -> |FFT|^2 -> 40 filtros Mel -> 10 log10(max(., 1e-10))
 * Piso fixo MFCC_PISO_DB: no treino o top_db=80 do torchaudio valia sobre o máximo de cada LOTE de 256 áudios,
 * que no GSC é quase constante; o piso efetivo foi cerca de -37 dB (detalhes em exportar/comum.py).
 * Depois a DCT ortonormal (10 coeficientes) e, por coeficiente, (x - média) / (desvio amostral + 1e-6) no tempo.
 */
#include <math.h>

#include "fft640.h"
#include "kws.h"
#include "gerado/mfcc_tabelas.h"

_Static_assert(MFCC_N_COEF == KWS_N_COEF && MFCC_N_FRAMES == KWS_N_FRAMES, "MFCC diferente do kws.h");
_Static_assert(MFCC_N_FFT == FFT640_N, "a FFT é de 640 pontos");

void kws_mfcc(const int16_t *audio, float *mfcc)
{
    static float janela[MFCC_N_FFT];
    static float potencia[MFCC_N_BINS];
    static float db[MFCC_N_FRAMES][MFCC_N_MELS];

    /* Só os bins que algum filtro Mel usa */
    int n_bins = 0;
    for (int m = 0; m < MFCC_N_MELS; m++) {
        int fim = MFCC_MEL_INICIO[m] + MFCC_MEL_TAM[m];
        if (fim > n_bins) n_bins = fim;
    }

    for (int t = 0; t < MFCC_N_FRAMES; t++) {
        const int16_t *x = audio + t * MFCC_HOP;
        for (int i = 0; i < MFCC_N_FFT; i++) {
            janela[i] = ((float)x[i] / 32768.0f) * MFCC_JANELA[i];     /* int16 -> [-1, 1), como o soundfile */
        }
        fft640_potencia(janela, potencia, n_bins);

        const float *pesos = MFCC_MEL_PESOS;
        for (int m = 0; m < MFCC_N_MELS; m++) {
            const float *p = potencia + MFCC_MEL_INICIO[m];
            float s = 0.0f;
            for (int k = 0; k < MFCC_MEL_TAM[m]; k++) s += pesos[k] * p[k];
            pesos += MFCC_MEL_TAM[m];
            float v = 10.0f * log10f(s > MFCC_AMIN ? s : MFCC_AMIN);
            db[t][m] = v < MFCC_PISO_DB ? MFCC_PISO_DB : v;
        }
    }

    for (int k = 0; k < MFCC_N_COEF; k++) {
        const float *d = MFCC_DCT + k * MFCC_N_MELS;
        float *saida = mfcc + k * MFCC_N_FRAMES;
        double soma = 0.0;
        for (int t = 0; t < MFCC_N_FRAMES; t++) {
            float s = 0.0f;
            for (int m = 0; m < MFCC_N_MELS; m++) {
                s += d[m] * db[t][m];
            }
            saida[t] = s;
            soma += s;
        }

        /* Normalização por amostra: desvio com n - 1, como o torch.std */
        double media = soma / MFCC_N_FRAMES, q = 0.0;
        for (int t = 0; t < MFCC_N_FRAMES; t++) q += (saida[t] - media) * (saida[t] - media);
        float desvio = (float)sqrt(q / (MFCC_N_FRAMES - 1));
        for (int t = 0; t < MFCC_N_FRAMES; t++) {
            saida[t] = (float)((saida[t] - media) / (desvio + MFCC_NORM_EPS));
        }
    }
}
