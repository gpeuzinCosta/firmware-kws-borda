/*
 * FFT de 640 pontos (a janela de 40 ms a 16 kHz do MFCC 'hello_edge').
 * 640 não é potência de 2: 640 = 5 x 128, então são 5 FFTs radix-2 de 128 e depois 128 DFTs de 5 pontos
 * (Cooley-Tukey em um passo).
 */
#pragma once

#define FFT640_N 640

/* Tabela de senos e cossenos (uma vez). */
void fft640_iniciar(void);

/* Espectro de potência |X[k]|^2 de um sinal real x[640], para k = 0 .. n_bins - 1 (n_bins <= 321). */
void fft640_potencia(const float *x, float *potencia, int n_bins);
