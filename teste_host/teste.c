/*
 * Teste no Mac: o núcleo em C (components/kws) contra o PyTorch, nos vetores de exportar/vetores_teste.py.
 *
 *   make -C teste_host && ./teste_host/teste
 *
 * Confere a FFT de 640 pontos, o MFCC normalizado e os logits das duas redes (inglês e, se houver, português).
 * Sai com código 1 se algum limite falhar.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "kws.h"
#include "fft640.h"

#define PASTA "vetores/"

static int s_falhas;

static void *ler(const char *nome, size_t tam_item, size_t *n)
{
    char caminho[256];
    snprintf(caminho, sizeof caminho, PASTA "%s", nome);
    FILE *f = fopen(caminho, "rb");
    if (!f) { *n = 0; return NULL; }
    fseek(f, 0, SEEK_END);
    long bytes = ftell(f);
    fseek(f, 0, SEEK_SET);
    void *p = malloc(bytes);
    if (fread(p, 1, bytes, f) != (size_t)bytes) { fclose(f); free(p); *n = 0; return NULL; }
    fclose(f);
    *n = bytes / tam_item;
    return p;
}

static void checar(int ok, const char *o_que)
{
    printf("  [%s] %s\n", ok ? "ok" : "FALHOU", o_que);
    if (!ok) s_falhas++;
}

static double agora_ms(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

static void teste_fft(void)
{
    size_t nx, np_;
    float *x = ler("fft_x.f32", sizeof(float), &nx);
    float *ref = ler("fft_p.f32", sizeof(float), &np_);
    if (!x || !ref) { printf("FFT: sem vetores\n"); s_falhas++; return; }
    int n = (int)(nx / 640);
    float p[321];
    double erro = 0.0;
    for (int i = 0; i < n; i++) {
        fft640_potencia(x + i * 640, p, 321);
        double maior = 0.0;
        for (int k = 0; k < 321; k++) if (ref[i * 321 + k] > maior) maior = ref[i * 321 + k];
        for (int k = 0; k < 321; k++) {
            double e = fabs(p[k] - ref[i * 321 + k]) / maior;
            if (e > erro) erro = e;
        }
    }
    printf("FFT 640 (%d sinais): erro relativo máximo %.2e\n", n, erro);
    checar(erro < 1e-5, "FFT igual ao numpy.fft.rfft");
    free(x); free(ref);
}

static void teste_conjunto(const char *prefixo, kws_idioma_t idioma)
{
    char nome[128];
    size_t n_audio, n_rot, n_mfcc;
    snprintf(nome, sizeof nome, "%s_audio.i16", prefixo);
    int16_t *audio = ler(nome, sizeof(int16_t), &n_audio);
    snprintf(nome, sizeof nome, "%s_rotulo.i32", prefixo);
    int32_t *rotulo = ler(nome, sizeof(int32_t), &n_rot);
    snprintf(nome, sizeof nome, "%s_mfcc.f32", prefixo);
    float *mfcc_ref = ler(nome, sizeof(float), &n_mfcc);
    if (!audio || !rotulo || !mfcc_ref) { printf("%s: sem vetores (pulando)\n", prefixo); return; }
    int n = (int)(n_audio / KWS_AMOSTRAS);
    const int tam = KWS_N_COEF * KWS_N_FRAMES;
    printf("\n%s: %d áudios\n", prefixo, n);

    float *mfcc = malloc(sizeof(float) * tam * n);
    double erro = 0.0, t0 = agora_ms();
    for (int i = 0; i < n; i++) {
        kws_mfcc(audio + (size_t)i * KWS_AMOSTRAS, mfcc + (size_t)i * tam);
        for (int j = 0; j < tam; j++) {
            double e = fabs(mfcc[i * tam + j] - mfcc_ref[i * tam + j]);
            if (e > erro) erro = e;
        }
    }
    printf("MFCC: erro máximo %.2e | %.3f ms por áudio no Mac\n", erro, (agora_ms() - t0) / n);
    checar(erro < 2e-3, "MFCC normalizado igual ao do treino (torchaudio + normalizar)");

    for (kws_modelo_t m = 0; m < KWS_N_MODELOS; m++) {
        int nc = kws_n_classes(m, idioma);
        const char *sufixo = m == KWS_DSCNN ? "dscnn" : "resnet";
        size_t n_ref;
        snprintf(nome, sizeof nome, "%s_%s_%s.f32", prefixo, sufixo, idioma == KWS_EN ? "en" : "pt");
        float *ref = ler(nome, sizeof(float), &n_ref);
        if (!ref || nc == 0) { printf("%s: sem logits de referência ou sem head (pulando)\n", kws_nome_modelo(m)); free(ref); continue; }
        if ((int)(n_ref / n) != nc) { printf("%s: referência com %zu classes, o C tem %d\n", kws_nome_modelo(m), n_ref / n, nc); s_falhas++; free(ref); continue; }

        float logits[KWS_MAX_CLASSES];
        int iguais = 0, certos_c = 0, certos_ref = 0;
        double erro_l = 0.0;
        t0 = agora_ms();
        for (int i = 0; i < n; i++) {
            /* Logits a partir do MFCC do C: confere o pipeline inteiro, não só a rede */
            kws_logits(m, idioma, mfcc + (size_t)i * tam, logits);
            int a = 0, b = 0;
            for (int k = 0; k < nc; k++) {
                double e = fabs(logits[k] - ref[i * nc + k]);
                if (e > erro_l) erro_l = e;
                if (logits[k] > logits[a]) a = k;
                if (ref[i * nc + k] > ref[i * nc + b]) b = k;
            }
            iguais += a == b;
            certos_c += a == rotulo[i];
            certos_ref += b == rotulo[i];
        }
        double ms = (agora_ms() - t0) / n;
        printf("%s %s: logits erro máximo %.2e | mesma palavra em %.1f%% | acurácia C %.1f%%, PyTorch %.1f%% | "
               "%.3f ms por áudio no Mac\n", kws_nome_modelo(m), idioma == KWS_EN ? "EN" : "PT", erro_l,
               100.0 * iguais / n, 100.0 * certos_c / n, 100.0 * certos_ref / n, ms);
        checar(100.0 * iguais / n >= 99.5, "mesma palavra prevista que o PyTorch em pelo menos 99,5% dos áudios");
        free(ref);
    }
    free(mfcc); free(audio); free(rotulo); free(mfcc_ref);
}

int main(void)
{
    if (kws_iniciar() != 0) { printf("sem memória\n"); return 1; }
    teste_fft();
    teste_conjunto("gsc", KWS_EN);
    teste_conjunto("pt", KWS_PT);
    printf("\n%s\n", s_falhas ? "ALGUM TESTE FALHOU" : "todos os testes passaram");
    return s_falhas ? 1 : 0;
}
