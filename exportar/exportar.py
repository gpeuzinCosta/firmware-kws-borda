"""Exporta os modelos do kws-borda (PyTorch) para headers C do firmware (components/kws/gerado/).

O que sai:
- mfcc_tabelas.h: janela de Hann, banco de filtros Mel (esparso) e matriz DCT, tirados do PRÓPRIO
  torchaudio.transforms.MFCC do treino (dados/mfcc.py). Assim o C não reimplementa as fórmulas, só multiplica;
- <modelo>_pesos.h: cada convolução com o BatchNorm seguinte já dobrado nela (peso e bias), a head em inglês
  (35 palavras do GSC) e, se existir, a head em português (exportar/heads/head_pt_<modelo>.pt);
- rotulos.h: as palavras em inglês e em português, em ASCII (o OLED não tem acentos).

Antes de escrever, confere que a rede com o BN dobrado (rodada aqui no torch, na MESMA sequência de camadas que o
C executa) dá os mesmos logits do modelo original em 500 entradas do teste do GSC.

    ~/projetos/kws-borda/.venv/bin/python exportar/exportar.py
"""
import argparse
from datetime import date

import numpy as np
import torch
import torch.nn.functional as F

from comum import (MODELOS, PISO_DB, DRIVE_KWS, PASTA_GERADO, ascii_oled, carregar_head_pt, carregar_modelo,
                   classes_en)


#--------------------------------------------------------------------------------------------------------------
#Camadas: cada convolução vira um dicionário com o BatchNorm já dobrado
#--------------------------------------------------------------------------------------------------------------

def dobrar_bn(conv, bn):
    """Conv2d (sem bias) seguida de BatchNorm2d (modo eval) -> (peso, bias) de uma conv só.

    BN(y) = gamma * (y - media) / sqrt(var + eps) + beta, e y = W * x, então
    W' = W * escala (por canal de saída) e b' = beta - media * escala, com escala = gamma / sqrt(var + eps).
    """
    escala = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    peso = conv.weight * escala.view(-1, 1, 1, 1)
    bias = bn.bias - bn.running_mean * escala
    if conv.bias is not None:
        bias = bias + conv.bias * escala
    return peso.detach().clone(), bias.detach().clone()


def camada(conv, bn, pad=None):
    """Descrição de uma conv para o C. pad = (cima, baixo, esquerda, direita); None = o padding da própria conv."""
    peso, bias = dobrar_bn(conv, bn)
    if pad is None:
        ph, pw = conv.padding
        pad = (ph, ph, pw, pw)
    return dict(peso=peso, bias=bias, cin=conv.in_channels, cout=conv.out_channels,
                kh=conv.kernel_size[0], kw=conv.kernel_size[1], sh=conv.stride[0], sw=conv.stride[1],
                dh=conv.dilation[0], dw=conv.dilation[1], pt=pad[0], pb=pad[1], pl=pad[2], pr=pad[3],
                grupos=conv.groups)


def camadas_dscnn(m):
    """DS-CNN: a conv inicial (com o ZeroPad SAME) e, por bloco, depthwise e pointwise. Todas seguidas de ReLU."""
    esq, dir_, cima, baixo = m.padding_inicial.padding      #ZeroPad2d: (esquerda, direita, cima, baixo)
    convs = [camada(m.conv_inicial, m.bn_inicial, pad=(cima, baixo, esq, dir_))]
    for bloco in m.blocos:
        convs.append(camada(bloco.depthwise, bloco.bn_depthwise))
        convs.append(camada(bloco.pointwise, bloco.bn_pointwise))
    return convs


def camadas_resnet(m):
    """ResNet: conv0 e depois as convs dos blocos residuais (2 por bloco) e a conv extra (res15)."""
    assert isinstance(m.pool, torch.nn.Identity), "o C só implementa a ResNet sem pooling na entrada (res15)"
    convs = [camada(m.conv0, m.bn0)]
    for conv, bn in zip(m.convs, m.bns):
        convs.append(camada(conv, bn))
    return convs


EXTRAIR = {"dscnn": camadas_dscnn, "resnet": camadas_resnet}


#--------------------------------------------------------------------------------------------------------------
#Referência: a MESMA sequência que o C executa (rede_dscnn.c e rede_resnet.c), com as camadas dobradas
#--------------------------------------------------------------------------------------------------------------

def conv(c, x):
    x = F.pad(x, (c["pl"], c["pr"], c["pt"], c["pb"]))
    return F.conv2d(x, c["peso"], c["bias"], stride=(c["sh"], c["sw"]), dilation=(c["dh"], c["dw"]),
                    groups=c["grupos"])


def embedding_dobrado(nome, convs, x):
    """x: MFCC normalizado (B, 10, 49) -> embedding (B, canais), antes da head."""
    if nome == "dscnn":
        x = x.transpose(1, 2).unsqueeze(1)          #(B, 1, tempo, frequência), como no modelo
        for c in convs:
            x = F.relu(conv(c, x))
    else:
        x = x.unsqueeze(1)                          #(B, 1, n_mfcc, n_frames)
        x = F.relu(conv(convs[0], x))
        resto = convs[1:]
        for b in range(len(resto) // 2):
            y = F.relu(conv(resto[2 * b], x))
            y = conv(resto[2 * b + 1], y)
            x = F.relu(y + x)
        if len(resto) % 2 == 1:
            x = F.relu(conv(resto[-1], x))
    return x.mean(dim=(2, 3))


def conferir(nome, modelo, convs, n=500):
    """Logits da rede dobrada contra o modelo original, em n entradas do teste do GSC (cache do Drive)."""
    from dados import normalizar
    X, _ = torch.load(DRIVE_KWS / "cache" / "gsc_teste_gsc35_10x49_40-20msnc.pt")
    x = normalizar(X[:n].float())
    with torch.no_grad():
        original = modelo(x)
        dobrado = F.linear(embedding_dobrado(nome, convs, x), modelo.head.weight, modelo.head.bias)
    erro = (original - dobrado).abs().max().item()
    iguais = (original.argmax(1) == dobrado.argmax(1)).float().mean().item()
    print(f"{nome}: BN dobrado vs. original | erro máximo dos logits {erro:.2e} | mesma classe em {iguais:.1%}")
    assert erro < 1e-3 and iguais == 1.0, "a rede dobrada não bate com o modelo original"


#--------------------------------------------------------------------------------------------------------------
#Escrita dos headers
#--------------------------------------------------------------------------------------------------------------

CABECALHO = """/*
 * GERADO por kws/exportar/{script} em {data}. Não edite à mão: rode o exportador de novo.
 * {descricao}
 */
#pragma once
"""


def floats(valores, por_linha=8):
    if isinstance(valores, torch.Tensor):
        valores = valores.detach().cpu().numpy()
    v = np.asarray(valores, dtype=np.float32).ravel()
    assert np.isfinite(v).all(), "peso não finito"

    def literal(x):
        s = f"{x:.9g}"                    #9 dígitos: o float32 volta exatamente
        if not any(ch in s for ch in ".en"):
            s += ".0"                     #"0" -> "0.0": "0f" não é um literal C válido
        return s + "f"

    linhas = []
    for i in range(0, len(v), por_linha):
        linhas.append("    " + ", ".join(literal(x) for x in v[i:i + por_linha]) + ",")
    return "{\n" + "\n".join(linhas) + "\n}"


def cabecalho(descricao):
    return CABECALHO.format(script="exportar.py", data=date.today().isoformat(), descricao=descricao)


def escrever_mfcc(hp):
    from dados import ExtratorMFCC
    t = ExtratorMFCC.de_hparams(hp).mfcc
    esp = t.MelSpectrogram.spectrogram
    assert esp.power == 2.0 and not esp.normalized and not esp.center and not t.log_mels
    janela = esp.window.numpy()                                     #(n_fft,) Hann periódica
    fb = t.MelSpectrogram.mel_scale.fb.numpy().T                    #(n_mels, n_fft/2+1)
    dct = t.dct_mat.numpy().T                                       #(n_mfcc, n_mels)
    a2db = t.amplitude_to_DB

    inicio, tam, pesos = [], [], []
    for linha in fb:
        nz = np.nonzero(linha)[0]
        inicio.append(int(nz[0])); tam.append(int(nz[-1] - nz[0] + 1))
        pesos.extend(linha[nz[0]:nz[-1] + 1])

    n_fft, hop = esp.n_fft, esp.hop_length
    n_frames = 1 + (16000 - n_fft) // hop
    texto = cabecalho("Tabelas do MFCC 'hello_edge' do treino (torchaudio.transforms.MFCC de dados/mfcc.py).") + f"""
#define MFCC_N_AMOSTRAS   16000
#define MFCC_N_FFT        {n_fft}
#define MFCC_HOP          {hop}
#define MFCC_N_FRAMES     {n_frames}
#define MFCC_N_BINS       {n_fft // 2 + 1}
#define MFCC_N_MELS       {fb.shape[0]}
#define MFCC_N_COEF       {dct.shape[0]}
#define MFCC_PISO_DB      ({PISO_DB:.1f}f)  /* piso fixo do Mel em dB: o top_db=80 do treino, por lote (ver exportar/comum.py) */
#define MFCC_AMIN         {a2db.amin:g}f
#define MFCC_NORM_EPS     1e-6f      /* normalizar(): (x - média) / (desvio + 1e-6) */
#define MFCC_N_PESOS_MEL  {len(pesos)}

/* Janela de Hann periódica (torch.hann_window). */
static const float MFCC_JANELA[MFCC_N_FFT] = {floats(janela)};

/* Filtro Mel m: pesos nos bins MFCC_MEL_INICIO[m] .. + MFCC_MEL_TAM[m] - 1, em sequência em MFCC_MEL_PESOS. */
static const unsigned short MFCC_MEL_INICIO[MFCC_N_MELS] = {{{", ".join(map(str, inicio))}}};
static const unsigned short MFCC_MEL_TAM[MFCC_N_MELS] = {{{", ".join(map(str, tam))}}};
static const float MFCC_MEL_PESOS[MFCC_N_PESOS_MEL] = {floats(pesos)};

/* DCT ortonormal: coef[k] = soma_m MFCC_DCT[k][m] * mel_db[m]. */
static const float MFCC_DCT[MFCC_N_COEF * MFCC_N_MELS] = {floats(dct)};
"""
    (PASTA_GERADO / "mfcc_tabelas.h").write_text(texto)
    print(f"mfcc_tabelas.h: {len(pesos)} pesos Mel, bins {min(inicio)} a {max(i + n for i, n in zip(inicio, tam)) - 1}")


def escrever_modelo(nome, modelo, hp, run, convs, head_pt):
    P = nome.upper()
    partes = [cabecalho(f"Pesos de {nome} (float32, BatchNorm dobrado nas convoluções). Treino: {run}.")]
    partes.append('#include "camadas.h"\n')
    for i, c in enumerate(convs):
        partes.append(f"static const float {nome}_w{i}[{c['peso'].numel()}] = {floats(c['peso'])};")
        partes.append(f"static const float {nome}_b{i}[{c['bias'].numel()}] = {floats(c['bias'])};")
    descr = ",\n".join(
        f"    {{ .cin = {c['cin']}, .cout = {c['cout']}, .kh = {c['kh']}, .kw = {c['kw']}, .sh = {c['sh']}, "
        f".sw = {c['sw']}, .dh = {c['dh']}, .dw = {c['dw']}, .pt = {c['pt']}, .pb = {c['pb']}, .pl = {c['pl']}, "
        f".pr = {c['pr']}, .grupos = {c['grupos']}, .peso = {nome}_w{i}, .bias = {nome}_b{i} }}"
        for i, c in enumerate(convs))
    canais = modelo.head.in_features
    n_params = sum(c["peso"].numel() + c["bias"].numel() for c in convs)
    partes.append(f"""
#define {P}_N_CONVS   {len(convs)}
#define {P}_CANAIS    {canais}      /* tamanho do embedding (saída da média global) */
#define {P}_PARAMS_REDE {n_params}  /* parâmetros das convs dobradas (sem as heads) */

static const kws_conv_t {P}_CONVS[{P}_N_CONVS] = {{
{descr}
}};

/* Head em inglês: as 35 palavras do GSC v2 (a head treinada junto com a rede). */
#define {P}_N_EN 35
static const float {nome}_head_en_w[{P}_N_EN * {P}_CANAIS] = {floats(modelo.head.weight)};
static const float {nome}_head_en_b[{P}_N_EN] = {floats(modelo.head.bias)};
""")
    if head_pt is None:
        partes.append(f"/* Sem head em português: rode exportar/treinar_head_pt.py. */\n#define {P}_N_PT 0\n"
                      f"static const float {nome}_head_pt_w[1] = {{0}};\nstatic const float {nome}_head_pt_b[1] = {{0}};\n")
    else:
        w, b = head_pt["peso"], head_pt["bias"]
        assert w.shape[1] == canais
        partes.append(f"""/* Head em português: Linear nova sobre a rede congelada, treinada no MSWC-pt (treinar_head_pt.py).
 * Acurácia na avaliação (falantes nunca vistos): {head_pt['acc_avaliacao']:.1f}%. */
#define {P}_N_PT {w.shape[0]}
static const float {nome}_head_pt_w[{P}_N_PT * {P}_CANAIS] = {floats(w)};
static const float {nome}_head_pt_b[{P}_N_PT] = {floats(b)};
""")
    (PASTA_GERADO / f"{nome}_pesos.h").write_text("\n".join(partes))
    print(f"{nome}_pesos.h: {len(convs)} convs, {n_params} params na rede, embedding {canais}, "
          f"PT {'sem head' if head_pt is None else str(len(head_pt['palavras'])) + ' palavras'}")


def escrever_rotulos(palavras_pt):
    en = classes_en()
    texto = cabecalho("Palavras de cada head, em ASCII (a fonte do OLED não tem acentos).")
    texto += f"\n#define KWS_N_EN {len(en)}\nstatic const char *const KWS_ROTULOS_EN[KWS_N_EN] = {{\n"
    texto += "".join(f'    "{w}",\n' for w in en) + "};\n"
    texto += f"\n#define KWS_N_PT {len(palavras_pt)}\n"
    if palavras_pt:
        texto += "static const char *const KWS_ROTULOS_PT[KWS_N_PT] = {\n"
        texto += "".join(f'    "{ascii_oled(w)}",   /* {w} */\n' for w in palavras_pt) + "};\n"
    else:
        texto += "static const char *const KWS_ROTULOS_PT[1] = { \"\" };\n"
    (PASTA_GERADO / "rotulos.h").write_text(texto)


def main():
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    PASTA_GERADO.mkdir(parents=True, exist_ok=True)
    palavras_pt = None
    for nome in MODELOS:
        modelo, hp, run = carregar_modelo(nome)
        assert (hp["n_mfcc"], hp["n_frames"], hp["normalizacao"]) == (10, 49, "por_amostra")
        convs = EXTRAIR[nome](modelo)
        conferir(nome, modelo, convs)
        head_pt = carregar_head_pt(nome)
        if head_pt is not None:
            if palavras_pt is not None and head_pt["palavras"] != palavras_pt:
                raise ValueError("as heads PT dos modelos têm palavras diferentes: rode treinar_head_pt.py de novo")
            palavras_pt = head_pt["palavras"]
        escrever_modelo(nome, modelo, hp, run, convs, head_pt)
        if nome == MODELOS[0]:
            escrever_mfcc(hp)
    escrever_rotulos(palavras_pt or [])
    print(f"headers em {PASTA_GERADO}")


if __name__ == "__main__":
    main()
