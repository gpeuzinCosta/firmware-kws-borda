"""Vetores de teste para o teste no Mac (teste_host/): o C tem de reproduzir o PyTorch.

Gera em teste_host/vetores/ (arquivos binários crus, float32/int16/int32 little-endian):
- fft_x.f32 / fft_p.f32: sinais aleatórios de 640 amostras e o |rfft|^2 do numpy;
- <conjunto>_audio.i16: áudios de 1 s (16000 amostras int16);
- <conjunto>_rotulo.i32: a palavra certa;
- <conjunto>_mfcc.f32: MFCC normalizado como o gravador calcula (comum.mfcc_gravador), 10 x 49 por áudio;
- <conjunto>_<modelo>_<idioma>.f32: logits do modelo ORIGINAL (não o dobrado).
Conjuntos: 'gsc' (amostra do teste oficial do GSC, inglês) e 'pt' (avaliação do MSWC-pt, se já houver head PT).

O GSC sai do .tar.gz do Drive: lê o arquivo inteiro uma vez (2,3 GB) e extrai só N_POR_PALAVRA áudios de teste
por palavra para ~/projetos/kws-borda/data/gsc_amostra_teste/.

    ~/projetos/kws-borda/.venv/bin/python exportar/vetores_teste.py
"""
import hashlib
import re
import tarfile

import numpy as np
import soundfile as sf
import torch

from comum import (DADOS_KWS, DRIVE_KWS, MODELOS, PASTA_VETORES, carregar_head_pt, carregar_modelo,
                   classes_en, mfcc_gravador)

N_POR_PALAVRA = 10
PASTA_GSC = DADOS_KWS / "gsc_amostra_teste"


def eh_teste_pelo_hash(nome_arquivo):
    """A regra do README do GSC para o split (o testing_list.txt foi gerado por ela): 10% validação, 10% teste."""
    base = re.sub(r"_nohash_.*$", "", nome_arquivo)
    h = int(hashlib.sha1(base.encode()).hexdigest(), 16)
    pct = (h % (2 ** 27)) * (100.0 / (2 ** 27 - 1))
    return 10 <= pct < 20


def extrair_gsc():
    lista = PASTA_GSC / "testing_list.txt"
    if lista.exists():
        return
    PASTA_GSC.mkdir(parents=True, exist_ok=True)
    classes = set(classes_en())
    contagem = {c: 0 for c in classes}
    tar = DRIVE_KWS / "downloads" / "speech_commands_v0.02.tar.gz"
    print(f"lendo {tar} (2,3 GB, alguns minutos)...")
    with tarfile.open(tar, "r:gz") as t:
        for m in t:
            nome = m.name[2:] if m.name.startswith("./") else m.name
            if nome == "testing_list.txt":
                lista.write_bytes(t.extractfile(m).read())
                continue
            partes = nome.split("/")
            if len(partes) != 2 or partes[0] not in classes or not partes[1].endswith(".wav"):
                continue
            if contagem[partes[0]] >= 3 * N_POR_PALAVRA or not eh_teste_pelo_hash(partes[1]):
                continue
            destino = PASTA_GSC / nome
            destino.parent.mkdir(exist_ok=True)
            destino.write_bytes(t.extractfile(m).read())
            contagem[partes[0]] += 1
    print("extraído:", sum(contagem.values()), "candidatos")


def amostra_gsc():
    """(áudios int16 (N, 16000), rótulos): N_POR_PALAVRA de cada palavra, todos do testing_list oficial."""
    extrair_gsc()
    oficiais = set((PASTA_GSC / "testing_list.txt").read_text().split())
    audios, rotulos = [], []
    for i, c in enumerate(classes_en()):
        arquivos = sorted(p for p in (PASTA_GSC / c).glob("*.wav") if f"{c}/{p.name}" in oficiais)
        assert len(arquivos) >= N_POR_PALAVRA, f"poucos áudios de teste de {c}"
        for p in arquivos[:N_POR_PALAVRA]:
            x, sr = sf.read(p, dtype="int16")
            assert sr == 16000 and x.ndim == 1
            y = np.zeros(16000, np.int16)                #fix_length: completa com zeros no fim (GSC <= 1 s)
            y[:min(len(x), 16000)] = x[:16000]
            audios.append(y); rotulos.append(i)
    return np.stack(audios), np.array(rotulos, np.int32)


def referencia(audios, idioma):
    """MFCC normalizado e logits de cada modelo, pelo pipeline do treino em PyTorch."""
    from treino.few_shot import RedeCongelada

    wav = torch.from_numpy(audios.astype(np.float32) / 32768.0)      #igual ao soundfile em float32
    saidas = {}
    for nome in MODELOS:
        modelo, hp, _ = carregar_modelo(nome)
        x = mfcc_gravador(wav, hp)
        saidas["mfcc"] = x.numpy()
        if idioma == "pt":
            head = carregar_head_pt(nome)
            if head is None:
                continue
            rede = RedeCongelada(modelo, len(head["palavras"]))
            rede.rede.head.weight.data[:] = head["peso"]; rede.rede.head.bias.data[:] = head["bias"]
            modelo = rede.eval()
        with torch.no_grad():
            saidas[f"{nome}_{idioma}"] = modelo(x).numpy()
    return saidas


def gravar(prefixo, audios, rotulos, saidas):
    audios.astype("<i2").tofile(PASTA_VETORES / f"{prefixo}_audio.i16")
    rotulos.astype("<i4").tofile(PASTA_VETORES / f"{prefixo}_rotulo.i32")
    for k, v in saidas.items():
        v.astype("<f4").tofile(PASTA_VETORES / f"{prefixo}_{k}.f32")
    print(f"{prefixo}: {len(audios)} áudios |", ", ".join(saidas))


def main():
    PASTA_VETORES.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    x = rng.standard_normal((20, 640)).astype(np.float32) * 0.3
    (np.abs(np.fft.rfft(x.astype(np.float64), axis=1)) ** 2).astype("<f4").tofile(PASTA_VETORES / "fft_p.f32")
    x.astype("<f4").tofile(PASTA_VETORES / "fft_x.f32")

    audios, rotulos = amostra_gsc()
    gravar("gsc", audios, rotulos, referencia(audios, "en"))

    pt = PASTA_VETORES.parent.parent / "exportar" / "heads" / "avaliacao_pt.npz"
    if pt.exists():
        d = np.load(pt)
        gravar("pt", d["audios"], d["rotulos"], referencia(d["audios"], "pt"))
    else:
        print("sem avaliacao_pt.npz: rode treinar_head_pt.py para os vetores em português")


if __name__ == "__main__":
    main()
