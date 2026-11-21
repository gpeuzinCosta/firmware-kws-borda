"""Head em português para cada modelo: a rede treinada no GSC fica CONGELADA e só uma Linear nova é treinada no
MSWC-pt, como no few-shot da §10.4 do relatório (treino/few_shot.py), mas com todos os clipes de treino (não só K)
para a demonstração ficar melhor.

- Divisão por falante (dados/mswc_pt.py:dividir_por_falante): ninguém do treino aparece na avaliação.
- Como a rede está congelada (BatchNorm em modo eval), treinar a head sobre os embeddings (saída da média global)
  é o mesmo que treinar a RedeCongelada inteira, só que muito mais rápido.
- Os áudios passam por int16 antes do MFCC, como no gravador (o microfone entrega int16), e o MFCC é o do
  gravador (comum.mfcc_gravador: um áudio por vez, piso fixo).

Saída: exportar/heads/head_pt_<modelo>.pt (pesos da head e a acurácia na avaliação) e
exportar/heads/avaliacao_pt.npz (uma amostra da avaliação, para os vetores de teste do C).
Depois rode exportar.py de novo.

Pré-requisito: MSWC-pt extraído em ~/projetos/kws-borda/data/mswc
    (cd ~/projetos/kws-borda && .venv/bin/python dados/baixar.py --dataset mswc --destino data --arquivos <Drive>/downloads)

    ~/projetos/kws-borda/.venv/bin/python exportar/treinar_head_pt.py
"""
import numpy as np
import torch

from comum import DADOS_KWS, MODELOS, PASTA_HEADS, ascii_oled, carregar_modelo, mfcc_gravador

K_MAX = 20              #só entram palavras com pelo menos K_MAX clipes de treino (o critério da §10.4)
MAX_TREINO = 400        #clipes de treino por palavra (no máximo 3 por falante)
MAX_AVALIACAO = 100
EPOCAS = 200
LR = 1e-2
WEIGHT_DECAY = 1e-2
AMOSTRA_VETORES = 10    #áudios de avaliação por palavra guardados para o teste do C


def carregar_audios():
    from dados.mfcc import ler_audio
    from dados.mswc_pt import carregar_splits, dividir_por_falante

    pasta = DADOS_KWS / "mswc"
    splits = carregar_splits(pasta)
    palavras, amostras = dividir_por_falante(splits, pasta / "pt" / "clips", k_max=K_MAX,
                                             max_treino=MAX_TREINO, max_avaliacao=MAX_AVALIACAO)
    print("palavras:", palavras)
    print(amostras.groupby(["WORD", "conjunto"]).size().unstack())

    audios = []
    for c in amostras["caminho"]:
        w = ler_audio(c).numpy()
        audios.append(np.clip(np.round(w * 32768.0), -32768, 32767).astype(np.int16))
    rotulos = np.array([palavras.index(w) for w in amostras["WORD"]], np.int64)
    eh_treino = (amostras["conjunto"] == "treino").to_numpy()
    return palavras, np.stack(audios), rotulos, eh_treino


def embeddings(modelo, hp, audios):
    head = modelo.head
    modelo.head = torch.nn.Identity()
    try:
        with torch.no_grad():
            x = mfcc_gravador(torch.from_numpy(audios.astype(np.float32) / 32768.0), hp)
            return torch.cat([modelo(x[i:i + 512]) for i in range(0, len(x), 512)])
    finally:
        modelo.head = head


def treinar_head(emb, y, eh_treino, n_classes, seed=0):
    torch.manual_seed(seed)
    head = torch.nn.Linear(emb.shape[1], n_classes)
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCAS)
    xt, yt = emb[eh_treino], torch.from_numpy(y[eh_treino])
    for _ in range(EPOCAS):
        perm = torch.randperm(len(xt))
        for i in range(0, len(xt), 64):
            idx = perm[i:i + 64]
            perda = torch.nn.functional.cross_entropy(head(xt[idx]), yt[idx])
            opt.zero_grad(); perda.backward(); opt.step()
        sched.step()
    with torch.no_grad():
        pred = head(emb[~eh_treino]).argmax(1).numpy()
    acc = (pred == y[~eh_treino]).mean() * 100
    return head, acc, pred


def main():
    PASTA_HEADS.mkdir(parents=True, exist_ok=True)
    palavras, audios, y, eh_treino = carregar_audios()
    print(f"{eh_treino.sum()} clipes de treino, {(~eh_treino).sum()} de avaliação, {len(palavras)} palavras "
          f"(acaso: {100 / len(palavras):.1f}%)")

    for nome in MODELOS:
        modelo, hp, run = carregar_modelo(nome)
        emb = embeddings(modelo, hp, audios)
        head, acc, pred = treinar_head(emb, y, eh_treino, len(palavras))
        print(f"{nome}: acurácia na avaliação (falantes nunca vistos) {acc:.1f}%")
        yv = y[~eh_treino]
        for i, w in enumerate(palavras):
            print(f"    {ascii_oled(w):8s} {100 * (pred[yv == i] == i).mean():5.1f}%")
        torch.save(dict(palavras=palavras, peso=head.weight.detach().clone(), bias=head.bias.detach().clone(),
                        acc_avaliacao=float(acc), n_treino=int(eh_treino.sum()), run=run),
                   PASTA_HEADS / f"head_pt_{nome}.pt")

    #Amostra da avaliação para os vetores de teste do C
    idx = np.concatenate([np.where(~eh_treino & (y == i))[0][:AMOSTRA_VETORES] for i in range(len(palavras))])
    np.savez(PASTA_HEADS / "avaliacao_pt.npz", audios=audios[idx], rotulos=y[idx].astype(np.int32))
    print(f"heads em {PASTA_HEADS}; agora rode exportar.py e vetores_teste.py")


if __name__ == "__main__":
    main()
