"""Partes comuns do exportador: onde estão o repositório kws-borda e os treinos, e como carregar um modelo.

Rode sempre com o Python do kws-borda (ele tem torch e torchaudio):

    ~/projetos/kws-borda/.venv/bin/python exportar/exportar.py

Os caminhos locais ficam no arquivo .env na raiz do repositório (fora do git; modelo em .env.exemplo):
- KWS_REPO:  o clone do kws-borda (código dos modelos, MFCC e dados);
- KWS_DRIVE: a pasta com runs/, cache/ e downloads/ do treino (no Colab, MyDrive/kws-borda);
- KWS_DADOS: onde ficam os dados extraídos (opcional; padrão: $KWS_REPO/data).
Variáveis já definidas no ambiente têm prioridade sobre o .env.
"""
import os
import sys
import unicodedata
from pathlib import Path

import torch

PASTA_EXPORTAR = Path(__file__).resolve().parent
PASTA_RAIZ = PASTA_EXPORTAR.parent                                  #raiz do repositório firmware-kws-borda
PASTA_GERADO = PASTA_RAIZ / "components" / "kws" / "gerado"         #headers C gerados
PASTA_HEADS = PASTA_EXPORTAR / "heads"                              #heads PT treinadas (pequenas, vão no git)
PASTA_VETORES = PASTA_RAIZ / "teste_host" / "vetores"               #vetores de teste (gerados, fora do git)


def ler_env(arquivo=PASTA_RAIZ / ".env"):
    """Lê CHAVE=valor do .env para os.environ (sem sobrescrever o que já existe). Aceita aspas e ~."""
    if not arquivo.exists():
        return
    for linha in arquivo.read_text().splitlines():
        linha = linha.strip()
        if not linha or linha.startswith("#") or "=" not in linha:
            continue
        chave, valor = linha.split("=", 1)
        os.environ.setdefault(chave.strip(), valor.strip().strip("'\""))


def caminho_env(chave, padrao=None):
    valor = os.environ.get(chave)
    if not valor:
        if padrao is not None:
            return padrao
        raise SystemExit(f"defina {chave} no arquivo .env (copie .env.exemplo para .env)")
    return Path(valor).expanduser()


ler_env()
REPO_KWS = caminho_env("KWS_REPO")
DRIVE_KWS = caminho_env("KWS_DRIVE")
DADOS_KWS = caminho_env("KWS_DADOS", REPO_KWS / "data")             #dados extraídos (fora de qualquer git)

#Os dois modelos do firmware, com a seed 0 da configuração atual de HP_MODELOS (os vencedores da §8)
MODELOS = ["dscnn", "resnet"]
SEED = 0

#Piso fixo do Mel em dB, no lugar do top_db=80 do torchaudio.
#No treino (dados/speech_commands.py:extrair_mfcc) o MFCC foi calculado em lotes de 256 áudios, e o
#amplitude_to_DB do torchaudio aplica o top_db sobre o máximo do LOTE inteiro, não de cada áudio. Em todo lote de
#256 do GSC há algum áudio no limite da escala, então esse máximo fica quase constante (40,3 a 42,7 dB, mediana
#42,7, medido numa amostra do teste) e o piso efetivo foi 42,7 - 80 = -37,3 dB. O gravador processa um comando por
#vez, então usa esse piso fixo: reproduz o treino sem depender de lote.
PISO_DB = -37.3


def mfcc_gravador(wav, hp):
    """MFCC normalizado como o gravador calcula: um áudio por vez, com o piso fixo PISO_DB.

    wav: (N, 16000) float em [-1, 1]. Usa o espectrograma Mel e a DCT do próprio torchaudio do treino.
    """
    import torchaudio
    from dados import ExtratorMFCC, normalizar
    t = ExtratorMFCC.de_hparams(hp).mfcc
    with torch.no_grad():
        mel = t.MelSpectrogram(wav)                                           #(N, n_mels, frames)
        db = torchaudio.functional.amplitude_to_DB(mel, 10.0, 1e-10, 0.0, None).clamp(min=PISO_DB)
        x = torch.matmul(db.transpose(-1, -2), t.dct_mat).transpose(-1, -2)  #(N, n_mfcc, frames)
        return normalizar(x)

if str(REPO_KWS) not in sys.path:
    sys.path.insert(0, str(REPO_KWS))


def carregar_modelo(nome):
    """(modelo em modo eval, hp, pasta do run) do treino salvo no Drive para (nome, SEED)."""
    from configs import carregar_config
    from modelos import criar_modelo
    from treino.loop import carregar_run

    hp = carregar_config(nome)
    run = carregar_run(DRIVE_KWS / "runs", hp, SEED)
    if run is None:
        raise FileNotFoundError(f"nenhum treino de {nome} (seed {SEED}) em {DRIVE_KWS / 'runs'}")
    estado, hist, _ = run
    modelo = criar_modelo(nome, hp)
    modelo.load_state_dict(estado)
    return modelo.eval(), hp, Path(hist.get("pasta", "?")).name


def classes_en():
    from dados import classes_da_tarefa
    return list(classes_da_tarefa("gsc35"))


def ascii_oled(palavra):
    """'não' -> 'nao': a fonte 5x7 do OLED só tem ASCII."""
    return unicodedata.normalize("NFKD", palavra).encode("ascii", "ignore").decode()


def carregar_head_pt(nome):
    """A head PT treinada por treinar_head_pt.py, ou None se ainda não existir."""
    arquivo = PASTA_HEADS / f"head_pt_{nome}.pt"
    return torch.load(arquivo, map_location="cpu") if arquivo.exists() else None
