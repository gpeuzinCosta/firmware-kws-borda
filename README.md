# firmware-kws-borda

Demonstração do trabalho de Deep Learning [kws-borda](https://github.com/gpeuzinCosta/kws-borda) no gravador de
voz: você aperta o botão, fala um comando em inglês ou em português e o ESP32 mostra no OLED a palavra que
reconheceu. Tudo roda no próprio chip, sem Wi-Fi e sem computador: o MFCC, a rede e a última camada, em C puro.

| Ação | Botão |
|---|---|
| Ouvir um comando (2 s, depois de 150 ms) | clique curto |
| Trocar o modo: DS-CNN inglês, DS-CNN português, ResNet inglês, ResNet português | segurar 0,8 s |

Na tela: a palavra (com "?" se a confiança ficar abaixo de 50%), a confiança, a segunda palavra mais provável e
o tempo do MFCC e da rede **medidos no ESP32**. O monitor serial mostra as 3 mais prováveis.

## Modelos

Os vencedores da §8 do relatório (seed 0, a configuração de `HP_MODELOS`), com entrada MFCC `hello_edge`
(10 x 49) e normalização por amostra:

| Modelo | Parâmetros da rede | Acc teste GSC (35 palavras) | Head em português (MSWC-pt) |
|---|---|---|---|
| DS-CNN, 48 canais | 15.443 | 91,2% | 83,4% |
| ResNet res15-narrow | 43.640 | 92,3% | 78,1% |

A coluna do português é a acurácia em falantes nunca vistos (549 clipes de avaliação, 13 palavras, acaso de 7,7%).

- **Inglês:** a head treinada com a rede, nas 35 palavras do Google Speech Commands v2
  (yes, no, up, down, left, right, on, off, stop, go, zero a nine, e as auxiliares).
- **Português:** a rede fica congelada e só uma camada Linear nova é treinada no MSWC-pt (o método da §10.4,
  com todos os clipes de treino, 1.746, em vez de K = 20). As 13 palavras com clipes suficientes: **sim, não,
  vai, baixo, zero, dois, três, quatro, cinco, seis, sete, oito, nove** ("um" não existe no MSWC, e "cima",
  "esquerda", "direita", "parar", "ligar" e "desligar" têm poucos clipes). No OLED saem sem acento.
- **float32, sem quantização.** O C reproduz o PyTorch quase exatamente e os pesos cabem folgados na flash
  (62 KB a DS-CNN, 175 KB a ResNet). O "KB int8" do relatório continua sendo uma estimativa.

## Estrutura

```
components/kws/        o reconhecimento, em C puro (compila também no Mac)
  kws.c                as duas redes e a API (include/kws.h)
  mfcc.c, fft640.c     MFCC idêntico ao do treino (torchaudio) e a FFT de 640 pontos (5 x 128)
  camadas.c            convolução 2D genérica (stride, padding, dilatação, depthwise), linear, média global
  gerado/              pesos e tabelas do MFCC, gerados por exportar/exportar.py (não edite à mão)
components/oled, mic, botao, placa
                       drivers copiados do firmware-gravador-voz (commit 6d54773); só o buffer do mic diminuiu
main/main.c            telas, botão e medição dos tempos
exportar/              PyTorch -> C (Python, usa o repositório kws-borda)
teste_host/            teste no Mac: o C contra o PyTorch
```

## Gravar na placa

Mesmo hardware e mesma pinagem do gravador (`components/placa/include/placa_devkit_wroom32e.h`): ESP32-WROOM-32E
DevKit, INMP441 e OLED SSD1306. O botão é o BOOT.

```zsh
. ~/esp/ativar.sh
cd ~/projetos/firmware-kws-borda
idf.py set-target esp32          # só na primeira vez
idf.py build
idf.py -p /dev/cu.usbserial-XXXX flash monitor
```

Pare o servidor do gravador antes (só um programa usa a porta). Para voltar ao firmware do gravador, grave o
`firmware-gravador-voz` de novo.

## Refazer os pesos (Python)

Usa o ambiente do kws-borda (`~/projetos/kws-borda/.venv`, com torch 2.2.2 e numpy < 2) e os treinos salvos
no Google Drive (`My Drive/kws-borda/runs`). Os caminhos da sua máquina ficam num `.env` (fora do git):

```zsh
cp .env.exemplo .env      # e preencha KWS_REPO e KWS_DRIVE
PY=~/projetos/kws-borda/.venv/bin/python
$PY exportar/treinar_head_pt.py   # head em português (precisa do MSWC-pt extraído em kws-borda/data/mswc)
$PY exportar/exportar.py          # headers em components/kws/gerado/; confere o BN dobrado contra o modelo
$PY exportar/vetores_teste.py     # vetores para o teste no Mac (amostra do teste do GSC e do MSWC-pt)
make -C teste_host && (cd teste_host && ./teste)
```

O teste no Mac compara com o PyTorch a FFT, o MFCC normalizado e os logits das duas redes, nos dois idiomas, e
falha se a palavra prevista mudar em mais de 0,5% dos áudios. Resultado atual, em 350 áudios do teste do GSC
(10 por palavra) e 130 da avaliação do MSWC-pt: erro máximo de 4e-5 no MFCC e de 5e-5 nos logits, e a mesma
palavra que o PyTorch em 100% dos áudios, nos 4 modos.

### O piso do MFCC

O MFCC do torchaudio corta o espectrograma Mel 80 dB abaixo do máximo (`top_db=80`). No treino do kws-borda o MFCC
foi calculado em lotes de 256 áudios (`dados/speech_commands.py:extrair_mfcc`), e o torchaudio aplica esse corte
sobre o máximo do **lote**, não de cada áudio. Em todo lote do GSC há algum áudio no limite da escala, então o
máximo do lote é quase constante (cerca de 42,7 dB) e o piso efetivo foi um valor fixo de cerca de -37,3 dB. O
gravador processa um comando por vez, então usa esse piso fixo (`PISO_DB` em `exportar/comum.py`). Na amostra do
teste, as variantes (piso por áudio, por lote, fixo) mudam a acurácia em menos de 1 p.p.

## Ajustes

- **Diz "(silencio)" mesmo falando:** fale a um palmo do microfone, ou baixe `SILENCIO_DBFS` em `main/main.c`.
- **Áudio saturando:** `MIC_DESLOCAMENTO` em `components/mic/mic.c` (14 = +12 dB).
- **Confiança mínima:** `CONFIANCA_MINIMA` em `main/main.c`.
- **Imagem de cabeça para baixo:** `OLED_GIRAR_180` em `components/oled/include/oled.h`.

O modelo não tem classe de silêncio nem de "palavra desconhecida" (a tarefa é `gsc35`): qualquer som vira uma
das palavras. Por isso o "?" de baixa confiança e o limiar de silêncio.

## Créditos e licenças dos dados

- **Google Speech Commands v2** (Warden, 2018), CC-BY 4.0: treino dos modelos e a amostra do teste usada em
  `teste_host/` (os áudios não vão no repositório; `exportar/vetores_teste.py` extrai do arquivo oficial).
- **Multilingual Spoken Words Corpus (MSWC), português** (Mazumder et al., 2021), CC-BY 4.0: treino da head em
  português. `exportar/heads/avaliacao_pt.npz` traz 130 clipes de avaliação do MSWC (convertidos para 16 kHz,
  int16) para o teste do C.
- Os drivers em `components/oled`, `mic`, `botao` e `placa` vêm do firmware do gravador de voz do mesmo autor
  (no `mic`, só o buffer diminuiu de 64 KB para 16 KB, para sobrar memória para a rede).
