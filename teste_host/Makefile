# Teste do núcleo em C no Mac (clang), contra os vetores do PyTorch (exportar/vetores_teste.py).
#   make -C teste_host && (cd teste_host && ./teste)

KWS := ../components/kws
SRC := teste.c $(KWS)/kws.c $(KWS)/mfcc.c $(KWS)/fft640.c $(KWS)/camadas.c
CFLAGS := -std=c11 -O2 -Wall -Wextra -I$(KWS)/include -I$(KWS)

teste: $(SRC) $(wildcard $(KWS)/*.h $(KWS)/include/*.h $(KWS)/gerado/*.h)
	$(CC) $(CFLAGS) -o $@ $(SRC) -lm

clean:
	rm -f teste

.PHONY: clean
