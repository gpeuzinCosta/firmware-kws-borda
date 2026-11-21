# Núcleo do reconhecimento em C puro (também compila no Mac: teste_host/).
idf_component_register(SRCS "kws.c" "mfcc.c" "fft640.c" "camadas.c"
                       INCLUDE_DIRS "include"
                       PRIV_INCLUDE_DIRS ".")
# Laços de convolução e FFT: otimização para velocidade, não tamanho
target_compile_options(${COMPONENT_LIB} PRIVATE -O2)
