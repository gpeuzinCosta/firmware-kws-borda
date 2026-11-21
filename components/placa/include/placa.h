/*
 * Pinagem da placa em uso, escolhida pelo alvo do IDF (idf.py set-target):
 *
 *   esp32     placa_devkit_wroom32e.h   breadboard: ESP32-WROOM-32E DevKit
 *   esp32s3   placa_pcb_s3.h            PCB final: ESP32-S3-WROOM-1 (ainda não testada)
 */
#pragma once

#include "sdkconfig.h"

#if CONFIG_IDF_TARGET_ESP32S3
#include "placa_pcb_s3.h"
#else
#include "placa_devkit_wroom32e.h"
#endif
