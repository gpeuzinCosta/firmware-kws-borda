idf_component_register(SRCS "oled.c" "fonte5x7.c"
                       INCLUDE_DIRS "include"
                       REQUIRES placa esp_driver_i2c esp_lcd)
