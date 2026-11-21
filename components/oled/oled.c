/*
 * Driver do OLED SSD1306 usando o esp_lcd (painel SSD1306 nativo do IDF)
 * sobre o driver I2C novo (driver/i2c_master.h).
 */
#include "oled.h"

#include <string.h>
#include "driver/gpio.h"
#include "driver/i2c_master.h"
#include "esp_check.h"
#include "esp_lcd_panel_io.h"
#include "esp_lcd_panel_ops.h"
#include "esp_lcd_panel_ssd1306.h"
#include "esp_log.h"

static const char *TAG = "oled";

extern const uint8_t fonte5x7[95][5];

static i2c_master_bus_handle_t s_barramento;
static esp_lcd_panel_io_handle_t s_io;
static esp_lcd_panel_handle_t s_painel;

/* Framebuffer no formato do SSD1306: 8 "páginas" de 8 linhas; cada byte é
 * uma coluna de 8 pixels (bit 0 em cima). 128 x 64 / 8 = 1024 bytes. */
static uint8_t s_fb[OLED_LARGURA * OLED_ALTURA / 8];

/* ---------- Barramento I2C ---------- */

/* Com pull-up, SDA e SCL em repouso devem ler 1. Se algum lê 0, algo está
 * segurando a linha: módulo sem alimentação, VCC/GND trocados ou fio errado. */
static void verificar_linhas(void)
{
    const int pinos[2] = { OLED_PINO_SDA, OLED_PINO_SCL };
    const char *nomes[2] = { "SDA", "SCL" };
    for (int i = 0; i < 2; i++) {
        gpio_reset_pin(pinos[i]);
        gpio_set_direction(pinos[i], GPIO_MODE_INPUT);
        gpio_pullup_en(pinos[i]);
    }
    esp_rom_delay_us(100);
    for (int i = 0; i < 2; i++) {
        if (gpio_get_level(pinos[i]) == 0) {
            ESP_LOGE(TAG, "Linha %s (GPIO%d) esta em 0 com o barramento parado: deveria ser 1.",
                     nomes[i], pinos[i]);
            ESP_LOGE(TAG, "  Confira: VCC do OLED no 3,3V e GND no GND (sem alimentacao o modulo "
                          "puxa as linhas para 0); fio da %s no GPIO%d; nada encostando no GND.",
                     nomes[i], pinos[i]);
        }
    }

    /* Teste de curto: força SCL em 0 e vê se SDA vai junto. Se for, os dois
     * fios estão ligados entre si (mesma fileira da protoboard, por exemplo). */
    if (gpio_get_level(OLED_PINO_SDA) == 1 && gpio_get_level(OLED_PINO_SCL) == 1) {
        gpio_set_direction(OLED_PINO_SCL, GPIO_MODE_OUTPUT_OD);
        gpio_set_level(OLED_PINO_SCL, 0);
        esp_rom_delay_us(100);
        int sda = gpio_get_level(OLED_PINO_SDA);
        gpio_set_level(OLED_PINO_SCL, 1);
        gpio_set_direction(OLED_PINO_SCL, GPIO_MODE_INPUT);
        if (sda == 0) {
            /* Duas causas dão esse sintoma. Um OLED SEM ALIMENTAÇÃO também liga SDA e
             * SCL: os pull-ups e diodos de proteção do módulo vão ao VCC dele, que fica
             * solto e acaba servindo de ponte entre as duas linhas. */
            ESP_LOGE(TAG, "SDA (GPIO%d) e SCL (GPIO%d) estao ligados: ao puxar SCL para 0, SDA foi junto.",
                     OLED_PINO_SDA, OLED_PINO_SCL);
            ESP_LOGE(TAG, "  Causa 1 (mais comum): OLED sem alimentacao. Confira VCC do display no 3V3 "
                          "e GND no GND do ESP32 (leia a serigrafia: ha modulos com GND e VCC trocados).");
            ESP_LOGE(TAG, "  Causa 2: fios de SDA e SCL na mesma fileira da protoboard ou ponte de solda "
                          "entre os pinos do display.");
        } else {
            ESP_LOGI(TAG, "Linhas SDA/SCL em repouso OK e sem curto entre si.");
        }
    }
}

static esp_err_t criar_barramento(void)
{
    if (s_barramento) {
        return ESP_OK;
    }
    verificar_linhas();
    i2c_master_bus_config_t cfg = {
        .i2c_port = I2C_NUM_0,
        .sda_io_num = OLED_PINO_SDA,
        .scl_io_num = OLED_PINO_SCL,
        .clk_source = I2C_CLK_SRC_DEFAULT,
        .glitch_ignore_cnt = 7,
        /* Pull-ups internos (~45 kΩ) ligados por garantia: se o módulo já tiver
         * os dele (o comum é 4,7 kΩ ou 10 kΩ), os dois em paralelo não atrapalham. */
        .flags.enable_internal_pullup = true,
    };
    esp_err_t err = i2c_new_master_bus(&cfg, &s_barramento);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Nao consegui criar o barramento I2C (SDA=%d, SCL=%d): %s",
                 OLED_PINO_SDA, OLED_PINO_SCL, esp_err_to_name(err));
    }
    return err;
}

int oled_scanner(void)
{
    if (criar_barramento() != ESP_OK) {
        return 0;
    }
    int achados = 0;
    printf("Scanner I2C (SDA=GPIO%d, SCL=GPIO%d):\n", OLED_PINO_SDA, OLED_PINO_SCL);
    for (uint8_t end = 0x08; end < 0x78; end++) {
        if (i2c_master_probe(s_barramento, end, 50) == ESP_OK) {
            printf("  -> dispositivo encontrado em 0x%02X\n", end);
            achados++;
        }
    }
    if (achados == 0) {
        printf("  nenhum dispositivo respondeu.\n");
    } else if (achados > 10) {
        /* Um display real responde em 1 endereço. "Todos respondem" = SDA presa em 0,
         * que o I2C interpreta como confirmação (ACK). */
        printf("  %d enderecos 'responderam': isso nao e real. A linha SDA esta presa em 0\n"
               "  (OLED sem alimentacao, VCC/GND trocados, ou SDA/SCL em pinos errados).\n",
               achados);
        return -1;
    }
    return achados;
}

/* ---------- Painel ---------- */

esp_err_t oled_iniciar(void)
{
    if (s_painel) {
        return ESP_OK;
    }
    ESP_RETURN_ON_ERROR(criar_barramento(), TAG, "barramento I2C");

    /* Antes de configurar o painel, confere se alguém responde em 0x3C. */
    esp_err_t err = i2c_master_probe(s_barramento, OLED_ENDERECO, 100);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "OLED nao respondeu no endereco 0x%02X (%s)", OLED_ENDERECO, esp_err_to_name(err));
        return ESP_ERR_NOT_FOUND;
    }

    /* Camada de E/S: cada transferência começa com um byte de controle
     * cujo bit 6 diz se é comando (0) ou dado (1). */
    esp_lcd_panel_io_i2c_config_t io_cfg = {
        .dev_addr = OLED_ENDERECO,
        .scl_speed_hz = 400 * 1000,
        .control_phase_bytes = 1,
        .dc_bit_offset = 6,
        .lcd_cmd_bits = 8,
        .lcd_param_bits = 8,
    };
    ESP_RETURN_ON_ERROR(esp_lcd_new_panel_io_i2c(s_barramento, &io_cfg, &s_io), TAG, "panel io");

    esp_lcd_panel_ssd1306_config_t ssd_cfg = { .height = OLED_ALTURA };
    esp_lcd_panel_dev_config_t painel_cfg = {
        .bits_per_pixel = 1,
        .reset_gpio_num = -1,          /* módulo sem pino de reset */
        .vendor_config = &ssd_cfg,
    };
    ESP_RETURN_ON_ERROR(esp_lcd_new_panel_ssd1306(s_io, &painel_cfg, &s_painel), TAG, "painel ssd1306");
    ESP_RETURN_ON_ERROR(esp_lcd_panel_reset(s_painel), TAG, "reset");
    ESP_RETURN_ON_ERROR(esp_lcd_panel_init(s_painel), TAG, "init");
#if OLED_GIRAR_180
    /* O init do IDF deixa a imagem invertida na maioria dos módulos 0,96". */
    ESP_RETURN_ON_ERROR(esp_lcd_panel_mirror(s_painel, true, true), TAG, "espelhar");
#endif
    ESP_RETURN_ON_ERROR(esp_lcd_panel_disp_on_off(s_painel, true), TAG, "ligar display");

    oled_limpar();
    ESP_LOGI(TAG, "OLED pronto em 0x%02X", OLED_ENDERECO);
    return oled_atualizar();
}

/* ---------- Desenho ---------- */

void oled_limpar(void)
{
    memset(s_fb, 0, sizeof(s_fb));
}

static void pixel(int x, int y)
{
    if (x < 0 || x >= OLED_LARGURA || y < 0 || y >= OLED_ALTURA) {
        return;
    }
    s_fb[x + (y / 8) * OLED_LARGURA] |= (uint8_t)(1u << (y % 8));
}

void oled_retangulo(int x, int y, int largura, int altura, bool cheio)
{
    for (int dx = 0; dx < largura; dx++) {
        for (int dy = 0; dy < altura; dy++) {
            if (cheio || dx == 0 || dy == 0 || dx == largura - 1 || dy == altura - 1) {
                pixel(x + dx, y + dy);
            }
        }
    }
}

void oled_texto(int x, int y, const char *texto, int escala)
{
    if (escala < 1) {
        escala = 1;
    }
    for (const char *p = texto; *p; p++) {
        unsigned char c = (unsigned char)*p;
        if (c < 0x20 || c > 0x7E) {
            c = '?';                       /* acentos e afins viram '?' */
        }
        const uint8_t *glifo = fonte5x7[c - 0x20];
        for (int col = 0; col < 5; col++) {
            for (int lin = 0; lin < 8; lin++) {
                if (glifo[col] & (1u << lin)) {
                    for (int dx = 0; dx < escala; dx++) {
                        for (int dy = 0; dy < escala; dy++) {
                            pixel(x + col * escala + dx, y + lin * escala + dy);
                        }
                    }
                }
            }
        }
        x += 6 * escala;                   /* 5 colunas + 1 de espaço */
    }
}

esp_err_t oled_atualizar(void)
{
    if (!s_painel) {
        return ESP_ERR_INVALID_STATE;
    }
    return esp_lcd_panel_draw_bitmap(s_painel, 0, 0, OLED_LARGURA, OLED_ALTURA, s_fb);
}

esp_err_t oled_desligar(void)
{
    if (!s_painel) {
        return ESP_OK;
    }
    return esp_lcd_panel_disp_on_off(s_painel, false);
}
