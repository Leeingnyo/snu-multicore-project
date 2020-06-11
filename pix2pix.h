#pragma once

#include <cstddef>
#include <cstdint>

void pix2pix_init();

void pix2pix(uint8_t *input_buf, float *weight_buf, uint8_t *output_buf, size_t num_image);
