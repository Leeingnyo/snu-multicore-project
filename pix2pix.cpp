#include "pix2pix.h"

#include "util.h"

#include <immintrin.h> // include vector
#include <omp.h>

#include <string>
#include <map>
#include <cmath>

// #define SHOW_TIME
#define START double st = get_time();
#define END(x) double et = get_time(); printf("\n%s! (%lf s)", x, et - st);

#define VECTOR_SIZE 256
#define TYPE float
#if VECTOR_SIZE == 256
  #define VECTOR_TYPE __m256
  #define VECTOR_LOAD(x) _mm256_loadu_ps(x)
  #define VECTOR_STORE(x, y) _mm256_storeu_ps((x), (y))
  #define VECTOR_ADD(x, y) _mm256_add_ps((x), (y))
  #define VECTOR_MUL(x, y) _mm256_mul_ps((x), (y))
  #define VECTOR_SUB(x, y) _mm256_sub_ps((x), (y))
  #define VECTOR_DIV(x, y) _mm256_div_ps((x), (y))
  #define VECTOR_SET1(x) _mm256_set1_ps((x))
#elif VECTOR_TYPE == 128
  #define VECTOR_TYPE __m128
  #define VECTOR_LOAD(x) _mm_loadu_ps(x)
  #define VECTOR_STORE(x, y) _mm_storeu_ps((x), (y))
  #define VECTOR_ADD(x, y) _mm_add_ps((x), (y))
  #define VECTOR_MUL(x, y) _mm_mul_ps((x), (y))
  #define VECTOR_SUB(x, y) _mm_sub_ps((x), (y))
  #define VECTOR_DIV(x, y) _mm_div_ps((x), (y))
  #define VECTOR_SET1(x) _mm_set1_ps((x))
#endif
#define NUMBER_OF_VEC (VECTOR_SIZE / (sizeof(TYPE) * 8))
#define LOG2S(k, s) { size_t t = k; while (t >>= 1) s++; }

int num_threads = 16;

class Tensor {
public:
  Tensor();
  Tensor(float *buf_, std::vector<size_t> shape_);
  void alloc_once(std::vector<size_t> shape_);
  void set_sz();

  // For real world application, one should use smart pointer to prevent possible memory leak.
  // However, we just use raw pointer for simplicity. Memory will be freed anyway when process exits.
  float* buf;

  // Shape of tensor, from outermost dimension to innermost dimension.
  // e.g., [[1, 2, 3], [4, 5, 6]] => shape = [2, 3]
  std::vector<size_t> shape;

  // Size of tensor; product of all dimensions
  size_t sz;
};

// Helpers
static void register_weight(std::map<std::string, Tensor>& weights, float* (*buf), std::string name, std::vector<size_t> shape);
static std::map<std::string, Tensor> register_weights(float* weight_buf);
static Tensor preprocess(uint8_t *in, size_t num_image);
static void postprocess_one_image(Tensor input, uint8_t *out, size_t idx);
static void get_one_image(Tensor input, Tensor &output, size_t idx);

// Operators
static void conv2d(Tensor input, Tensor filter, Tensor bias, Tensor &output);
static void conv2d_transposed(Tensor input, Tensor filter, Tensor bias, Tensor &output);
static void leaky_relu(Tensor input, Tensor &output, float alpha);
static void relu(Tensor input, Tensor &output);
static void batchnorm(Tensor input, Tensor scale, Tensor offset, Tensor &output);
static void concat(Tensor input0, Tensor input1, Tensor &output);
static void elem_tanh(Tensor input, Tensor &output);

void pix2pix_init() {
  /*
   * You can do input-independent and input-size-independent jobs here.
   * e.g., Getting OpenCL platform, Compiling OpenCL kernel, ...
   * Execution time of this function is not measured, so do as much as possible!
   */
}

void pix2pix(uint8_t *input_buf, float *weight_buf, uint8_t *output_buf, size_t num_image) {
  /*
   * !!!!!!!! Caution !!!!!!!!
   * In MPI program, all buffers and num_image are only given to rank 0 process.
   * You should manually:
   *   1. allocate buffers on others
   *   2. send inputs from rank 0 to others
   *   3. gather outputs from others to rank 0
   */

  auto weights = register_weights(weight_buf); // Memory allocated for weights
  auto input = preprocess(input_buf, num_image); // Memory allocated for input

  // Declare feature maps
  // Memory for feature maps are allocated when they are written first time using Tensor::alloc_once(...)

  #pragma omp parallel for num_threads(num_threads)
  for (size_t img_idx = 0; img_idx < num_image; ++img_idx) {
    Tensor one_image;
    Tensor encoder_layer_input[9];
    Tensor encoder_layer_rectified[9];
    Tensor encoder_layer_convolved[9];
    Tensor encoder_layer[9];
    Tensor decoder_layer_input[9];
    Tensor decoder_layer_rectified[9];
    Tensor decoder_layer_convolved[9];
    Tensor decoder_layer[9];

    // Pick 1 image out of num_image
    get_one_image(input, one_image, img_idx);

    /*
     * Encoding phase
     */

    // Encoder 1 : conv
    auto filter = weights["generator/encoder_1/conv2d/kernel"];
    auto bias = weights["generator/encoder_1/conv2d/bias"];
    conv2d(one_image, filter, bias, encoder_layer[1]);

    for (int i = 2; i <= 8; ++i) {
      // Encoder i : leaky_relu => conv2d => batchnorm
      auto scope = "generator/encoder_" + std::to_string(i);
      auto filter = weights[scope + "/conv2d/kernel"];
      auto bias = weights[scope + "/conv2d/bias"];
      auto scale = weights[scope + "/batch_normalization/gamma"];
      auto offset = weights[scope + "/batch_normalization/beta"];
      encoder_layer_input[i] = encoder_layer[i - 1];
      leaky_relu(encoder_layer_input[i], encoder_layer_rectified[i], 0.2);
      conv2d(encoder_layer_rectified[i], filter, bias, encoder_layer_convolved[i]);
      batchnorm(encoder_layer_convolved[i], scale, offset, encoder_layer[i]);
    }

    /*
     * Decoding phase
     */

    for (int i = 8; i >= 1; --i) {
      // Decoder i : relu => conv2d_transposed => batchnorm
      auto scope = "generator/decoder_" + std::to_string(i);
      auto filter = weights[scope + "/conv2d_transpose/kernel"];
      auto bias = weights[scope + "/conv2d_transpose/bias"];
      auto scale = weights[scope + "/batch_normalization/gamma"];
      auto offset = weights[scope + "/batch_normalization/beta"];
      if (i == 8) {
        // For decoder 8, input is last layer of encoder
        decoder_layer_input[i] = encoder_layer[8];
      } else {
        // For other decoder, input is concatenation of previous layer and corresponding encoder layer
        concat(decoder_layer[i + 1], encoder_layer[i], decoder_layer_input[i]);
      }
      relu(decoder_layer_input[i], decoder_layer_rectified[i]);
      conv2d_transposed(decoder_layer_rectified[i], filter, bias, decoder_layer_convolved[i]);

      // Last decoder does not have batchnorm
      if (i == 1) break;
      batchnorm(decoder_layer_convolved[i], scale, offset, decoder_layer[i]);
    }

    // Convert values into [-1, 1] using tanh function
    elem_tanh(decoder_layer_convolved[1], decoder_layer[1]);

    // Put a image into output buffer
    postprocess_one_image(decoder_layer[1], output_buf, img_idx);
  }
}

Tensor::Tensor() : buf(NULL) {}

// If buf is given, use it. If not, allocate new one.
Tensor::Tensor(float *buf_, std::vector<size_t> shape_) : buf(buf_), shape(shape_) {
  set_sz();
  if (buf == NULL) {
    buf = (float*)malloc(sz * sizeof(float));
  }
}

// If buf is not allocated, allocate new one.
void Tensor::alloc_once(std::vector<size_t> shape_) {
  if (buf == NULL) {
    shape = shape_;
    set_sz();
    buf = (float*)malloc(sz * sizeof(float));
  }
}

void Tensor::set_sz() {
  sz = 1;
  for (auto x : shape) {
    sz *= x;
  }
}

// Make a new tensor from buffer and put the tensor into map. Advance buffer pointer by size.
void register_weight(std::map<std::string, Tensor>& weights, float* (*buf), std::string name, std::vector<size_t> shape) {
  Tensor tensor(*buf, shape);
  weights[name] = tensor;
  *buf += tensor.sz;
}

// Put all predefined weights into map. Order should not be changed.
std::map<std::string, Tensor> register_weights(float* weight_buf) {
  std::map<std::string, Tensor> weights;
  // auto generated
  register_weight(weights, &weight_buf, "generator/decoder_1/conv2d_transpose/bias", {3});
  register_weight(weights, &weight_buf, "generator/decoder_1/conv2d_transpose/kernel", {4, 4, 3, 128});
  register_weight(weights, &weight_buf, "generator/decoder_2/batch_normalization/beta", {64});
  register_weight(weights, &weight_buf, "generator/decoder_2/batch_normalization/gamma", {64});
  register_weight(weights, &weight_buf, "generator/decoder_2/batch_normalization/moving_mean", {64});
  register_weight(weights, &weight_buf, "generator/decoder_2/batch_normalization/moving_variance", {64});
  register_weight(weights, &weight_buf, "generator/decoder_2/conv2d_transpose/bias", {64});
  register_weight(weights, &weight_buf, "generator/decoder_2/conv2d_transpose/kernel", {4, 4, 64, 256});
  register_weight(weights, &weight_buf, "generator/decoder_3/batch_normalization/beta", {128});
  register_weight(weights, &weight_buf, "generator/decoder_3/batch_normalization/gamma", {128});
  register_weight(weights, &weight_buf, "generator/decoder_3/batch_normalization/moving_mean", {128});
  register_weight(weights, &weight_buf, "generator/decoder_3/batch_normalization/moving_variance", {128});
  register_weight(weights, &weight_buf, "generator/decoder_3/conv2d_transpose/bias", {128});
  register_weight(weights, &weight_buf, "generator/decoder_3/conv2d_transpose/kernel", {4, 4, 128, 512});
  register_weight(weights, &weight_buf, "generator/decoder_4/batch_normalization/beta", {256});
  register_weight(weights, &weight_buf, "generator/decoder_4/batch_normalization/gamma", {256});
  register_weight(weights, &weight_buf, "generator/decoder_4/batch_normalization/moving_mean", {256});
  register_weight(weights, &weight_buf, "generator/decoder_4/batch_normalization/moving_variance", {256});
  register_weight(weights, &weight_buf, "generator/decoder_4/conv2d_transpose/bias", {256});
  register_weight(weights, &weight_buf, "generator/decoder_4/conv2d_transpose/kernel", {4, 4, 256, 1024});
  register_weight(weights, &weight_buf, "generator/decoder_5/batch_normalization/beta", {512});
  register_weight(weights, &weight_buf, "generator/decoder_5/batch_normalization/gamma", {512});
  register_weight(weights, &weight_buf, "generator/decoder_5/batch_normalization/moving_mean", {512});
  register_weight(weights, &weight_buf, "generator/decoder_5/batch_normalization/moving_variance", {512});
  register_weight(weights, &weight_buf, "generator/decoder_5/conv2d_transpose/bias", {512});
  register_weight(weights, &weight_buf, "generator/decoder_5/conv2d_transpose/kernel", {4, 4, 512, 1024});
  register_weight(weights, &weight_buf, "generator/decoder_6/batch_normalization/beta", {512});
  register_weight(weights, &weight_buf, "generator/decoder_6/batch_normalization/gamma", {512});
  register_weight(weights, &weight_buf, "generator/decoder_6/batch_normalization/moving_mean", {512});
  register_weight(weights, &weight_buf, "generator/decoder_6/batch_normalization/moving_variance", {512});
  register_weight(weights, &weight_buf, "generator/decoder_6/conv2d_transpose/bias", {512});
  register_weight(weights, &weight_buf, "generator/decoder_6/conv2d_transpose/kernel", {4, 4, 512, 1024});
  register_weight(weights, &weight_buf, "generator/decoder_7/batch_normalization/beta", {512});
  register_weight(weights, &weight_buf, "generator/decoder_7/batch_normalization/gamma", {512});
  register_weight(weights, &weight_buf, "generator/decoder_7/batch_normalization/moving_mean", {512});
  register_weight(weights, &weight_buf, "generator/decoder_7/batch_normalization/moving_variance", {512});
  register_weight(weights, &weight_buf, "generator/decoder_7/conv2d_transpose/bias", {512});
  register_weight(weights, &weight_buf, "generator/decoder_7/conv2d_transpose/kernel", {4, 4, 512, 1024});
  register_weight(weights, &weight_buf, "generator/decoder_8/batch_normalization/beta", {512});
  register_weight(weights, &weight_buf, "generator/decoder_8/batch_normalization/gamma", {512});
  register_weight(weights, &weight_buf, "generator/decoder_8/batch_normalization/moving_mean", {512});
  register_weight(weights, &weight_buf, "generator/decoder_8/batch_normalization/moving_variance", {512});
  register_weight(weights, &weight_buf, "generator/decoder_8/conv2d_transpose/bias", {512});
  register_weight(weights, &weight_buf, "generator/decoder_8/conv2d_transpose/kernel", {4, 4, 512, 512});
  register_weight(weights, &weight_buf, "generator/encoder_1/conv2d/bias", {64});
  register_weight(weights, &weight_buf, "generator/encoder_1/conv2d/kernel", {4, 4, 3, 64});
  register_weight(weights, &weight_buf, "generator/encoder_2/batch_normalization/beta", {128});
  register_weight(weights, &weight_buf, "generator/encoder_2/batch_normalization/gamma", {128});
  register_weight(weights, &weight_buf, "generator/encoder_2/batch_normalization/moving_mean", {128});
  register_weight(weights, &weight_buf, "generator/encoder_2/batch_normalization/moving_variance", {128});
  register_weight(weights, &weight_buf, "generator/encoder_2/conv2d/bias", {128});
  register_weight(weights, &weight_buf, "generator/encoder_2/conv2d/kernel", {4, 4, 64, 128});
  register_weight(weights, &weight_buf, "generator/encoder_3/batch_normalization/beta", {256});
  register_weight(weights, &weight_buf, "generator/encoder_3/batch_normalization/gamma", {256});
  register_weight(weights, &weight_buf, "generator/encoder_3/batch_normalization/moving_mean", {256});
  register_weight(weights, &weight_buf, "generator/encoder_3/batch_normalization/moving_variance", {256});
  register_weight(weights, &weight_buf, "generator/encoder_3/conv2d/bias", {256});
  register_weight(weights, &weight_buf, "generator/encoder_3/conv2d/kernel", {4, 4, 128, 256});
  register_weight(weights, &weight_buf, "generator/encoder_4/batch_normalization/beta", {512});
  register_weight(weights, &weight_buf, "generator/encoder_4/batch_normalization/gamma", {512});
  register_weight(weights, &weight_buf, "generator/encoder_4/batch_normalization/moving_mean", {512});
  register_weight(weights, &weight_buf, "generator/encoder_4/batch_normalization/moving_variance", {512});
  register_weight(weights, &weight_buf, "generator/encoder_4/conv2d/bias", {512});
  register_weight(weights, &weight_buf, "generator/encoder_4/conv2d/kernel", {4, 4, 256, 512});
  register_weight(weights, &weight_buf, "generator/encoder_5/batch_normalization/beta", {512});
  register_weight(weights, &weight_buf, "generator/encoder_5/batch_normalization/gamma", {512});
  register_weight(weights, &weight_buf, "generator/encoder_5/batch_normalization/moving_mean", {512});
  register_weight(weights, &weight_buf, "generator/encoder_5/batch_normalization/moving_variance", {512});
  register_weight(weights, &weight_buf, "generator/encoder_5/conv2d/bias", {512});
  register_weight(weights, &weight_buf, "generator/encoder_5/conv2d/kernel", {4, 4, 512, 512});
  register_weight(weights, &weight_buf, "generator/encoder_6/batch_normalization/beta", {512});
  register_weight(weights, &weight_buf, "generator/encoder_6/batch_normalization/gamma", {512});
  register_weight(weights, &weight_buf, "generator/encoder_6/batch_normalization/moving_mean", {512});
  register_weight(weights, &weight_buf, "generator/encoder_6/batch_normalization/moving_variance", {512});
  register_weight(weights, &weight_buf, "generator/encoder_6/conv2d/bias", {512});
  register_weight(weights, &weight_buf, "generator/encoder_6/conv2d/kernel", {4, 4, 512, 512});
  register_weight(weights, &weight_buf, "generator/encoder_7/batch_normalization/beta", {512});
  register_weight(weights, &weight_buf, "generator/encoder_7/batch_normalization/gamma", {512});
  register_weight(weights, &weight_buf, "generator/encoder_7/batch_normalization/moving_mean", {512});
  register_weight(weights, &weight_buf, "generator/encoder_7/batch_normalization/moving_variance", {512});
  register_weight(weights, &weight_buf, "generator/encoder_7/conv2d/bias", {512});
  register_weight(weights, &weight_buf, "generator/encoder_7/conv2d/kernel", {4, 4, 512, 512});
  register_weight(weights, &weight_buf, "generator/encoder_8/batch_normalization/beta", {512});
  register_weight(weights, &weight_buf, "generator/encoder_8/batch_normalization/gamma", {512});
  register_weight(weights, &weight_buf, "generator/encoder_8/batch_normalization/moving_mean", {512});
  register_weight(weights, &weight_buf, "generator/encoder_8/batch_normalization/moving_variance", {512});
  register_weight(weights, &weight_buf, "generator/encoder_8/conv2d/bias", {512});
  register_weight(weights, &weight_buf, "generator/encoder_8/conv2d/kernel", {4, 4, 512, 512});
  return weights;
}

// Convert 8-bit depth images (value range [0, 255]) into floating-point ones (value range [-1, 1])
Tensor preprocess(uint8_t *in, size_t num_image) {
  Tensor out(NULL, {num_image, 256, 256, 3});
  #ifdef SHOW_TIME
  START
  #endif
  const size_t img_size = out.sz;
  const size_t block_size = img_size / 128 / 1024 + (img_size % (128 * 1024) != 0);
  for (size_t block = 0; block < block_size; block++) {
    const size_t block_min = img_size * (block) / block_size;
    const size_t block_max = img_size * (block + 1) / block_size;
    for (size_t i = block_min; i < block_max; ++i) {
      out.buf[i] = in[i] / 255.0f * 2 - 1;
    }
  }
  #ifdef SHOW_TIME
  END("preprocess")
  #endif
  // PROJECT - USE CPU MULTITHREAD TO CALCULATE
  // CPU
  // Caching
  return out;
}

// Inverse of preprocess
void postprocess_one_image(Tensor input, uint8_t *out, size_t idx) {
  // input shape = (height, width, channels)
  size_t H = input.shape[0], W = input.shape[1], C = input.shape[2];
  #ifdef SHOW_TIME
  START
  #endif
  const size_t img_size = H * W * C;
  const size_t block_size = img_size / 128 / 1024 + (img_size % (128 * 1024) != 0);
  for (size_t block = 0; block < block_size; block++) {
    const size_t block_min = img_size * (block) / block_size;
    const size_t block_max = img_size * (block + 1) / block_size;
    for (size_t i = block_min; i < block_max; ++i) { // 256 * 256 * 3 * 8
      float x = (input.buf[i] + 1) / 2 * 255;
      out[idx * img_size + i] = x < 0 ? 0 : (x > 255 ? 255 : x);
    }
  }
  #ifdef SHOW_TIME
  END("postprocess_one_image")
  #endif
  // PROJECT - USE CPU MULTITHREAD TO CALCULATE
  // CPU
  // Caching
}

// Pick single image from images
void get_one_image(Tensor input, Tensor &output, size_t idx) {
  // input shape = (num_image, height, width, channels)
  // output shape = (height, width, channels)
  size_t H = input.shape[1], W = input.shape[2], C = input.shape[3];
  output.alloc_once({H, W, C});
  #ifdef SHOW_TIME
  START
  #endif
  const size_t img_size = H * W * C;
  const size_t block_size = img_size / 128 / 1024 + (img_size % (128 * 1024) != 0);
  for (size_t block = 0; block < block_size; block++) {
    const size_t block_min = img_size * (block) / block_size;
    const size_t block_max = img_size * (block + 1) / block_size;
    for (size_t i = block_min; i < block_max; ++i) { // 256 * 256 * 3 * 8
      output.buf[i] = input.buf[idx * img_size + i];
    }
  }
  #ifdef SHOW_TIME
  END("get_one_image")
  #endif
  // PROJECT - USE CPU MULTITHREAD TO COPY
  // CPU
  // Caching
}

// Convolution (2-dimension, stride = 2, pad = 1)
void conv2d(Tensor input, Tensor filter, Tensor bias, Tensor &output) {
  // input shape = (in_height, in_width, in_channels)
  // filter shape = (filter_height, filter_width, in_channels, output_channels)
  // bias shape = (output_channels)
  // output shape = (in_height / stride, in_width / stride, output_channels)
  size_t H = input.shape[0], W = input.shape[1], C = input.shape[2];
  size_t R = filter.shape[0], S = filter.shape[1], K = filter.shape[3];
  const size_t stride = 2, pad = 1;
  size_t OH = H / stride, OW = W / stride;
  output.alloc_once({OH, OW, K});

  #ifdef SHOW_TIME
  START
  #endif
  size_t K_p = 0;
  size_t OH_p = 0;
  size_t OW_p = 0;
  LOG2S(K, K_p);
  LOG2S(OH, OH_p);
  LOG2S(OW, OW_p);
  const size_t OWK_p = OW_p + K_p;
  const size_t OHOWK = OH * OW * K;
  for (size_t ohowk = 0; ohowk < OHOWK; ohowk += NUMBER_OF_VEC) {
    size_t oh = ohowk >> OWK_p;
    size_t ow = (ohowk >> K_p) & ((1 << OW_p) - 1);
    size_t k = (ohowk) & ((1 << K_p) - 1);
    VECTOR_TYPE x = VECTOR_LOAD(bias.buf + k);
    for (size_t r = 0; r < R; ++r) {
      for (size_t s = 0; s < S; ++s) {
        // input (oh * stride - pad + r, ow * stride - pad + s, c)
        size_t ih = oh * stride - pad + r;
        size_t iw = ow * stride - pad + s;
        if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;
        for (size_t c = 0; c < C; ++c) {
          float ii = input.buf[ih * W * C + iw * C + c];
          // filter (r, s, c, k)
          x = VECTOR_ADD(x,
              VECTOR_MUL(VECTOR_SET1(ii),
                VECTOR_LOAD(filter.buf + (r * S * C * K + s * C * K + c * K + k))));
        }
      }
    }
    // output (oh, ow, k)
    VECTOR_STORE(output.buf + ohowk, x);
  }
  #ifdef SHOW_TIME
  END("conv2d")
  #endif
  // PROJECT - USE CPU/GPU MULTITHREAD TO CALCULATE
}

// Transposed convolution (2-dimension, stride = 2, pad = 1)
void conv2d_transposed(Tensor input, Tensor filter, Tensor bias, Tensor &output) {
  // input shape = (in_height, in_width, in_channels)
  // filter shape = (filter_height, filter_width, output_channels, in_channels)
  // bias shape = (output_channels)
  // output shape = (in_height * stride, in_width * stride, output_channels)
  size_t H = input.shape[0], W = input.shape[1], C = input.shape[2];
  size_t R = filter.shape[0], S = filter.shape[1], K = filter.shape[2];
  // assume stride 2, pad 1
  const size_t stride = 2, pad = 1;
  size_t OH = H * stride, OW = W * stride;
  output.alloc_once({OH, OW, K});

  #ifdef SHOW_TIME
  START
  #endif
  const size_t OWK = OW * K;
  const size_t OHOWK = OH * OW * K;

  float HOLDER[NUMBER_OF_VEC] = {0, };
  for (size_t ohowk = 0; ohowk < OHOWK; ++ohowk) {
    size_t oh = ohowk / OWK;
    size_t ow = ohowk / K % OW;
    size_t k = ohowk % K;
    VECTOR_TYPE x = VECTOR_SET1(0);
    for (size_t r = 0; r < R; ++r) {
      for (size_t s = 0; s < S; ++s) {
        // input ((oh - r + pad) / stride, (ow - s + pad) / stride, c)
        //   where (oh - r + pad) % stride == 0 && (ow - s + pad) % stride == 0
        if ((oh - r + pad) % stride != 0 || (ow - s + pad) % stride != 0) continue;
        size_t ih = (oh - r + pad) / stride;
        size_t iw = (ow - s + pad) / stride;
        if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;
        for (size_t c = 0; c < C; c += NUMBER_OF_VEC) {
          // filter (r, s, k, c)
          x = VECTOR_ADD(x, VECTOR_MUL(
            VECTOR_LOAD(input.buf + (ih * W * C + iw * C + c)),
            VECTOR_LOAD(filter.buf + (r * S * K * C + s * K * C + k * C + c))
          ));
        }
      }
    }
    // output (oh, ow, k)
    float r = 0.0f;
    VECTOR_STORE(HOLDER, x);
    for (size_t i = 0; i < NUMBER_OF_VEC; ++i) {
      r += HOLDER[i];
    }
    output.buf[ohowk] = r + bias.buf[k];
  }
  #ifdef SHOW_TIME
  END("conv2d_transposed")
  #endif
  // PROJECT - USE CPU/GPU MULTITHREAD TO CALCULATE
}

// Leaky ReLU
void leaky_relu(Tensor input, Tensor &output, float alpha) {
  // input shape = (height, width, channels)
  // output shape = (height, width, channels)
  size_t H = input.shape[0], W = input.shape[1], C = input.shape[2];
  output.alloc_once({H, W, C});
  #ifdef SHOW_TIME
  START
  #endif
  for (size_t i = 0; i < H * W * C; ++i) {
    output.buf[i] = input.buf[i] >= 0 ? input.buf[i] : alpha * input.buf[i];
  }
  #ifdef SHOW_TIME
  END("leaky_relu")
  #endif
  // PROJECT - USE CPU/GPU MULTITHREAD TO CALCULATE
}

// ReLU
void relu(Tensor input, Tensor &output) {
  // input shape = (height, width, channels)
  // output shape = (height, width, channels)
  size_t H = input.shape[0], W = input.shape[1], C = input.shape[2];
  output.alloc_once({H, W, C});
  #ifdef SHOW_TIME
  START
  #endif
  for (size_t i = 0; i < H * W * C; ++i) {
    output.buf[i] = input.buf[i] >= 0 ? input.buf[i] : 0;
  }
  #ifdef SHOW_TIME
  END("relu")
  #endif
  // PROJECT - USE CPU/GPU MULTITHREAD TO CALCULATE
}

// Batch normalization (channel-wise)
void batchnorm(Tensor input, Tensor scale, Tensor offset, Tensor &output) {
  // input shape = (height, width, channels)
  // scale shape = (channels)
  // offset shape = (channels)
  // output shape = (height, width, channels)
  size_t H = input.shape[0], W = input.shape[1], C = input.shape[2];
  output.alloc_once({H, W, C});
  #ifdef SHOW_TIME
  START
  #endif
  for (size_t c = 0; c < C; ++c) {
    float sum = 0;
    for (size_t h = 0; h < H; ++h) {
      for (size_t w = 0; w < W; ++w) {
        float ii = input.buf[h * W * C + w * C + c];
        sum += ii;
      }
    }
    // PROJECT - USE CPU MULTITHREAD TO CALCULATE reduction
    float mean = sum / (H * W);

    float sqsum = 0;
    for (size_t h = 0; h < H; ++h) {
      for (size_t w = 0; w < W; ++w) {
        float ii = input.buf[h * W * C + w * C + c];
        sqsum += (ii - mean) * (ii - mean);
      }
    }
    // PROJECT - USE CPU MULTITHREAD TO CALCULATE reduction
    float variance = sqsum / (H * W);

    const float epsilon = 1e-5;
    for (size_t h = 0; h < H; ++h) {
      for (size_t w = 0; w < W; ++w) {
        size_t idx = h * W * C + w * C + c;
        output.buf[idx] = offset.buf[c] + (input.buf[idx] - mean) * scale.buf[c] / sqrtf(variance + epsilon);
      }
    }
    // PROJECT - USE CPU MULTITHREAD TO CALCULATE
  }
  #ifdef SHOW_TIME
  END("batchnorm")
  #endif
}

// Concatenation (along channel dimension)
void concat(Tensor input0, Tensor input1, Tensor &output) {
  // input0 shape = (height, width, channels0)
  // input1 shape = (height, width, channels1)
  // output shape = (height, width, channels0 + channels1)
  size_t H = input0.shape[0], W = input0.shape[1], C0 = input0.shape[2];
  size_t C1 = input1.shape[2];
  output.alloc_once({H, W, C0 + C1});
  #ifdef SHOW_TIME
  START
  #endif
  for (size_t h = 0; h < H; ++h) {
    for (size_t w = 0; w < W; ++w) {
      for (size_t c = 0; c < C0; ++c) {
        output.buf[h * W * (C0 + C1) + w * (C0 + C1) + c] = input0.buf[h * W * C0 + w * C0 + c];
      }
      for (size_t c = 0; c < C1; ++c) {
        output.buf[h * W * (C0 + C1) + w * (C0 + C1) + (C0 + c)] = input1.buf[h * W * C1 + w * C1 + c];
      }
    }
  }
  #ifdef SHOW_TIME
  END("concat")
  #endif
  // PROJECT - USE CPU/GPU MULTITHREAD TO CALCULATE
  // GPU
  // Caching
}

// Elementwise tanh
void elem_tanh(Tensor input, Tensor &output) {
  // input shape = (height, width, channels)
  // output shape = (height, width, channels)
  size_t H = input.shape[0], W = input.shape[1], C = input.shape[2];
  output.alloc_once({H, W, C});
  #ifdef SHOW_TIME
  START
  #endif
  for (size_t i = 0; i < H * W * C; ++i) {
    output.buf[i] = tanhf(input.buf[i]);
  }
  #ifdef SHOW_TIME
  END("elem_tanh")
  #endif
  // PROJECT - USE CPU/GPU MULTITHREAD TO CALCULATE
}
