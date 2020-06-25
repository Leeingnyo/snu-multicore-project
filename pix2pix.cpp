#include "pix2pix.h"

#include "util.h"

#include <immintrin.h> // include vector
#include <omp.h>
#include <CL/cl.h>
#include <mpi.h>

#include <string>
#include <map>
#include <cmath>

// #define SHOW_TIME
// #define FINISH
#define START double st = get_time();
#define END(x) double et = get_time(); printf("\n%s! (%lf s)", x, et - st);
#define START_RE st = get_time();
#define END_RE(x) et = get_time(); printf("\n%s! (%lf s)", x, et - st);

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

#define CHECK_ERROR(err) \
  if (err != CL_SUCCESS) { \
    printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
    exit(EXIT_FAILURE); \
  }

#define TILE_SIZE 28
#define PADDING(x, y) (((x)-1)/(y)*(y)+(y))

#define DEVICE_NUM 1
#define KERNEL_NUM 12

static cl_int err;
static cl_platform_id platform;
static cl_device_id device[DEVICE_NUM];
static cl_context context[DEVICE_NUM];
static cl_command_queue queue[DEVICE_NUM];
static cl_program program[DEVICE_NUM];
static cl_kernel kernel[DEVICE_NUM][KERNEL_NUM];
enum kernel_type { K_CONV2D, K_CONV2D_TRANSPOSED, K_CONV2D_LEAKYRELU, K_CONV2D_BATCHNORM_LEAKYRELU, K_MEAN, K_VARIANCE, K_BATCHNORM, K_BATCHNORM_LEAKYRELU, K_LEAKYRELU, CONCAT, TANH };

int num_threads = 4;

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
static void run_kernel_conv2d(Tensor input, Tensor filter, Tensor bias, Tensor &output, float alpha);
void run_kernel_conv2d2(Tensor input, Tensor filter, Tensor bias, Tensor &output, float alpha, Tensor scale, Tensor offset);
static void conv2d(Tensor input, Tensor filter, Tensor bias, Tensor &output);
static void conv2d_transposed(Tensor input, Tensor filter, Tensor bias, Tensor &output);
static void leaky_relu(Tensor input, Tensor &output, float alpha);
static void relu(Tensor input, Tensor &output);
static void batchnorm(Tensor input, Tensor scale, Tensor offset, Tensor &output);
static void concat(Tensor input0, Tensor input1, Tensor &output);
static void elem_tanh(Tensor input, Tensor &output);

void encoding(
  int device_num,
  Tensor one_image,
  Tensor &encoded,
  std::map<std::string, Tensor> &weights
);
void conv2d_kernel(int device_num, cl_mem &input, cl_mem &output, Tensor filter, Tensor bias, size_t &H, size_t &W, size_t &C);
void leakyrelu(int device_num, cl_mem &input, cl_mem &output, float alpha, size_t H, size_t W, size_t C);
void mean_kernel(int device_num, cl_mem &input, cl_mem &mean_mem, size_t &H, size_t &W, size_t &C);
void variance_kernel(int device_num, cl_mem &input, cl_mem &mean_mem, cl_mem &variance_mem, size_t H, size_t W, size_t C);

static cl_program create_and_build_program_with_source(cl_context context, cl_device_id device, const char *file_name) {
  FILE *file = fopen(file_name, "rb");
  if (file == NULL) {
    printf("Failed to open %s\n", file_name);
    exit(EXIT_FAILURE);
  }
  fseek(file, 0, SEEK_END);
  size_t source_size = ftell(file);
  rewind(file);
  char *source_code = (char*)malloc(source_size + 1);
  fread(source_code, sizeof(char), source_size, file);
  source_code[source_size] = '\0';
  fclose(file);
  cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source_code, &source_size, &err);
  CHECK_ERROR(err);
  free(source_code);
  err = clBuildProgram(program, 1, &device, "", NULL, NULL);
  if (err == CL_BUILD_PROGRAM_FAILURE) {
    size_t log_size;
    CHECK_ERROR(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size));
    char *log = (char*)malloc(log_size + 1);
    CHECK_ERROR(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL));
    log[log_size] = 0;
    printf("Compile error:\n%s\n", log);
    free(log);
  }
  CHECK_ERROR(err);
  return program;
}

void pix2pix_init() {
  /*
   * You can do input-independent and input-size-independent jobs here.
   * e.g., Getting OpenCL platform, Compiling OpenCL kernel, ...
   * Execution time of this function is not measured, so do as much as possible!
   */

  // platform
  err = clGetPlatformIDs(1, &platform, NULL);
  CHECK_ERROR(err);

  // device
  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, DEVICE_NUM, device, NULL);
  CHECK_ERROR(err);

  // context
  for (int d = 0; d < DEVICE_NUM; d++) {
    context[d] = clCreateContext(NULL, 1, &device[d], NULL, NULL, &err);
    CHECK_ERROR(err);
  }

  // command queue
  for (int d = 0; d < DEVICE_NUM; d++) {
    queue[d] = clCreateCommandQueue(context[d], device[d], 0, &err);
    CHECK_ERROR(err);
  }

  // program
  for (int d = 0; d < DEVICE_NUM; d++) {
    program[d] = create_and_build_program_with_source(context[d], device[d], "kernel.cl");
  }

  // kernel
  for (int d = 0; d < DEVICE_NUM; d++) {
    kernel[d][K_CONV2D] = clCreateKernel(program[d], "conv2d", &err);
    kernel[d][K_CONV2D_TRANSPOSED] = clCreateKernel(program[d], "conv2d_transposed", &err);
    kernel[d][K_CONV2D_LEAKYRELU] = clCreateKernel(program[d], "conv2d_leakyrelu", &err);
    kernel[d][K_CONV2D_BATCHNORM_LEAKYRELU] = clCreateKernel(program[d], "conv2d_batchnorm_leakyrelu", &err);
    kernel[d][K_MEAN] = clCreateKernel(program[d], "mean", &err);
    kernel[d][K_VARIANCE] = clCreateKernel(program[d], "variance", &err);
    kernel[d][K_BATCHNORM] = clCreateKernel(program[d], "batchnorm", &err);
    kernel[d][K_BATCHNORM_LEAKYRELU] = clCreateKernel(program[d], "batchnorm_leakyrelu", &err);
    kernel[d][K_LEAKYRELU] = clCreateKernel(program[d], "leakyrelu", &err);
    CHECK_ERROR(err);
  }

  // alloc buffers
  // alloc mpi buffers
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

  // #pragma omp parallel for num_threads(num_threads)
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
  cl_mem input_d[DEVICE_NUM], filter_d[DEVICE_NUM], bias_d[DEVICE_NUM], output_d[DEVICE_NUM];
  for (int d = 0; d < DEVICE_NUM; d++) {
    input_d[d] = clCreateBuffer(context[d], CL_MEM_READ_WRITE, input.sz * sizeof(float), NULL, &err);
    CHECK_ERROR(err);
    filter_d[d] = clCreateBuffer(context[d], CL_MEM_READ_WRITE, filter.sz * sizeof(float), NULL, &err);
    CHECK_ERROR(err);
    bias_d[d] = clCreateBuffer(context[d], CL_MEM_READ_WRITE, bias.sz * sizeof(float), NULL, &err);
    CHECK_ERROR(err);
    output_d[d] = clCreateBuffer(context[d], CL_MEM_READ_WRITE, output.sz * sizeof(float), NULL, &err);
    CHECK_ERROR(err);
  }

  for (int d = 0; d < DEVICE_NUM; d++) {
    err = clEnqueueWriteBuffer(queue[d], input_d[d], CL_TRUE, 0, input.sz * sizeof(float), input.buf, 0, NULL, NULL);
    CHECK_ERROR(err);
    err = clEnqueueWriteBuffer(queue[d], filter_d[d], CL_TRUE, 0, filter.sz * sizeof(float), filter.buf, 0, NULL, NULL);
    CHECK_ERROR(err);
    err = clEnqueueWriteBuffer(queue[d], bias_d[d], CL_TRUE, 0, bias.sz * sizeof(float), bias.buf, 0, NULL, NULL);
    CHECK_ERROR(err);
  }

  int H_ = H;
  int W_ = W;
  int C_ = C;
  int R_ = R;
  int S_ = S;
  int K_ = K;
  int OH_ = OH;
  int OW_ = OW;
  int stride_ = stride;
  int pad_ = pad;

  size_t K_p = 0;
  size_t OW_p = 0;
  LOG2S(K, K_p);
  LOG2S(OW, OW_p);
  const size_t K_mask = ((1 << K_p) - 1);
  const size_t OW_mask = ((1 << OW_p) - 1);

  for (int d = 0; d < DEVICE_NUM; d++) {
    err = clSetKernelArg(kernel[d][K_CONV2D], 0, sizeof(cl_mem), &input_d[d]);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel[d][K_CONV2D], 1, sizeof(cl_mem), &filter_d[d]);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel[d][K_CONV2D], 2, sizeof(cl_mem), &bias_d[d]);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel[d][K_CONV2D], 3, sizeof(cl_mem), &output_d[d]);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel[d][K_CONV2D], 4, sizeof(int), &H_);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel[d][K_CONV2D], 5, sizeof(int), &W_);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel[d][K_CONV2D], 6, sizeof(int), &C_);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel[d][K_CONV2D], 7, sizeof(int), &R_);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel[d][K_CONV2D], 8, sizeof(int), &S_);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel[d][K_CONV2D], 9, sizeof(int), &K_);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel[d][K_CONV2D], 10, sizeof(int), &OH_);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel[d][K_CONV2D], 11, sizeof(int), &OW_);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel[d][K_CONV2D], 12, sizeof(int), &stride_);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel[d][K_CONV2D], 13, sizeof(int), &pad_);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel[d][K_CONV2D], 14, sizeof(int), &K_p);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel[d][K_CONV2D], 15, sizeof(int), &OW_p);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel[d][K_CONV2D], 16, sizeof(int), &K_mask);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel[d][K_CONV2D], 17, sizeof(int), &OW_mask);
    CHECK_ERROR(err);
  }

  #ifdef FINISH
  for (int d = 0; d < DEVICE_NUM; d++) {
    clFinish(queue[d]);
  }
  #endif
  #ifdef SHOW_TIME
  END("conv2d write buffer")
  #endif

  #ifdef SHOW_TIME
  START_RE
  #endif
  size_t gws[1] = {OH * OW * K}, lws[1] = {128};
  for (int i = 0; i < 1; ++i) {
    gws[i] = (gws[i] + lws[i] - 1) / lws[i] * lws[i];
  }

  // Run kernel
  for (int d = 0; d < DEVICE_NUM; d++) {
    err = clEnqueueNDRangeKernel(queue[d], kernel[d][K_CONV2D], 1, NULL, gws, lws, 0, NULL, NULL);
    CHECK_ERROR(err);
  }

  #ifdef FINISH
  for (int d = 0; d < DEVICE_NUM; d++) {
    clFinish(queue[d]);
  }
  #endif
  #ifdef SHOW_TIME
  END_RE("conv2d")
  #endif

  #ifdef SHOW_TIME
  START_RE
  #endif
  // write
  for (int d = 0; d < DEVICE_NUM; d++) {
    err = clEnqueueReadBuffer(queue[d], output_d[d], CL_TRUE, 0, output.sz * sizeof(float), output.buf, 0, NULL, NULL);
    CHECK_ERROR(err);
  }

  #ifdef FINISH
  for (int d = 0; d < DEVICE_NUM; d++) {
    clFinish(queue[d]);
  }
  #endif
  #ifdef SHOW_TIME
  END_RE("conv2d read buffer")
  #endif
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
  cl_mem input_d[DEVICE_NUM], filter_d[DEVICE_NUM], bias_d[DEVICE_NUM], output_d[DEVICE_NUM];
  for (int d = 0; d < DEVICE_NUM; d++) {
    input_d[d] = clCreateBuffer(context[d], CL_MEM_READ_WRITE, input.sz * sizeof(float), NULL, &err);
    CHECK_ERROR(err);
    filter_d[d] = clCreateBuffer(context[d], CL_MEM_READ_WRITE, filter.sz * sizeof(float), NULL, &err);
    CHECK_ERROR(err);
    bias_d[d] = clCreateBuffer(context[d], CL_MEM_READ_WRITE, bias.sz * sizeof(float), NULL, &err);
    CHECK_ERROR(err);
    output_d[d] = clCreateBuffer(context[d], CL_MEM_READ_WRITE, output.sz * sizeof(float), NULL, &err);
    CHECK_ERROR(err);
  }

  for (int d = 0; d < DEVICE_NUM; d++) {
    err = clEnqueueWriteBuffer(queue[d], input_d[d], CL_TRUE, 0, input.sz * sizeof(float), input.buf, 0, NULL, NULL);
    CHECK_ERROR(err);
    err = clEnqueueWriteBuffer(queue[d], filter_d[d], CL_TRUE, 0, filter.sz * sizeof(float), filter.buf, 0, NULL, NULL);
    CHECK_ERROR(err);
    err = clEnqueueWriteBuffer(queue[d], bias_d[d], CL_TRUE, 0, bias.sz * sizeof(float), bias.buf, 0, NULL, NULL);
    CHECK_ERROR(err);
  }

  int H_ = H;
  int W_ = W;
  int C_ = C;
  int R_ = R;
  int S_ = S;
  int K_ = K;
  int OH_ = OH;
  int OW_ = OW;
  int stride_ = stride;
  int pad_ = pad;
  int OWK = OW * K;

  for (int d = 0; d < DEVICE_NUM; d++) {
    err = clSetKernelArg(kernel[d][K_CONV2D_TRANSPOSED], 0, sizeof(cl_mem), &input_d[d]);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel[d][K_CONV2D_TRANSPOSED], 1, sizeof(cl_mem), &filter_d[d]);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel[d][K_CONV2D_TRANSPOSED], 2, sizeof(cl_mem), &bias_d[d]);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel[d][K_CONV2D_TRANSPOSED], 3, sizeof(cl_mem), &output_d[d]);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel[d][K_CONV2D_TRANSPOSED], 4, sizeof(int), &H_);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel[d][K_CONV2D_TRANSPOSED], 5, sizeof(int), &W_);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel[d][K_CONV2D_TRANSPOSED], 6, sizeof(int), &C_);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel[d][K_CONV2D_TRANSPOSED], 7, sizeof(int), &R_);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel[d][K_CONV2D_TRANSPOSED], 8, sizeof(int), &S_);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel[d][K_CONV2D_TRANSPOSED], 9, sizeof(int), &K_);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel[d][K_CONV2D_TRANSPOSED], 10, sizeof(int), &OH_);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel[d][K_CONV2D_TRANSPOSED], 11, sizeof(int), &OW_);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel[d][K_CONV2D_TRANSPOSED], 12, sizeof(int), &stride_);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel[d][K_CONV2D_TRANSPOSED], 13, sizeof(int), &pad_);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel[d][K_CONV2D_TRANSPOSED], 14, sizeof(int), &OWK);
    CHECK_ERROR(err);
  }

  #ifdef FINISH
  for (int d = 0; d < DEVICE_NUM; d++) {
    clFinish(queue[d]);
  }
  #endif
  #ifdef SHOW_TIME
  END("conv2d_transposed write buffer")
  #endif

  #ifdef SHOW_TIME
  START_RE
  #endif
  size_t gws[1] = {OH * OW * K}, lws[1] = {128};
  for (int i = 0; i < 1; ++i) {
    gws[i] = (gws[i] + lws[i] - 1) / lws[i] * lws[i];
  }

  // Run kernel
  for (int d = 0; d < DEVICE_NUM; d++) {
    err = clEnqueueNDRangeKernel(queue[d], kernel[d][K_CONV2D_TRANSPOSED], 1, NULL, gws, lws, 0, NULL, NULL);
    CHECK_ERROR(err);
  }
  #ifdef SHOW_TIME
  END_RE("conv2d_transposed")
  #endif

  #ifdef SHOW_TIME
  START_RE
  #endif
  // write
  for (int d = 0; d < DEVICE_NUM; d++) {
    err = clEnqueueReadBuffer(queue[d], output_d[d], CL_TRUE, 0, output.sz * sizeof(float), output.buf, 0, NULL, NULL);
    CHECK_ERROR(err);
  }

  #ifdef FINISH
  for (int d = 0; d < DEVICE_NUM; d++) {
    clFinish(queue[d]);
  }
  #endif
  #ifdef SHOW_TIME
  END_RE("conv2d_transposed read buffer")
  #endif
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

void run_kernel_conv2d(Tensor input, Tensor filter, Tensor bias, Tensor &output, float alpha) {
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
  cl_mem input_d[DEVICE_NUM], filter_d[DEVICE_NUM], bias_d[DEVICE_NUM], output_d[DEVICE_NUM];
  for (int d = 0; d < DEVICE_NUM; d++) {
    input_d[d] = clCreateBuffer(context[d], CL_MEM_READ_WRITE, input.sz * sizeof(float), NULL, &err);
    CHECK_ERROR(err);
    filter_d[d] = clCreateBuffer(context[d], CL_MEM_READ_WRITE, filter.sz * sizeof(float), NULL, &err);
    CHECK_ERROR(err);
    bias_d[d] = clCreateBuffer(context[d], CL_MEM_READ_WRITE, bias.sz * sizeof(float), NULL, &err);
    CHECK_ERROR(err);
    output_d[d] = clCreateBuffer(context[d], CL_MEM_READ_WRITE, output.sz * sizeof(float), NULL, &err);
    CHECK_ERROR(err);
  }

  for (int d = 0; d < DEVICE_NUM; d++) {
    err = clEnqueueWriteBuffer(queue[d], input_d[d], CL_TRUE, 0, input.sz * sizeof(float), input.buf, 0, NULL, NULL);
    CHECK_ERROR(err);
    err = clEnqueueWriteBuffer(queue[d], filter_d[d], CL_TRUE, 0, filter.sz * sizeof(float), filter.buf, 0, NULL, NULL);
    CHECK_ERROR(err);
    err = clEnqueueWriteBuffer(queue[d], bias_d[d], CL_TRUE, 0, bias.sz * sizeof(float), bias.buf, 0, NULL, NULL);
    CHECK_ERROR(err);
  }

  int H_ = H;
  int W_ = W;
  int C_ = C;
  int R_ = R;
  int S_ = S;
  int K_ = K;
  int OH_ = OH;
  int OW_ = OW;
  int stride_ = stride;
  int pad_ = pad;

  size_t K_p = 0;
  size_t OW_p = 0;
  LOG2S(K, K_p);
  LOG2S(OW, OW_p);
  const size_t K_mask = ((1 << K_p) - 1);
  const size_t OW_mask = ((1 << OW_p) - 1);

  for (int d = 0; d < DEVICE_NUM; d++) {
    err = clSetKernelArg(kernel[d][K_CONV2D_LEAKYRELU], 0, sizeof(cl_mem), &input_d[d]);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel[d][K_CONV2D_LEAKYRELU], 1, sizeof(cl_mem), &filter_d[d]);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel[d][K_CONV2D_LEAKYRELU], 2, sizeof(cl_mem), &bias_d[d]);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel[d][K_CONV2D_LEAKYRELU], 3, sizeof(cl_mem), &output_d[d]);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel[d][K_CONV2D_LEAKYRELU], 4, sizeof(int), &H_);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel[d][K_CONV2D_LEAKYRELU], 5, sizeof(int), &W_);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel[d][K_CONV2D_LEAKYRELU], 6, sizeof(int), &C_);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel[d][K_CONV2D_LEAKYRELU], 7, sizeof(int), &R_);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel[d][K_CONV2D_LEAKYRELU], 8, sizeof(int), &S_);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel[d][K_CONV2D_LEAKYRELU], 9, sizeof(int), &K_);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel[d][K_CONV2D_LEAKYRELU], 10, sizeof(int), &OH_);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel[d][K_CONV2D_LEAKYRELU], 11, sizeof(int), &OW_);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel[d][K_CONV2D_LEAKYRELU], 12, sizeof(int), &stride_);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel[d][K_CONV2D_LEAKYRELU], 13, sizeof(int), &pad_);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel[d][K_CONV2D_LEAKYRELU], 14, sizeof(int), &K_p);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel[d][K_CONV2D_LEAKYRELU], 15, sizeof(int), &OW_p);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel[d][K_CONV2D_LEAKYRELU], 16, sizeof(int), &K_mask);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel[d][K_CONV2D_LEAKYRELU], 17, sizeof(int), &OW_mask);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel[d][K_CONV2D_LEAKYRELU], 18, sizeof(int), &alpha);
    CHECK_ERROR(err);
  }

  #ifdef FINISH
  for (int d = 0; d < DEVICE_NUM; d++) {
    clFinish(queue[d]);
  }
  #endif
  #ifdef SHOW_TIME
  END("conv2d write buffer")
  #endif

  #ifdef SHOW_TIME
  START_RE
  #endif
  size_t gws[1] = {OH * OW * K}, lws[1] = {128};
  for (int i = 0; i < 1; ++i) {
    gws[i] = (gws[i] + lws[i] - 1) / lws[i] * lws[i];
  }

  // Run kernel
  for (int d = 0; d < DEVICE_NUM; d++) {
    err = clEnqueueNDRangeKernel(queue[d], kernel[d][K_CONV2D_LEAKYRELU], 1, NULL, gws, lws, 0, NULL, NULL);
    CHECK_ERROR(err);
  }

  #ifdef FINISH
  for (int d = 0; d < DEVICE_NUM; d++) {
    clFinish(queue[d]);
  }
  #endif
  #ifdef SHOW_TIME
  END_RE("conv2d")
  #endif

  #ifdef SHOW_TIME
  START_RE
  #endif
  // write
  for (int d = 0; d < DEVICE_NUM; d++) {
    err = clEnqueueReadBuffer(queue[d], output_d[d], CL_TRUE, 0, output.sz * sizeof(float), output.buf, 0, NULL, NULL);
    CHECK_ERROR(err);
  }

  #ifdef FINISH
  for (int d = 0; d < DEVICE_NUM; d++) {
    clFinish(queue[d]);
  }
  #endif
  #ifdef SHOW_TIME
  END_RE("conv2d read buffer")
  #endif
}

void run_kernel_conv2d2(Tensor input, Tensor filter, Tensor bias, Tensor &output, float alpha, Tensor scale, Tensor offset) {
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
  cl_mem input_d[DEVICE_NUM], filter_d[DEVICE_NUM], bias_d[DEVICE_NUM], output_d[DEVICE_NUM];
  cl_mem scale_d[DEVICE_NUM], offset_d[DEVICE_NUM];
  cl_mem sum[DEVICE_NUM], sqsum[DEVICE_NUM];
  for (int d = 0; d < DEVICE_NUM; d++) {
    input_d[d] = clCreateBuffer(context[d], CL_MEM_READ_WRITE, input.sz * sizeof(float), NULL, &err);
    CHECK_ERROR(err);
    filter_d[d] = clCreateBuffer(context[d], CL_MEM_READ_WRITE, filter.sz * sizeof(float), NULL, &err);
    CHECK_ERROR(err);
    bias_d[d] = clCreateBuffer(context[d], CL_MEM_READ_WRITE, bias.sz * sizeof(float), NULL, &err);
    CHECK_ERROR(err);
    output_d[d] = clCreateBuffer(context[d], CL_MEM_READ_WRITE, output.sz * sizeof(float), NULL, &err);
    CHECK_ERROR(err);
    scale_d[d] = clCreateBuffer(context[d], CL_MEM_READ_WRITE, scale.sz * sizeof(float), NULL, &err);
    CHECK_ERROR(err);
    offset_d[d] = clCreateBuffer(context[d], CL_MEM_READ_WRITE, offset.sz * sizeof(float), NULL, &err);
    CHECK_ERROR(err);
    sum[d] = clCreateBuffer(context[d], CL_MEM_READ_WRITE, K * sizeof(float), NULL, &err);
    CHECK_ERROR(err);
    sqsum[d] = clCreateBuffer(context[d], CL_MEM_READ_WRITE, K * sizeof(float), NULL, &err);
    CHECK_ERROR(err);
  }

  for (int d = 0; d < DEVICE_NUM; d++) {
    err = clEnqueueWriteBuffer(queue[d], input_d[d], CL_TRUE, 0, input.sz * sizeof(float), input.buf, 0, NULL, NULL);
    CHECK_ERROR(err);
    err = clEnqueueWriteBuffer(queue[d], filter_d[d], CL_TRUE, 0, filter.sz * sizeof(float), filter.buf, 0, NULL, NULL);
    CHECK_ERROR(err);
    err = clEnqueueWriteBuffer(queue[d], bias_d[d], CL_TRUE, 0, bias.sz * sizeof(float), bias.buf, 0, NULL, NULL);
    CHECK_ERROR(err);
    err = clEnqueueWriteBuffer(queue[d], scale_d[d], CL_TRUE, 0, scale.sz * sizeof(float), scale.buf, 0, NULL, NULL);
    CHECK_ERROR(err);
    err = clEnqueueWriteBuffer(queue[d], offset_d[d], CL_TRUE, 0, offset.sz * sizeof(float), offset.buf, 0, NULL, NULL);
    CHECK_ERROR(err);
  }

  int H_ = H;
  int W_ = W;
  int C_ = C;
  int R_ = R;
  int S_ = S;
  int K_ = K;
  int OH_ = OH;
  int OW_ = OW;
  int stride_ = stride;
  int pad_ = pad;

  size_t K_p = 0;
  size_t OW_p = 0;
  LOG2S(K, K_p);
  LOG2S(OW, OW_p);
  const size_t K_mask = ((1 << K_p) - 1);
  const size_t OW_mask = ((1 << OW_p) - 1);

  for (int d = 0; d < DEVICE_NUM; d++) {
    err = clSetKernelArg(kernel[d][K_CONV2D_BATCHNORM_LEAKYRELU], 0, sizeof(cl_mem), &input_d[d]);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel[d][K_CONV2D_BATCHNORM_LEAKYRELU], 1, sizeof(cl_mem), &filter_d[d]);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel[d][K_CONV2D_BATCHNORM_LEAKYRELU], 2, sizeof(cl_mem), &bias_d[d]);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel[d][K_CONV2D_BATCHNORM_LEAKYRELU], 3, sizeof(cl_mem), &output_d[d]);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel[d][K_CONV2D_BATCHNORM_LEAKYRELU], 4, sizeof(int), &H_);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel[d][K_CONV2D_BATCHNORM_LEAKYRELU], 5, sizeof(int), &W_);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel[d][K_CONV2D_BATCHNORM_LEAKYRELU], 6, sizeof(int), &C_);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel[d][K_CONV2D_BATCHNORM_LEAKYRELU], 7, sizeof(int), &R_);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel[d][K_CONV2D_BATCHNORM_LEAKYRELU], 8, sizeof(int), &S_);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel[d][K_CONV2D_BATCHNORM_LEAKYRELU], 9, sizeof(int), &K_);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel[d][K_CONV2D_BATCHNORM_LEAKYRELU], 10, sizeof(int), &OH_);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel[d][K_CONV2D_BATCHNORM_LEAKYRELU], 11, sizeof(int), &OW_);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel[d][K_CONV2D_BATCHNORM_LEAKYRELU], 12, sizeof(int), &stride_);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel[d][K_CONV2D_BATCHNORM_LEAKYRELU], 13, sizeof(int), &pad_);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel[d][K_CONV2D_BATCHNORM_LEAKYRELU], 14, sizeof(int), &K_p);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel[d][K_CONV2D_BATCHNORM_LEAKYRELU], 15, sizeof(int), &OW_p);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel[d][K_CONV2D_BATCHNORM_LEAKYRELU], 16, sizeof(int), &K_mask);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel[d][K_CONV2D_BATCHNORM_LEAKYRELU], 17, sizeof(int), &OW_mask);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel[d][K_CONV2D_BATCHNORM_LEAKYRELU], 18, sizeof(float), &alpha);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel[d][K_CONV2D_BATCHNORM_LEAKYRELU], 19, sizeof(cl_mem), &scale_d[d]);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel[d][K_CONV2D_BATCHNORM_LEAKYRELU], 20, sizeof(cl_mem), &offset_d[d]);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel[d][K_CONV2D_BATCHNORM_LEAKYRELU], 21, 128 * sizeof(cl_float), NULL);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel[d][K_CONV2D_BATCHNORM_LEAKYRELU], 22, sizeof(cl_mem), &sum[d]);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel[d][K_CONV2D_BATCHNORM_LEAKYRELU], 23, sizeof(cl_mem), &sqsum[d]);
    CHECK_ERROR(err);
  }

  #ifdef FINISH
  for (int d = 0; d < DEVICE_NUM; d++) {
    clFinish(queue[d]);
  }
  #endif
  #ifdef SHOW_TIME
  END("conv2d write buffer")
  #endif

  #ifdef SHOW_TIME
  START_RE
  #endif
  size_t gws[1] = {OH * OW * K}, lws[1] = {128};
  for (int i = 0; i < 1; ++i) {
    gws[i] = (gws[i] + lws[i] - 1) / lws[i] * lws[i];
  }

  // Run kernel
  for (int d = 0; d < DEVICE_NUM; d++) {
    err = clEnqueueNDRangeKernel(queue[d], kernel[d][K_CONV2D_BATCHNORM_LEAKYRELU], 1, NULL, gws, lws, 0, NULL, NULL);
    CHECK_ERROR(err);
  }

  #ifdef FINISH
  for (int d = 0; d < DEVICE_NUM; d++) {
    clFinish(queue[d]);
  }
  #endif
  #ifdef SHOW_TIME
  END_RE("conv2d")
  #endif

  #ifdef SHOW_TIME
  START_RE
  #endif
  float sssum;
  // write
  for (int d = 0; d < DEVICE_NUM; d++) {
    err = clEnqueueReadBuffer(queue[d], output_d[d], CL_TRUE, 0, output.sz * sizeof(float), output.buf, 0, NULL, NULL);
    CHECK_ERROR(err);
    err = clEnqueueReadBuffer(queue[d], sum[d], CL_TRUE, 0, sizeof(float), &sssum, 0, NULL, NULL);
    CHECK_ERROR(err);
  }

  #ifdef FINISH
  for (int d = 0; d < DEVICE_NUM; d++) {
    clFinish(queue[d]);
  }
  #endif
  #ifdef SHOW_TIME
  END_RE("conv2d read buffer")
  #endif
}








































// one_image -> 최종
void encoding(
  int device_num,
  Tensor one_image,
  Tensor &encoded,
  std::map<std::string, Tensor> &weights
) {
  #ifdef SHOW_TIME
  float st, et;
  START_RE
  #endif
  cl_mem A = clCreateBuffer(context[device_num], CL_MEM_READ_WRITE, 1024 * 1024 * 8 * sizeof(float), NULL, &err);
  CHECK_ERROR(err);
  cl_mem B = clCreateBuffer(context[device_num], CL_MEM_READ_WRITE, 1024 * 1024 * 8 * sizeof(float), NULL, &err);
  CHECK_ERROR(err);

  cl_mem S[9];
  for (int i = 0; i < 9; i++) {
    S[i] = clCreateBuffer(context[device_num], CL_MEM_READ_WRITE, 1024 * 1024 * 8 * sizeof(float), NULL, &err);
    CHECK_ERROR(err);
  }

  clEnqueueWriteBuffer(queue[device_num], S[0], CL_TRUE, 0, one_image.sz * sizeof(float), one_image.buf, 0, NULL, NULL);
  CHECK_ERROR(err);

  #ifdef FINISH
  clFinish(queue[device_num]);
  #endif
  #ifdef SHOW_TIME
  END_RE("0 -> 1 write first buffer");
  #endif

  size_t H_ = one_image.shape[0], W_ = one_image.shape[1], C_ = one_image.shape[2];
  size_t OH_, OW_, K_; 
  // 1 -> 2
  // conv2d
  // leakyrelu
  //
  {
    {
      auto filter = weights["generator/encoder_1/conv2d/kernel"];
      auto bias = weights["generator/encoder_1/conv2d/bias"];
      conv2d_kernel(device_num, S[0], S[1], filter, bias, H_, W_, C_);
    }

    { // leakyrelu (i = 2)
      cl_mem &input = S[1];
      cl_mem &output = B;
      leakyrelu(device_num, input, output, 0.2f, H_, W_, C_);
    }
  }
  // 2 -> 8
  // conv2d
  // mean
  // variance
  // batchnorm
  // leakyrelu
  for (int step = 2; step < 8; step++) {
    auto scope = "generator/encoder_" + std::to_string(step);
    auto filter = weights[scope + "/conv2d/kernel"];
    auto bias = weights[scope + "/conv2d/bias"];
    auto scale = weights[scope + "/batch_normalization/gamma"];
    auto offset = weights[scope + "/batch_normalization/beta"];

    #ifdef SHOW_TIME
    START_RE
    #endif
    cl_mem scale_mem = clCreateBuffer(context[device_num], CL_MEM_READ_WRITE, scale.sz * sizeof(float), NULL, &err);
    CHECK_ERROR(err);
    cl_mem offset_mem = clCreateBuffer(context[device_num], CL_MEM_READ_WRITE, offset.sz * sizeof(float), NULL, &err);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(queue[device_num], scale_mem, CL_TRUE, 0, scale.sz * sizeof(float), scale.buf, 0, NULL, NULL);
    CHECK_ERROR(err);
    err = clEnqueueWriteBuffer(queue[device_num], offset_mem, CL_TRUE, 0, offset.sz * sizeof(float), offset.buf, 0, NULL, NULL);
    CHECK_ERROR(err);
    #ifdef SHOW_TIME
    // 이거 중복 아닌가?
    END_RE("3 -> 4 write filter bias scale offset")
    #endif

    { // conv2d
      cl_mem &input = B;
      cl_mem &output = A;
      conv2d_kernel(device_num, input, output, filter, bias, H_, W_, C_);
    }

    cl_mem mean_mem = clCreateBuffer(context[device_num], CL_MEM_READ_WRITE, C_ * sizeof(float), NULL, &err);
    CHECK_ERROR(err);
    cl_mem variance_mem = clCreateBuffer(context[device_num], CL_MEM_READ_WRITE, C_ * sizeof(float), NULL, &err);
    CHECK_ERROR(err);
    { // mean
      cl_mem &input = A;
      mean_kernel(device_num, input, mean_mem, H_, W_, C_);
    }

    { // variance
      cl_mem &input = A;
      variance_kernel(device_num, input, mean_mem, variance_mem, H_, W_, C_);
    }

    { // batchnorm
      #ifdef SHOW_TIME
      START_RE
      #endif
      size_t H = H_;
      size_t W = W_;
      size_t C = C_;
      size_t K_p = 0;
      LOG2S(C, K_p);
      const size_t K_mask = ((1 << K_p) - 1);
      float alpha = 0.2f;

      cl_mem &input = A;
      cl_mem &output = S[step];

      err = clSetKernelArg(kernel[device_num][K_BATCHNORM_LEAKYRELU], 0, sizeof(cl_mem), &input);
      CHECK_ERROR(err);
      err = clSetKernelArg(kernel[device_num][K_BATCHNORM_LEAKYRELU], 1, sizeof(cl_mem), &mean_mem);
      CHECK_ERROR(err);
      err = clSetKernelArg(kernel[device_num][K_BATCHNORM_LEAKYRELU], 2, sizeof(cl_mem), &variance_mem);
      CHECK_ERROR(err);
      err = clSetKernelArg(kernel[device_num][K_BATCHNORM_LEAKYRELU], 3, sizeof(cl_mem), &output);
      CHECK_ERROR(err);
      err = clSetKernelArg(kernel[device_num][K_BATCHNORM_LEAKYRELU], 4, sizeof(cl_mem), &offset_mem);
      CHECK_ERROR(err);
      err = clSetKernelArg(kernel[device_num][K_BATCHNORM_LEAKYRELU], 5, sizeof(cl_mem), &scale_mem);
      CHECK_ERROR(err);
      err = clSetKernelArg(kernel[device_num][K_BATCHNORM_LEAKYRELU], 6, sizeof(int), &K_mask);
      CHECK_ERROR(err);
      err = clSetKernelArg(kernel[device_num][K_BATCHNORM_LEAKYRELU], 7, sizeof(int), &alpha);
      CHECK_ERROR(err);

      size_t gws[1] = {H * W * C}, lws[1] = {128};
      for (int i = 0; i < 1; ++i) {
        gws[i] = (gws[i] + lws[i] - 1) / lws[i] * lws[i];
      }

      err = clEnqueueNDRangeKernel(queue[device_num], kernel[device_num][K_BATCHNORM_LEAKYRELU], 1, NULL, gws, lws, 0, NULL, NULL);
      CHECK_ERROR(err);
      #ifdef FINISH
      clFinish(queue[device_num]);
      #endif
      #ifdef SHOW_TIME
      END_RE("3 -> 4 run kernel batchnorm")
      #endif
    }

    { // leakyrelu (i = step)
      #ifdef SHOW_TIME
      START_RE
      #endif
      size_t H = H_;
      size_t W = W_;
      size_t C = C_;
      float alpha = 0.2f;

      cl_mem &input = S[step];
      cl_mem &output = B;

      err = clSetKernelArg(kernel[device_num][K_LEAKYRELU], 0, sizeof(cl_mem), &input);
      CHECK_ERROR(err);
      err = clSetKernelArg(kernel[device_num][K_LEAKYRELU], 1, sizeof(cl_mem), &output);
      CHECK_ERROR(err);
      err = clSetKernelArg(kernel[device_num][K_LEAKYRELU], 2, sizeof(float), &alpha);
      CHECK_ERROR(err);

      size_t gws[1] = {H * W * C}, lws[1] = {128};
      for (int i = 0; i < 1; ++i) {
        gws[i] = (gws[i] + lws[i] - 1) / lws[i] * lws[i];
      }

      err = clEnqueueNDRangeKernel(queue[device_num], kernel[device_num][K_LEAKYRELU], 1, NULL, gws, lws, 0, NULL, NULL);
      CHECK_ERROR(err);
      #ifdef FINISH
      clFinish(queue[device_num]);
      #endif
      #ifdef SHOW_TIME
      END_RE("<step> run kernel leakyrelu")
      #endif
    }
    clReleaseMemObject(scale_mem);
    clReleaseMemObject(offset_mem);

    clReleaseMemObject(mean_mem);
    clReleaseMemObject(variance_mem);
  }
  // 8 -> 9
  // conv2d
  // mean
  // variance
  // batchnorm
  {
    auto scope = "generator/encoder_" + std::to_string(8);
    auto filter = weights[scope + "/conv2d/kernel"];
    auto bias = weights[scope + "/conv2d/bias"];
    auto scale = weights[scope + "/batch_normalization/gamma"];
    auto offset = weights[scope + "/batch_normalization/beta"];

    #ifdef SHOW_TIME
    START_RE
    #endif
    cl_mem filter_mem = clCreateBuffer(context[device_num], CL_MEM_READ_WRITE, filter.sz * sizeof(float), NULL, &err);
    CHECK_ERROR(err);
    cl_mem bias_mem = clCreateBuffer(context[device_num], CL_MEM_READ_WRITE, bias.sz * sizeof(float), NULL, &err);
    CHECK_ERROR(err);
    cl_mem scale_mem = clCreateBuffer(context[device_num], CL_MEM_READ_WRITE, scale.sz * sizeof(float), NULL, &err);
    CHECK_ERROR(err);
    cl_mem offset_mem = clCreateBuffer(context[device_num], CL_MEM_READ_WRITE, offset.sz * sizeof(float), NULL, &err);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(queue[device_num], filter_mem, CL_TRUE, 0, filter.sz * sizeof(float), filter.buf, 0, NULL, NULL);
    CHECK_ERROR(err);
    err = clEnqueueWriteBuffer(queue[device_num], bias_mem, CL_TRUE, 0, bias.sz * sizeof(float), bias.buf, 0, NULL, NULL);
    CHECK_ERROR(err);
    err = clEnqueueWriteBuffer(queue[device_num], scale_mem, CL_TRUE, 0, scale.sz * sizeof(float), scale.buf, 0, NULL, NULL);
    CHECK_ERROR(err);
    err = clEnqueueWriteBuffer(queue[device_num], offset_mem, CL_TRUE, 0, offset.sz * sizeof(float), offset.buf, 0, NULL, NULL);
    CHECK_ERROR(err);
    #ifdef SHOW_TIME
    END_RE("8 -> 9 write filter bias scale offset")
    #endif

    { // conv2d
      #ifdef SHOW_TIME
      START_RE
      #endif
      size_t H = H_, W = W_, C = C_;
      size_t R = filter.shape[0], S = filter.shape[1], K = filter.shape[3];
      const size_t stride = 2, pad = 1;
      size_t OH = H / stride, OW = W / stride;
      size_t K_p = 0;
      size_t OW_p = 0;
      LOG2S(K, K_p);
      LOG2S(OW, OW_p);
      const size_t K_mask = ((1 << K_p) - 1);
      const size_t OW_mask = ((1 << OW_p) - 1);
      float alpha = 0.2f;

      cl_mem &input = B;
      cl_mem &output = A;
      err = clSetKernelArg(kernel[device_num][K_CONV2D], 0, sizeof(cl_mem), &input);
      CHECK_ERROR(err);
      err = clSetKernelArg(kernel[device_num][K_CONV2D], 1, sizeof(cl_mem), &filter_mem);
      CHECK_ERROR(err);
      err = clSetKernelArg(kernel[device_num][K_CONV2D], 2, sizeof(cl_mem), &bias_mem);
      CHECK_ERROR(err);
      err = clSetKernelArg(kernel[device_num][K_CONV2D], 3, sizeof(cl_mem), &output);
      CHECK_ERROR(err);
      err = clSetKernelArg(kernel[device_num][K_CONV2D], 4, sizeof(int), &H);
      CHECK_ERROR(err);
      err = clSetKernelArg(kernel[device_num][K_CONV2D], 5, sizeof(int), &W);
      CHECK_ERROR(err);
      err = clSetKernelArg(kernel[device_num][K_CONV2D], 6, sizeof(int), &C);
      CHECK_ERROR(err);
      err = clSetKernelArg(kernel[device_num][K_CONV2D], 7, sizeof(int), &R);
      CHECK_ERROR(err);
      err = clSetKernelArg(kernel[device_num][K_CONV2D], 8, sizeof(int), &S);
      CHECK_ERROR(err);
      err = clSetKernelArg(kernel[device_num][K_CONV2D], 9, sizeof(int), &K);
      CHECK_ERROR(err);
      err = clSetKernelArg(kernel[device_num][K_CONV2D], 10, sizeof(int), &OH);
      CHECK_ERROR(err);
      err = clSetKernelArg(kernel[device_num][K_CONV2D], 11, sizeof(int), &OW);
      CHECK_ERROR(err);
      err = clSetKernelArg(kernel[device_num][K_CONV2D], 12, sizeof(int), &stride);
      CHECK_ERROR(err);
      err = clSetKernelArg(kernel[device_num][K_CONV2D], 13, sizeof(int), &pad);
      CHECK_ERROR(err);
      err = clSetKernelArg(kernel[device_num][K_CONV2D], 14, sizeof(int), &K_p);
      CHECK_ERROR(err);
      err = clSetKernelArg(kernel[device_num][K_CONV2D], 15, sizeof(int), &OW_p);
      CHECK_ERROR(err);
      err = clSetKernelArg(kernel[device_num][K_CONV2D], 16, sizeof(int), &K_mask);
      CHECK_ERROR(err);
      err = clSetKernelArg(kernel[device_num][K_CONV2D], 17, sizeof(int), &OW_mask);
      CHECK_ERROR(err);

      size_t gws[1] = {OH * OW * K}, lws[1] = {128};
      for (int i = 0; i < 1; ++i) {
        gws[i] = (gws[i] + lws[i] - 1) / lws[i] * lws[i];
      }

      err = clEnqueueNDRangeKernel(queue[device_num], kernel[device_num][K_CONV2D], 1, NULL, gws, lws, 0, NULL, NULL);
      CHECK_ERROR(err);
      #ifdef FINISH
      clFinish(queue[device_num]);
      #endif
      #ifdef SHOW_TIME
      END_RE("8 -> 9 run kernel conv2d")
      #endif

      // printf("%lu %lu %lu %lu %lu %lu | %lu %lu %lu %lu\n", H, W, C, OH, OW, K, OH, OW, stride, pas);
      // printf("%f %f\n", filter.buf[0], bias.buf[0]);

      H_ = OH;
      W_ = OW;
      C_ = K;
    }

    cl_mem mean_mem = clCreateBuffer(context[device_num], CL_MEM_READ_WRITE, C_ * sizeof(float), NULL, &err);
    CHECK_ERROR(err);
    cl_mem variance_mem = clCreateBuffer(context[device_num], CL_MEM_READ_WRITE, C_ * sizeof(float), NULL, &err);
    CHECK_ERROR(err);
    { // mean
      #ifdef SHOW_TIME
      START_RE
      #endif
      size_t H = H_;
      size_t W = W_;
      size_t C = C_;

      cl_mem &input = A;

      err = clSetKernelArg(kernel[device_num][K_MEAN], 0, sizeof(cl_mem), &input);
      CHECK_ERROR(err);
      err = clSetKernelArg(kernel[device_num][K_MEAN], 1, sizeof(cl_mem), &mean_mem);
      CHECK_ERROR(err);
      err = clSetKernelArg(kernel[device_num][K_MEAN], 2, sizeof(int), &H);
      CHECK_ERROR(err);
      err = clSetKernelArg(kernel[device_num][K_MEAN], 3, sizeof(int), &W);
      CHECK_ERROR(err);
      err = clSetKernelArg(kernel[device_num][K_MEAN], 4, sizeof(int), &C);
      CHECK_ERROR(err);

      size_t gws[1] = {C}, lws[1] = {C};
      for (int i = 0; i < 1; ++i) {
        gws[i] = (gws[i] + lws[i] - 1) / lws[i] * lws[i];
      }

      err = clEnqueueNDRangeKernel(queue[device_num], kernel[device_num][K_MEAN], 1, NULL, gws, lws, 0, NULL, NULL);
      CHECK_ERROR(err);
      #ifdef FINISH
      clFinish(queue[device_num]);
      #endif
      #ifdef SHOW_TIME
      END_RE("8 -> 9 run kernel mean")
      #endif
    }

    { // variance
      #ifdef SHOW_TIME
      START_RE
      #endif
      size_t H = H_;
      size_t W = W_;
      size_t C = C_;

      cl_mem &input = A;

      err = clSetKernelArg(kernel[device_num][K_VARIANCE], 0, sizeof(cl_mem), &input);
      CHECK_ERROR(err);
      err = clSetKernelArg(kernel[device_num][K_VARIANCE], 1, sizeof(cl_mem), &mean_mem);
      CHECK_ERROR(err);
      err = clSetKernelArg(kernel[device_num][K_VARIANCE], 2, sizeof(cl_mem), &variance_mem);
      CHECK_ERROR(err);
      err = clSetKernelArg(kernel[device_num][K_VARIANCE], 3, sizeof(int), &H);
      CHECK_ERROR(err);
      err = clSetKernelArg(kernel[device_num][K_VARIANCE], 4, sizeof(int), &W);
      CHECK_ERROR(err);
      err = clSetKernelArg(kernel[device_num][K_VARIANCE], 5, sizeof(int), &C);
      CHECK_ERROR(err);

      size_t gws[1] = {C}, lws[1] = {C};
      for (int i = 0; i < 1; ++i) {
        gws[i] = (gws[i] + lws[i] - 1) / lws[i] * lws[i];
      }

      err = clEnqueueNDRangeKernel(queue[device_num], kernel[device_num][K_VARIANCE], 1, NULL, gws, lws, 0, NULL, NULL);
      CHECK_ERROR(err);
      #ifdef FINISH
      clFinish(queue[device_num]);
      #endif
      #ifdef SHOW_TIME
      END_RE("8 -> 9 run kernel variance")
      #endif
    }

    { // batchnorm_relu
      #ifdef SHOW_TIME
      START_RE
      #endif
      size_t H = H_;
      size_t W = W_;
      size_t C = C_;
      size_t K_p = 0;
      LOG2S(C, K_p);
      const size_t K_mask = ((1 << K_p) - 1);

      cl_mem &input = A;
      cl_mem &output = B;

      err = clSetKernelArg(kernel[device_num][K_BATCHNORM], 0, sizeof(cl_mem), &input);
      CHECK_ERROR(err);
      err = clSetKernelArg(kernel[device_num][K_BATCHNORM], 1, sizeof(cl_mem), &mean_mem);
      CHECK_ERROR(err);
      err = clSetKernelArg(kernel[device_num][K_BATCHNORM], 2, sizeof(cl_mem), &variance_mem);
      CHECK_ERROR(err);
      err = clSetKernelArg(kernel[device_num][K_BATCHNORM], 3, sizeof(cl_mem), &output);
      CHECK_ERROR(err);
      err = clSetKernelArg(kernel[device_num][K_BATCHNORM], 4, sizeof(cl_mem), &offset_mem);
      CHECK_ERROR(err);
      err = clSetKernelArg(kernel[device_num][K_BATCHNORM], 5, sizeof(cl_mem), &scale_mem);
      CHECK_ERROR(err);
      err = clSetKernelArg(kernel[device_num][K_BATCHNORM], 6, sizeof(int), &K_mask);
      CHECK_ERROR(err);

      size_t gws[1] = {H * W * C}, lws[1] = {128};
      for (int i = 0; i < 1; ++i) {
        gws[i] = (gws[i] + lws[i] - 1) / lws[i] * lws[i];
      }

      err = clEnqueueNDRangeKernel(queue[device_num], kernel[device_num][K_BATCHNORM], 1, NULL, gws, lws, 0, NULL, NULL);
      CHECK_ERROR(err);
      #ifdef FINISH
      clFinish(queue[device_num]);
      #endif
      #ifdef SHOW_TIME
      END_RE("8 -> 9 run kernel batchnorm")
      #endif
    }

    clReleaseMemObject(filter_mem);
    clReleaseMemObject(bias_mem);

    clReleaseMemObject(scale_mem);
    clReleaseMemObject(offset_mem);

    clReleaseMemObject(mean_mem);
    clReleaseMemObject(variance_mem);
  }

  #ifdef SHOW_TIME
  START_RE
  #endif
  encoded.alloc_once({H_, W_, C_});
  err = clEnqueueReadBuffer(queue[device_num], B, CL_TRUE, 0, encoded.sz * sizeof(float), encoded.buf, 0, NULL, NULL);
  CHECK_ERROR(err);
  #ifdef FINISH
  clFinish(queue[device_num]);
  #endif
  #ifdef SHOW_TIME
  END_RE("read")
  #endif

  #ifdef SHOW_TIME
  START_RE
  #endif
  clReleaseMemObject(A);
  clReleaseMemObject(B);
  for (int i = 0; i < 9; i++) {
    clReleaseMemObject(S[i]);
  }
  #ifdef SHOW_TIME
  END_RE("release mem object")
  #endif
}

void conv2d_kernel(int device_num, cl_mem &input, cl_mem &output, Tensor filter, Tensor bias, size_t &H, size_t &W, size_t &C) {
  size_t R = filter.shape[0], S = filter.shape[1], K = filter.shape[3];
  const size_t stride = 2, pad = 1;
  size_t OH = H / stride, OW = W / stride;
  #ifdef SHOW_TIME
  float st, et;
  START_RE
  #endif
  cl_mem filter_mem = clCreateBuffer(context[device_num], CL_MEM_READ_WRITE, filter.sz * sizeof(float), NULL, &err);
  CHECK_ERROR(err);
  cl_mem bias_mem = clCreateBuffer(context[device_num], CL_MEM_READ_WRITE, bias.sz * sizeof(float), NULL, &err);
  CHECK_ERROR(err);

  err = clEnqueueWriteBuffer(queue[device_num], filter_mem, CL_TRUE, 0, filter.sz * sizeof(float), filter.buf, 0, NULL, NULL);
  CHECK_ERROR(err);
  err = clEnqueueWriteBuffer(queue[device_num], bias_mem, CL_TRUE, 0, bias.sz * sizeof(float), bias.buf, 0, NULL, NULL);
  CHECK_ERROR(err);
  #ifdef SHOW_TIME
  END_RE("1 -> 2 write filter bias")
  #endif

  #ifdef SHOW_TIME
  START_RE
  #endif
  size_t K_p = 0;
  size_t OW_p = 0;
  LOG2S(K, K_p);
  LOG2S(OW, OW_p);
  const size_t K_mask = ((1 << K_p) - 1);
  const size_t OW_mask = ((1 << OW_p) - 1);

  err = clSetKernelArg(kernel[device_num][K_CONV2D], 0, sizeof(cl_mem), &input);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel[device_num][K_CONV2D], 1, sizeof(cl_mem), &filter_mem);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel[device_num][K_CONV2D], 2, sizeof(cl_mem), &bias_mem);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel[device_num][K_CONV2D], 3, sizeof(cl_mem), &output);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel[device_num][K_CONV2D], 4, sizeof(int), &H);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel[device_num][K_CONV2D], 5, sizeof(int), &W);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel[device_num][K_CONV2D], 6, sizeof(int), &C);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel[device_num][K_CONV2D], 7, sizeof(int), &R);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel[device_num][K_CONV2D], 8, sizeof(int), &S);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel[device_num][K_CONV2D], 9, sizeof(int), &K);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel[device_num][K_CONV2D], 10, sizeof(int), &OH);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel[device_num][K_CONV2D], 11, sizeof(int), &OW);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel[device_num][K_CONV2D], 12, sizeof(int), &stride);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel[device_num][K_CONV2D], 13, sizeof(int), &pad);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel[device_num][K_CONV2D], 14, sizeof(int), &K_p);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel[device_num][K_CONV2D], 15, sizeof(int), &OW_p);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel[device_num][K_CONV2D], 16, sizeof(int), &K_mask);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel[device_num][K_CONV2D], 17, sizeof(int), &OW_mask);
  CHECK_ERROR(err);

  size_t gws[1] = {OH * OW * K}, lws[1] = {128};
  for (int i = 0; i < 1; ++i) {
    gws[i] = (gws[i] + lws[i] - 1) / lws[i] * lws[i];
  }

  err = clEnqueueNDRangeKernel(queue[device_num], kernel[device_num][K_CONV2D], 1, NULL, gws, lws, 0, NULL, NULL);
  CHECK_ERROR(err);
  #ifdef FINISH
  clFinish(queue[device_num]);
  #endif
  #ifdef SHOW_TIME
  END_RE("run kernel conv2d_leakyrelu")
  #endif

  clReleaseMemObject(filter_mem);
  clReleaseMemObject(bias_mem);

  H = OH;
  W = OW;
  C = K;
}

void leakyrelu(int device_num, cl_mem &input, cl_mem &output, float alpha, size_t H, size_t W, size_t C) {
  #ifdef SHOW_TIME
  float st, et;
  START_RE
  #endif
  err = clSetKernelArg(kernel[device_num][K_LEAKYRELU], 0, sizeof(cl_mem), &input);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel[device_num][K_LEAKYRELU], 1, sizeof(cl_mem), &output);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel[device_num][K_LEAKYRELU], 2, sizeof(float), &alpha);
  CHECK_ERROR(err);

  size_t gws[1] = {H * W * C}, lws[1] = {128};
  for (int i = 0; i < 1; ++i) {
    gws[i] = (gws[i] + lws[i] - 1) / lws[i] * lws[i];
  }

  err = clEnqueueNDRangeKernel(queue[device_num], kernel[device_num][K_LEAKYRELU], 1, NULL, gws, lws, 0, NULL, NULL);
  CHECK_ERROR(err);
  #ifdef FINISH
  clFinish(queue[device_num]);
  #endif
  #ifdef SHOW_TIME
  END_RE("run kernel leakyrelu")
  #endif
}

void mean_kernel(int device_num, cl_mem &input, cl_mem &mean_mem, size_t &H, size_t &W, size_t &C) {
  #ifdef SHOW_TIME
  float st, et;
  START_RE
  #endif
  err = clSetKernelArg(kernel[device_num][K_MEAN], 0, sizeof(cl_mem), &input);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel[device_num][K_MEAN], 1, sizeof(cl_mem), &mean_mem);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel[device_num][K_MEAN], 2, sizeof(int), &H);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel[device_num][K_MEAN], 3, sizeof(int), &W);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel[device_num][K_MEAN], 4, sizeof(int), &C);
  CHECK_ERROR(err);

  size_t gws[1] = {C}, lws[1] = {C};
  for (int i = 0; i < 1; ++i) {
    gws[i] = (gws[i] + lws[i] - 1) / lws[i] * lws[i];
  }

  err = clEnqueueNDRangeKernel(queue[device_num], kernel[device_num][K_MEAN], 1, NULL, gws, lws, 0, NULL, NULL);
  CHECK_ERROR(err);
  #ifdef FINISH
  clFinish(queue[device_num]);
  #endif
  #ifdef SHOW_TIME
  END_RE("run kernel mean")
  #endif
}

void variance_kernel(int device_num, cl_mem &input, cl_mem &mean_mem, cl_mem &variance_mem, size_t H, size_t W, size_t C) {
  #ifdef SHOW_TIME
  float st; et;
  START_RE
  #endif

  err = clSetKernelArg(kernel[device_num][K_VARIANCE], 0, sizeof(cl_mem), &input);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel[device_num][K_VARIANCE], 1, sizeof(cl_mem), &mean_mem);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel[device_num][K_VARIANCE], 2, sizeof(cl_mem), &variance_mem);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel[device_num][K_VARIANCE], 3, sizeof(int), &H);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel[device_num][K_VARIANCE], 4, sizeof(int), &W);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel[device_num][K_VARIANCE], 5, sizeof(int), &C);
  CHECK_ERROR(err);

  size_t gws[1] = {C}, lws[1] = {C};
  for (int i = 0; i < 1; ++i) {
    gws[i] = (gws[i] + lws[i] - 1) / lws[i] * lws[i];
  }

  err = clEnqueueNDRangeKernel(queue[device_num], kernel[device_num][K_VARIANCE], 1, NULL, gws, lws, 0, NULL, NULL);
  CHECK_ERROR(err);
  #ifdef FINISH
  clFinish(queue[device_num]);
  #endif
  #ifdef SHOW_TIME
  END_RE("run kernel variance")
  #endif
}
