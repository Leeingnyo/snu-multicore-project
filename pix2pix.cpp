#include "pix2pix.h"

#include "util.h"

#include <immintrin.h> // include vector
#include <omp.h>
#include <CL/cl.h>
#include <mpi.h>

#include <string>
#include <map>
#include <cmath>
#include <mutex>

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
#define KERNEL_NUM 8

static cl_int err;
static cl_platform_id platform;
static cl_device_id device[DEVICE_NUM];
static cl_context context[DEVICE_NUM];
static cl_command_queue queue[DEVICE_NUM];
static cl_program program[DEVICE_NUM];
static cl_kernel kernel[DEVICE_NUM][KERNEL_NUM];
enum kernel_type { K_CONV2D, K_CONV2D_TRANSPOSED, K_MEAN, K_VARIANCE, K_BATCHNORM, K_LEAKYRELU, K_CONCAT, K_TANH };

int num_threads = DEVICE_NUM;

std::map<std::string, cl_mem> weight_buffers[DEVICE_NUM];
std::map<std::string, bool> weight_buffers_bound[DEVICE_NUM];
std::mutex mut[DEVICE_NUM];

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
static void elem_tanh(Tensor input, Tensor &output);

void pix2pix_iter(int device_num, Tensor one_image, Tensor &encoded, std::map<std::string, Tensor> &weights);
void conv2d_kernel(int device_num, cl_mem &input, cl_mem &output, cl_mem &filter_mem, cl_mem &bias_mem, size_t &H, size_t &W, size_t &C, size_t R, size_t S, size_t K);
void leakyrelu(int device_num, cl_mem &input, cl_mem &output, float alpha, size_t H, size_t W, size_t C);
void tanh_kernel(int device_num, cl_mem &input, cl_mem &output, size_t H, size_t W, size_t C);
void mean_kernel(int device_num, cl_mem &input, cl_mem &mean_mem, size_t &H, size_t &W, size_t &C);
void variance_kernel(int device_num, cl_mem &input, cl_mem &mean_mem, cl_mem &variance_mem, size_t H, size_t W, size_t C);
void batchnorm_kernel(int device_num, cl_mem &input, cl_mem &mean_mem, cl_mem &variance_mem, cl_mem &output, cl_mem &offset_mem, cl_mem &scale_mem, size_t H, size_t W, size_t C);
void conv2d_transposed_kernel(int device_num, cl_mem &input, cl_mem &output, cl_mem &filter_mem, cl_mem &bias_mem, size_t &H, size_t &W, size_t &C, size_t R, size_t S, size_t K);
void concat_kernel(int device_num, cl_mem &input, cl_mem &input2, cl_mem &output, int H, int W, int C0, int C1, size_t &C);

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

cl_mem A[DEVICE_NUM];
cl_mem B[DEVICE_NUM];
cl_mem intermediate[DEVICE_NUM][9];

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
    kernel[d][K_MEAN] = clCreateKernel(program[d], "mean", &err);
    kernel[d][K_VARIANCE] = clCreateKernel(program[d], "variance", &err);
    kernel[d][K_BATCHNORM] = clCreateKernel(program[d], "batchnorm", &err);
    kernel[d][K_LEAKYRELU] = clCreateKernel(program[d], "leakyrelu", &err);
    kernel[d][K_CONCAT] = clCreateKernel(program[d], "concat", &err);
    kernel[d][K_TANH] = clCreateKernel(program[d], "elem_tanh", &err);
    CHECK_ERROR(err);
  }

  for (int d = 0; d < DEVICE_NUM; d++) {
    A[d] = clCreateBuffer(context[d], CL_MEM_READ_WRITE, 1024 * 1024 * 8 * sizeof(float), NULL, &err);
    CHECK_ERROR(err);
    B[d] = clCreateBuffer(context[d], CL_MEM_READ_WRITE, 1024 * 1024 * 8 * sizeof(float), NULL, &err);
    CHECK_ERROR(err);

    for (int i = 0; i < 9; i++) {
      intermediate[d][i] = clCreateBuffer(context[d], CL_MEM_READ_WRITE, 1024 * 1024 * 8 * sizeof(float), NULL, &err);
      CHECK_ERROR(err);
    }
  }

  // alloc buffers
  // alloc mpi buffers
  for (int d = 0; d < DEVICE_NUM; d++) {
    clFinish(queue[d]);
  }
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

  #ifdef USE_MPI
  int RANKS;
  MPI_Comm_size(MPI_COMM_WORLD, &RANKS);
  printf("mpi %d\n", RANKS);
  size_t one_image_sz = 256 * 256 * 3;
  int rank = get_rank();

  size_t max_num_image = (num_image + RANKS - 1) / RANKS;
  size_t my_num_image = max_num_image;
  if (num_image % max_num_image) {
    if (rank >= num_image % max_num_image) {
      my_num_image--;
    }
  }
  size_t input_sz = my_num_image * (one_image_sz * sizeof(uint8_t));
  
  if (!input_buf) {
    input_buf = (uint8_t *)malloc(input_sz);
  }
  if (!output_buf) {
    output_buf = (uint8_t *)malloc(input_sz);
  }
  if (!weight_buf) {
    weight_buf = (float *)malloc(54431363 * sizeof(float));
  }

  printf("before %d my_num_image %d \n", my_num_image);

  /*
  if (rank) {
    MPI_Recv(weight_buf, 54431363, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, NULL);
  } else {
    for (int r = 1; r < RANKS; r++) {
      MPI_Send(weight_buf, 54431363, MPI_FLOAT, r, 0, MPI_COMM_WORLD);
    }
  }
  */
  MPI_Bcast(weight_buf, 54431363, MPI_FLOAT, 0, MPI_COMM_WORLD);
  printf("B cast %d \n", rank);

  if (rank) {
    printf("%d wait\n", rank);
    printf("%p\n", input_buf);
    int err = MPI_Recv(input_buf, my_num_image * one_image_sz, MPI_UINT8_T, 0, 0, MPI_COMM_WORLD, NULL);
    printf("%d recv %d\n", rank, err);
  } else {
    int sum = my_num_image;
    for (int r = 1; r < RANKS; r++) {
      printf("%d send", r);
      size_t ye_num_image = max_num_image;
      if (num_image % max_num_image && r >= num_image % max_num_image) ye_num_image--;
      int err = MPI_Send(input_buf + sum * one_image_sz, one_image_sz * ye_num_image, MPI_UINT8_T, r, 0, MPI_COMM_WORLD);
      sum += ye_num_image;
      printf("%d done %d ye_num_image %d sum %d", r, err, ye_num_image, sum);
    }
  }
  printf("A cast %d \n", rank);
  #endif

  auto weights = register_weights(weight_buf); // Memory allocated for weights
  #ifdef USE_MPI
  auto input = preprocess(input_buf, my_num_image); // Memory allocated for input
  #else
  auto input = preprocess(input_buf, num_image); // Memory allocated for input
  #endif

  // Declare feature maps
  // Memory for feature maps are allocated when they are written first time using Tensor::alloc_once(...)

  #if DEVICE_NUM == 1
  #else
  #pragma omp parallel for num_threads(num_threads)
  #endif
  #ifdef USE_MPI
  for (size_t img_idx = 0; img_idx < my_num_image; ++img_idx) {
  #else
  for (size_t img_idx = 0; img_idx < num_image; ++img_idx) {
  #endif
    Tensor one_image;

    // Pick 1 image out of num_image
    get_one_image(input, one_image, img_idx);

    /*
     * Encoding phase
     */
    Tensor processed;
    pix2pix_iter(omp_get_thread_num(), one_image, processed, weights);

    // Put a image into output buffer
    postprocess_one_image(processed, output_buf, img_idx);
  }

  #ifdef USE_MPI
  if (rank) {
    printf("%d wait\n", rank);
    printf("%p\n", input_buf);
    printf("%d recv %d\n", rank, err);
    int err = MPI_Send(output_buf, one_image_sz * my_num_image, MPI_UINT8_T, 0, 0, MPI_COMM_WORLD);
  } else {
    int sum = my_num_image;
    for (int r = 1; r < RANKS; r++) {
      printf("%d send", r);
      size_t ye_num_image = max_num_image;
      if (num_image % max_num_image && r >= num_image % max_num_image) ye_num_image--;
      int err = MPI_Recv(output_buf + sum * one_image_sz, ye_num_image * one_image_sz, MPI_UINT8_T, 0, 0, MPI_COMM_WORLD, NULL);
      sum += ye_num_image;
      printf("%d done %d ye_num_image %d sum %d", r, err, ye_num_image, sum);
    }
  }
  printf("A gather %d \n", rank);

  #endif
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

// one_image -> 최종
void pix2pix_iter(
  int device_num,
  Tensor one_image,
  Tensor &encoded,
  std::map<std::string, Tensor> &weights
) {
  #ifdef SHOW_TIME
  double st, et;
  START_RE
  #endif

  clEnqueueWriteBuffer(queue[device_num], intermediate[device_num][0], CL_TRUE, 0, one_image.sz * sizeof(float), one_image.buf, 0, NULL, NULL);
  CHECK_ERROR(err);

  #ifdef FINISH
  clFinish(queue[device_num]);
  #endif
  #ifdef SHOW_TIME
  END_RE("0 -> 1 write first buffer");
  #endif

  size_t H_ = one_image.shape[0], W_ = one_image.shape[1], C_ = one_image.shape[2];
  // 1 -> 2
  // conv2d
  // leakyrelu
  //
  {
    {
      #ifdef SHOW_TIME
      double st, et;
      START_RE
      #endif
      auto filter = weights["generator/encoder_1/conv2d/kernel"];
      auto bias = weights["generator/encoder_1/conv2d/bias"];

      mut[device_num].lock();
      if (!weight_buffers_bound[device_num]["generator/encoder_1/conv2d/kernel"]) {
        weight_buffers[device_num]["generator/encoder_1/conv2d/kernel"] = clCreateBuffer(context[device_num], CL_MEM_READ_ONLY, filter.sz * sizeof(float), NULL, &err);
        CHECK_ERROR(err);
        cl_mem &filter_mem = weight_buffers[device_num]["generator/encoder_1/conv2d/kernel"];
        err = clEnqueueWriteBuffer(queue[device_num], filter_mem, CL_TRUE, 0, filter.sz * sizeof(float), filter.buf, 0, NULL, NULL);
        CHECK_ERROR(err);
        weight_buffers_bound[device_num]["generator/encoder_1/conv2d/kernel"] = true;
      }
      mut[device_num].unlock();

      mut[device_num].lock();
      if (!weight_buffers_bound[device_num]["generator/encoder_1/conv2d/bias"]) {
        weight_buffers[device_num]["generator/encoder_1/conv2d/bias"] = clCreateBuffer(context[device_num], CL_MEM_READ_ONLY, bias.sz * sizeof(float), NULL, &err);
        cl_mem &bias_mem = weight_buffers[device_num]["generator/encoder_1/conv2d/bias"];
        CHECK_ERROR(err);
        err = clEnqueueWriteBuffer(queue[device_num], bias_mem, CL_TRUE, 0, bias.sz * sizeof(float), bias.buf, 0, NULL, NULL);
        CHECK_ERROR(err);
        weight_buffers[device_num]["generator/encoder_1/conv2d/bias"] = bias_mem;
        weight_buffers_bound[device_num]["generator/encoder_1/conv2d/bias"] = true;
      }
      mut[device_num].unlock();

      cl_mem &filter_mem = weight_buffers[device_num]["generator/encoder_1/conv2d/kernel"];
      cl_mem &bias_mem = weight_buffers[device_num]["generator/encoder_1/conv2d/bias"];
      CHECK_ERROR(err);
      #ifdef FINISH
      clFinish(queue[device_num]);
      #endif
      #ifdef SHOW_TIME
      END_RE("write filter bias")
      #endif

      size_t R = filter.shape[0], S = filter.shape[1], K = filter.shape[3];
      conv2d_kernel(device_num, intermediate[device_num][0], intermediate[device_num][1], filter_mem, bias_mem, H_, W_, C_, R, S, K);
    }
  }
  // 2 -> 8
  // conv2d
  // mean
  // variance
  // batchnorm
  // leakyrelu
  for (int step = 2; step <= 8; step++) {
    auto scope = "generator/encoder_" + std::to_string(step);
    auto filter = weights[scope + "/conv2d/kernel"];
    auto bias = weights[scope + "/conv2d/bias"];
    auto scale = weights[scope + "/batch_normalization/gamma"];
    auto offset = weights[scope + "/batch_normalization/beta"];

    #ifdef SHOW_TIME
    START_RE
    #endif
    mut[device_num].lock();
    if (!weight_buffers_bound[device_num][scope + "/batch_normalization/gamma"]) {
      weight_buffers[device_num][scope + "/batch_normalization/gamma"] = clCreateBuffer(context[device_num], CL_MEM_READ_ONLY, scale.sz * sizeof(float), NULL, &err);
      CHECK_ERROR(err);
      cl_mem &scale_mem = weight_buffers[device_num][scope + "/batch_normalization/gamma"];
      err = clEnqueueWriteBuffer(queue[device_num], scale_mem, CL_TRUE, 0, scale.sz * sizeof(float), scale.buf, 0, NULL, NULL);
      CHECK_ERROR(err);
      weight_buffers_bound[device_num][scope + "/batch_normalization/gamma"] = true;
    }
    mut[device_num].unlock();

    mut[device_num].lock();
    if (!weight_buffers_bound[device_num][scope + "/batch_normalization/beta"]) {
      weight_buffers[device_num][scope + "/batch_normalization/beta"] = clCreateBuffer(context[device_num], CL_MEM_READ_ONLY, offset.sz * sizeof(float), NULL, &err);
      CHECK_ERROR(err);
      cl_mem &offset_mem = weight_buffers[device_num][scope + "/batch_normalization/beta"];
      err = clEnqueueWriteBuffer(queue[device_num], offset_mem, CL_TRUE, 0, offset.sz * sizeof(float), offset.buf, 0, NULL, NULL);
      CHECK_ERROR(err);
      weight_buffers_bound[device_num][scope + "/batch_normalization/beta"] = true;
    }
    mut[device_num].unlock();
    #ifdef FINISH
    clFinish(queue[device_num]);
    #endif
    #ifdef SHOW_TIME
    END_RE("3 -> 4 write scale offset")
    #endif

    { // leakyrelu (i = step)
      cl_mem &input = intermediate[device_num][step - 1];
      cl_mem &output = B[device_num];
      leakyrelu(device_num, input, output, 0.2f, H_, W_, C_);
    }

    { // conv2d
      cl_mem &input = B[device_num];
      cl_mem &output = A[device_num];
      #ifdef SHOW_TIME
      double st, et;
      START_RE
      #endif
      mut[device_num].lock();
      if (!weight_buffers_bound[device_num][scope + "/conv2d/kernel"]) {
        weight_buffers[device_num][scope + "/conv2d/kernel"] = clCreateBuffer(context[device_num], CL_MEM_READ_ONLY, filter.sz * sizeof(float), NULL, &err);
        CHECK_ERROR(err);
        cl_mem &filter_mem = weight_buffers[device_num][scope + "/conv2d/kernel"];
        err = clEnqueueWriteBuffer(queue[device_num], filter_mem, CL_TRUE, 0, filter.sz * sizeof(float), filter.buf, 0, NULL, NULL);
        CHECK_ERROR(err);
        weight_buffers_bound[device_num][scope + "/conv2d/kernel"] = true;
      }
      mut[device_num].unlock();

      mut[device_num].lock();
      if (!weight_buffers_bound[device_num][scope + "/conv2d/bias"]) {
        auto bias = weights[scope + "/conv2d/bias"];
        weight_buffers[device_num][scope + "/conv2d/bias"] = clCreateBuffer(context[device_num], CL_MEM_READ_ONLY, bias.sz * sizeof(float), NULL, &err);
        cl_mem &bias_mem = weight_buffers[device_num][scope + "/conv2d/bias"];
        CHECK_ERROR(err);
        err = clEnqueueWriteBuffer(queue[device_num], bias_mem, CL_TRUE, 0, bias.sz * sizeof(float), bias.buf, 0, NULL, NULL);
        CHECK_ERROR(err);
        weight_buffers[device_num][scope + "/conv2d/bias"] = bias_mem;
        weight_buffers_bound[device_num][scope + "/conv2d/bias"] = true;
      }
      mut[device_num].unlock();

      cl_mem &filter_mem = weight_buffers[device_num][scope + "/conv2d/kernel"];
      cl_mem &bias_mem = weight_buffers[device_num][scope + "/conv2d/bias"];
      CHECK_ERROR(err);
      #ifdef FINISH
      clFinish(queue[device_num]);
      #endif
      #ifdef SHOW_TIME
      END_RE("write filter bias")
      #endif

      size_t R = filter.shape[0], S = filter.shape[1], K = filter.shape[3];
      conv2d_kernel(device_num, input, output, filter_mem, bias_mem, H_, W_, C_, R, S, K);
    }

    cl_mem mean_mem = clCreateBuffer(context[device_num], CL_MEM_READ_WRITE, C_ * sizeof(float), NULL, &err);
    CHECK_ERROR(err);
    cl_mem variance_mem = clCreateBuffer(context[device_num], CL_MEM_READ_WRITE, C_ * sizeof(float), NULL, &err);
    CHECK_ERROR(err);
    { // mean
      cl_mem &input = A[device_num];
      mean_kernel(device_num, input, mean_mem, H_, W_, C_);
    }

    { // variance
      cl_mem &input = A[device_num];
      variance_kernel(device_num, input, mean_mem, variance_mem, H_, W_, C_);
    }

    { // batchnorm
      cl_mem &input = A[device_num];
      cl_mem &output = intermediate[device_num][step];
      cl_mem &scale_mem = weight_buffers[device_num][scope + "/batch_normalization/gamma"];
      cl_mem &offset_mem = weight_buffers[device_num][scope + "/batch_normalization/beta"];
      batchnorm_kernel(device_num, input, mean_mem, variance_mem, output, offset_mem, scale_mem, H_, W_, C_);
    }

    clReleaseMemObject(mean_mem);
    clReleaseMemObject(variance_mem);
  }

  // decoding
  for (int i = 8; i >= 1; --i) {
    auto scope = "generator/decoder_" + std::to_string(i);
    auto filter = weights[scope + "/conv2d_transpose/kernel"];
    auto bias = weights[scope + "/conv2d_transpose/bias"];
    if (i == 8) {
      // For decoder 8, input is last layer of encoder
      // decoder_layer_input[i] = pppp;

      {
        leakyrelu(device_num, intermediate[device_num][8], A[device_num], 0.0f, H_, W_, C_);
      }
    } else {
      // For other decoder, input is concatenation of previous layer and corresponding encoder layer
      // concat(decoder_layer[i + 1], encoder_layer[i], decoder_layer_input[i]);
      {
        // A -> B
        cl_mem &input = A[device_num];
        cl_mem &input2 = intermediate[device_num][i];
        cl_mem &output = B[device_num];
        concat_kernel(device_num, input, input2, output, H_, W_, C_, C_, C_);
      }
        
      {
        leakyrelu(device_num, B[device_num], A[device_num], 0.0f, H_, W_, C_);
      }
    }

    {
      #ifdef SHOW_TIME
      double st, et;
      START_RE
      #endif
      mut[device_num].lock();
      if (!weight_buffers_bound[device_num][scope + "/conv2d_transpose/kernel"]) {
        weight_buffers[device_num][scope + "/conv2d_transpose/kernel"] = clCreateBuffer(context[device_num], CL_MEM_READ_ONLY, filter.sz * sizeof(float), NULL, &err);
        CHECK_ERROR(err);
        cl_mem &filter_mem = weight_buffers[device_num][scope + "/conv2d_transpose/kernel"];
        err = clEnqueueWriteBuffer(queue[device_num], filter_mem, CL_TRUE, 0, filter.sz * sizeof(float), filter.buf, 0, NULL, NULL);
        CHECK_ERROR(err);
        weight_buffers_bound[device_num][scope + "/conv2d_transpose/kernel"] = true;
      }
      mut[device_num].unlock();

      mut[device_num].lock();
      if (!weight_buffers_bound[device_num][scope + "/conv2d_transpose/bias"]) {
        weight_buffers[device_num][scope + "/conv2d_transpose/bias"] = clCreateBuffer(context[device_num], CL_MEM_READ_ONLY, bias.sz * sizeof(float), NULL, &err);
        CHECK_ERROR(err);
        cl_mem &bias_mem = weight_buffers[device_num][scope + "/conv2d_transpose/bias"];
        err = clEnqueueWriteBuffer(queue[device_num], bias_mem, CL_TRUE, 0, bias.sz * sizeof(float), bias.buf, 0, NULL, NULL);
        CHECK_ERROR(err);
        weight_buffers_bound[device_num][scope + "/conv2d_transpose/bias"] = true;
      }
      mut[device_num].unlock();
      #ifdef FINISH
      clFinish(queue[device_num]);
      #endif
      #ifdef SHOW_TIME
      END_RE("write filter bias")
      #endif

      cl_mem &input = A[device_num];
      cl_mem &output = B[device_num];
      cl_mem &filter_mem = weight_buffers[device_num][scope + "/conv2d_transpose/kernel"];
      cl_mem &bias_mem = weight_buffers[device_num][scope + "/conv2d_transpose/bias"];

      size_t R = filter.shape[0], S = filter.shape[1], K = filter.shape[2];
      conv2d_transposed_kernel(device_num, input, output, filter_mem, bias_mem, H_, W_, C_, R, S, K);
    }

    // Last decoder does not have batchnorm
    if (i == 1) continue;

    #ifdef SHOW_TIME
    START_RE
    #endif
    auto scale = weights[scope + "/batch_normalization/gamma"];
    auto offset = weights[scope + "/batch_normalization/beta"];

    mut[device_num].lock();
    if (!weight_buffers_bound[device_num][scope + "/batch_normalization/gamma"]) {
      weight_buffers[device_num][scope + "/batch_normalization/gamma"] = clCreateBuffer(context[device_num], CL_MEM_READ_ONLY, scale.sz * sizeof(float), NULL, &err);
      CHECK_ERROR(err);
      cl_mem &scale_mem = weight_buffers[device_num][scope + "/batch_normalization/gamma"];
      err = clEnqueueWriteBuffer(queue[device_num], scale_mem, CL_TRUE, 0, scale.sz * sizeof(float), scale.buf, 0, NULL, NULL);
      CHECK_ERROR(err);
      weight_buffers_bound[device_num][scope + "/batch_normalization/gamma"] = true;
    }
    mut[device_num].unlock();

    mut[device_num].lock();
    if (!weight_buffers_bound[device_num][scope + "/batch_normalization/beta"]) {
      weight_buffers[device_num][scope + "/batch_normalization/beta"] = clCreateBuffer(context[device_num], CL_MEM_READ_ONLY, offset.sz * sizeof(float), NULL, &err);
      CHECK_ERROR(err);
      cl_mem &offset_mem = weight_buffers[device_num][scope + "/batch_normalization/beta"];
      err = clEnqueueWriteBuffer(queue[device_num], offset_mem, CL_TRUE, 0, offset.sz * sizeof(float), offset.buf, 0, NULL, NULL);
      CHECK_ERROR(err);
      weight_buffers_bound[device_num][scope + "/batch_normalization/beta"] = true;
    }
    mut[device_num].unlock();
    #ifdef FINISH
    clFinish(queue[device_num]);
    #endif
    #ifdef SHOW_TIME
    END_RE("write scale offset")
    #endif

    cl_mem mean_mem = clCreateBuffer(context[device_num], CL_MEM_READ_ONLY, C_ * sizeof(float), NULL, &err);
    CHECK_ERROR(err);
    cl_mem variance_mem = clCreateBuffer(context[device_num], CL_MEM_READ_ONLY, C_ * sizeof(float), NULL, &err);
    CHECK_ERROR(err);
    { // mean
      cl_mem &input = B[device_num];
      mean_kernel(device_num, input, mean_mem, H_, W_, C_);
    }

    { // variance
      cl_mem &input = B[device_num];
      variance_kernel(device_num, input, mean_mem, variance_mem, H_, W_, C_);
    }

    { // batchnorm
      cl_mem &input = B[device_num];
      cl_mem &output = A[device_num];
      cl_mem &scale_mem = weight_buffers[device_num][scope + "/batch_normalization/gamma"];
      cl_mem &offset_mem = weight_buffers[device_num][scope + "/batch_normalization/beta"];
      batchnorm_kernel(device_num, input, mean_mem, variance_mem, output, offset_mem, scale_mem, H_, W_, C_);
    }
    clReleaseMemObject(mean_mem);
    clReleaseMemObject(variance_mem);
  }

  tanh_kernel(device_num, B[device_num], A[device_num], H_, W_, C_);

  #ifdef SHOW_TIME
  START_RE
  #endif
  encoded.alloc_once({H_, W_, C_});
  err = clEnqueueReadBuffer(queue[device_num], A[device_num], CL_TRUE, 0, encoded.sz * sizeof(float), encoded.buf, 0, NULL, NULL);
  CHECK_ERROR(err);
  #ifdef FINISH
  clFinish(queue[device_num]);
  #endif
  #ifdef SHOW_TIME
  END_RE("read")
  #endif
}

void conv2d_kernel(int device_num, cl_mem &input, cl_mem &output, cl_mem &filter_mem, cl_mem &bias_mem, size_t &H, size_t &W, size_t &C, size_t R, size_t S, size_t K) {
  const size_t stride = 2, pad = 1;
  size_t OH = H / stride, OW = W / stride;

  #ifdef SHOW_TIME
  double st, et;
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
  END_RE("run kernel conv2d")
  #endif

  H = OH;
  W = OW;
  C = K;
}

void leakyrelu(int device_num, cl_mem &input, cl_mem &output, float alpha, size_t H, size_t W, size_t C) {
  #ifdef SHOW_TIME
  double st, et;
  START_RE
  #endif
  err = clSetKernelArg(kernel[device_num][K_LEAKYRELU], 0, sizeof(cl_mem), &input);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel[device_num][K_LEAKYRELU], 1, sizeof(cl_mem), &output);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel[device_num][K_LEAKYRELU], 2, sizeof(float), &alpha);
  CHECK_ERROR(err);

  size_t gws[1] = {H * W * C}, lws[1] = {256};
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

void tanh_kernel(int device_num, cl_mem &input, cl_mem &output, size_t H, size_t W, size_t C) {
  #ifdef SHOW_TIME
  double st, et;
  START_RE
  #endif
  err = clSetKernelArg(kernel[device_num][K_TANH], 0, sizeof(cl_mem), &input);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel[device_num][K_TANH], 1, sizeof(cl_mem), &output);
  CHECK_ERROR(err);

  size_t gws[1] = {H * W * C}, lws[1] = {256};
  for (int i = 0; i < 1; ++i) {
    gws[i] = (gws[i] + lws[i] - 1) / lws[i] * lws[i];
  }

  err = clEnqueueNDRangeKernel(queue[device_num], kernel[device_num][K_TANH], 1, NULL, gws, lws, 0, NULL, NULL);
  CHECK_ERROR(err);
  #ifdef FINISH
  clFinish(queue[device_num]);
  #endif
  #ifdef SHOW_TIME
  END_RE("run kernel tanh")
  #endif
}

void mean_kernel(int device_num, cl_mem &input, cl_mem &mean_mem, size_t &H, size_t &W, size_t &C) {
  #ifdef SHOW_TIME
  double st, et;
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
  double st, et;
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

void batchnorm_kernel(int device_num, cl_mem &input, cl_mem &mean_mem, cl_mem &variance_mem, cl_mem &output, cl_mem &offset_mem, cl_mem &scale_mem, size_t H, size_t W, size_t C) {
  #ifdef SHOW_TIME
  double st, et;
  START_RE
  #endif
  size_t K_p = 0;
  LOG2S(C, K_p);
  const size_t K_mask = ((1 << K_p) - 1);

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
  END_RE("run kernel batchnorm")
  #endif
}

void conv2d_transposed_kernel(int device_num, cl_mem &input, cl_mem &output, cl_mem &filter_mem, cl_mem &bias_mem, size_t &H, size_t &W, size_t &C, size_t R, size_t S, size_t K) {
  const size_t stride = 2, pad = 1;
  size_t OH = H * stride, OW = W * stride;

  #ifdef SHOW_TIME
  double st, et;
  START_RE
  #endif
  int OWK = OW * K;

  err = clSetKernelArg(kernel[device_num][K_CONV2D_TRANSPOSED], 0, sizeof(cl_mem), &input);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel[device_num][K_CONV2D_TRANSPOSED], 1, sizeof(cl_mem), &filter_mem);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel[device_num][K_CONV2D_TRANSPOSED], 2, sizeof(cl_mem), &bias_mem);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel[device_num][K_CONV2D_TRANSPOSED], 3, sizeof(cl_mem), &output);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel[device_num][K_CONV2D_TRANSPOSED], 4, sizeof(int), &H);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel[device_num][K_CONV2D_TRANSPOSED], 5, sizeof(int), &W);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel[device_num][K_CONV2D_TRANSPOSED], 6, sizeof(int), &C);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel[device_num][K_CONV2D_TRANSPOSED], 7, sizeof(int), &R);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel[device_num][K_CONV2D_TRANSPOSED], 8, sizeof(int), &S);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel[device_num][K_CONV2D_TRANSPOSED], 9, sizeof(int), &K);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel[device_num][K_CONV2D_TRANSPOSED], 10, sizeof(int), &OH);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel[device_num][K_CONV2D_TRANSPOSED], 11, sizeof(int), &OW);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel[device_num][K_CONV2D_TRANSPOSED], 12, sizeof(int), &stride);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel[device_num][K_CONV2D_TRANSPOSED], 13, sizeof(int), &pad);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel[device_num][K_CONV2D_TRANSPOSED], 14, sizeof(int), &OWK);
  CHECK_ERROR(err);

  size_t gws[1] = {OH * OW * K}, lws[1] = {128};
  for (int i = 0; i < 1; ++i) {
    gws[i] = (gws[i] + lws[i] - 1) / lws[i] * lws[i];
  }

  // Run kernel
  err = clEnqueueNDRangeKernel(queue[device_num], kernel[device_num][K_CONV2D_TRANSPOSED], 1, NULL, gws, lws, 0, NULL, NULL);
  CHECK_ERROR(err);
  #ifdef FINISH
  clFinish(queue[device_num]);
  #endif
  #ifdef SHOW_TIME
  END_RE("conv2d_transposed")
  #endif

  H = OH;
  W = OW;
  C = K;
}

void concat_kernel(int device_num, cl_mem &input, cl_mem &input2, cl_mem &output, int H, int W, int C0, int C1, size_t &C) {
  #ifdef SHOW_TIME
  double st, et;
  START_RE
  #endif

  err = clSetKernelArg(kernel[device_num][K_CONCAT], 0, sizeof(cl_mem), &input);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel[device_num][K_CONCAT], 1, sizeof(cl_mem), &input2);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel[device_num][K_CONCAT], 2, sizeof(cl_mem), &output);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel[device_num][K_CONCAT], 3, sizeof(int), &H);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel[device_num][K_CONCAT], 4, sizeof(int), &W);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel[device_num][K_CONCAT], 5, sizeof(int), &C0);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel[device_num][K_CONCAT], 6, sizeof(int), &C1);
  CHECK_ERROR(err);

  size_t gws[1] = {H * W * (C0 + C1)}, lws[1] = {128};
  for (int i = 0; i < 1; ++i) {
    gws[i] = (gws[i] + lws[i] - 1) / lws[i] * lws[i];
  }

  err = clEnqueueNDRangeKernel(queue[device_num], kernel[device_num][K_CONCAT], 1, NULL, gws, lws, 0, NULL, NULL);
  CHECK_ERROR(err);
  #ifdef FINISH
  clFinish(queue[device_num]);
  #endif
  #ifdef SHOW_TIME
  END_RE("run kernel concat")
  #endif

  C = C0 + C1;
}
