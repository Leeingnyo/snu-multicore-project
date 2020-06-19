#include <cstdio>
#include <cstdlib>

#ifdef USE_MPI
#include <mpi.h>
#endif

#include "util.h"
#include "pix2pix.h"

int main(int argc, char **argv) {
#ifdef USE_MPI
  MPI_Init(&argc, &argv);
#endif

  double st, et;

  if (argc != 4) {
    if (get_rank() == 0) {
      printf("Usage: %s [network.bin] [input.rgb] [output.rgb]\n", argv[0]);
    }
    check_error(true);
  }

  float *weight_buf = NULL;
  uint8_t *input_buf = NULL;
  uint8_t *output_buf = NULL;
  size_t input_sz = 0, num_image = 0;

  if (get_rank() == 0) {
    printf("Reading inputs..."); fflush(stdout);
    st = get_time();
    weight_buf = (float*)read_file(argv[1], NULL);
    input_buf = (uint8_t*)read_file(argv[2], &input_sz);
    num_image = input_sz / (256 * 256 * 3 * sizeof(uint8_t));
    output_buf = (uint8_t*)malloc(input_sz);
    et = get_time();
    printf(" done! (%f s)\n", et - st);
    if (!weight_buf) {
      printf("Failed to read %s.\n", argv[1]);
    }
    if (!input_buf) {
      printf("Failed to read %s.\n", argv[2]);
    }
    if (!output_buf) {
      printf("Failed to allocate output buffer.\n");
    }
  }
  check_error(!weight_buf || !input_buf || !output_buf);

  if (get_rank() == 0) {
    printf("Initializing pix2pix..."); fflush(stdout);
  }

#ifdef USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif
  st = get_time();

  pix2pix_init();

#ifdef USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif
  et = get_time();

  if (get_rank() == 0) {
    printf(" done! (%f s)\n", et - st);
    printf("Calculating pix2pix..."); fflush(stdout);
  }

#ifdef USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif
  st = get_time();

  pix2pix(input_buf, weight_buf, output_buf, num_image);

#ifdef USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif
  et = get_time();

  if (get_rank() == 0) {
    printf(" done! (%f s)\n", et - st);
  }

  double elapsed_time = et - st;

  bool write_success = false;
  if (get_rank() == 0) {
    printf("Writing result..."); fflush(stdout);
    st = get_time();
    write_success = write_file(argv[3], input_sz, output_buf);
    et = get_time();
    printf(" done! (%f s)\n", et - st);
    if (!write_success) {
      printf("Failed to write %s.\n", argv[3]);
    }
  }
  check_error(!write_success);

  if (get_rank() == 0) {
    printf("Elapsed time : %f sec\n", elapsed_time);
    printf("Throughput : %f img / sec\n", num_image / elapsed_time);
  }

#ifdef USE_MPI
  MPI_Finalize();
#endif

  return 0;
}
