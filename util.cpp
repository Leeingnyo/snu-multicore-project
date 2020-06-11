#include "util.h"

#include <cstdio>
#include <cstdlib>
#include <sys/time.h>

#ifdef USE_MPI
#include <mpi.h>
#endif

void* read_file(const char *fn, size_t *sz) {
  size_t sz_;
  FILE *f = fopen(fn, "rb");
  if (f == NULL) return NULL;
  fseek(f, 0, SEEK_END);
  sz_ = ftell(f);
  rewind(f);
  void *buf = malloc(sz_);
  size_t ret = fread(buf, 1, sz_, f);
  fclose(f);
  if (sz_ != ret) return NULL;
  if (sz != NULL) *sz = sz_;
  return buf;
}

bool write_file(const char *fn, size_t sz, void *buf) {
  FILE *f = fopen(fn, "wb");
  if (f == NULL) return false;
  size_t ret = fwrite(buf, 1, sz, f);
  fclose(f);
  if (sz != ret) return false;
  return true;
}

int get_rank() {
  int rank = 0;
#ifdef USE_MPI
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
  return rank;
}

void check_error (int err) {
#ifdef USE_MPI
  MPI_Bcast(&err, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif
  if (err) {
#ifdef USE_MPI
    MPI_Finalize();
#endif
    exit(0);
  }
}

double get_time() {
  struct timeval tv;
  gettimeofday(&tv, 0);
  return tv.tv_sec + tv.tv_usec * 1e-6;
}
