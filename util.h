#pragma once

#include <vector>
#include <cstddef>

// Read file and return buffer with file contents. Set size if given.
void* read_file(const char *fn, size_t *sz);

// Write buffer to file
bool write_file(const char *fn, size_t sz, void *buf);

int get_rank();

// Check error on rank 0. Should be called by every process.
void check_error (int err);

double get_time();
