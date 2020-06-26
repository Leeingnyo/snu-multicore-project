__kernel void conv2d(
  __global float *input, __global float *filter, __global float *bias,
  __global float *output, 
  int H, int W, int C,
  int R, int S, int K, // filter 크기
  int OH, int OW, // 아웃풋 크기
  int stride, // 스트라이드
  int pad, // 패드
  int K_p, int OW_p,
  int K_mask, int OW_mask
) {
  // 아웃이 잘 해서 잘 한다
  /*
  int i = get_global_id(0); // OH
  int j = get_global_id(1); // OW
  int k = get_global_id(2); // K

  int ti = get_local_id(0); // OH
  int tj = get_local_id(1); // OW
  int tk = get_local_id(2); // K

  int gi = get_group_id(0); // OH
  int gj = get_group_id(1); // OW
  int gk = get_group_id(2); // K
  */

  // int idx = i * OW * K + j * K + k;
  int idx = get_global_id(0);
  int i = idx >> (OW_p + K_p);
  int j = (idx >> K_p) & OW_mask;
  int k = idx & K_mask;
  float x = bias[k];

  for (int r = 0; r < R; ++r) {
    for (int s = 0; s < S; ++s) {
      int ih = i * stride - pad + r;
      int iw = j * stride - pad + s;
      if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;
      for (int c = 0; c < C; ++c) {
        float ii = input[ih * W * C + iw * C + c];
        float ff = filter[r * S * C * K + s * C * K + c * K + k];
        x += ii * ff;
      }
    }
  }

  output[idx] = x;
}

__kernel void conv2d_transposed(
  __global float *input, __global float *filter, __global float *bias,
  __global float *output, 
  int H, int W, int C,
  int R, int S, int K, // filter 크기
  int OH, int OW, // 아웃풋 크기
  int stride, // 스트라이드
  int pad, // 패드
  int OWK
) {
  /*
  int i = get_global_id(0); // OH
  int j = get_global_id(1); // OW
  int k = get_global_id(2); // K

  int ti = get_local_id(0); // OH
  int tj = get_local_id(1); // OW
  int tk = get_local_id(2); // K

  int gi = get_group_id(0); // OH
  int gj = get_group_id(1); // OW
  int gk = get_group_id(2); // K
  */

  // int idx = i * OW * K + j * K + k;
  int idx = get_global_id(0);
  int i = idx / OWK;
  int j = idx / K % OW;
  int k = idx % K;
  float x = 0.0f;

  for (int r = 0; r < R; ++r) {
    for (int s = 0; s < S; ++s) {
      if ((i - r + pad) % stride != 0 || (j - s + pad) % stride != 0) continue;
      int ih = (i - r + pad) / stride;
      int iw = (j - s + pad) / stride;
      if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;
      for (int c = 0; c < C; ++c) {
        float ii = input[ih * W * C + iw * C + c];
        float ff = filter[r * S * K * C + s * K * C + k * C + c];
        x += ii * ff;
      }
    }
  }

  output[idx] = x + bias[k];
}

__kernel void mean(
  __global float *input,
  __global float *output,
  int H,
  int W,
  int K
) {
  int k = get_global_id(0); // K

  float sum = 0.0f;
  for (int h = 0; h < H; ++h) {
    for (int w = 0; w < W; ++w) {
      sum += input[h * W * K + w * K + k];
    }
  }

  output[k] = sum / (H * W);
}

__kernel void variance(
  __global float *input,
  __global float *mean,
  __global float *output,
  int H,
  int W,
  int K
) {
  int k = get_global_id(0); // K

  float sum = 0.0f;
  float mm = mean[k];
  for (int h = 0; h < H; ++h) {
    for (int w = 0; w < W; ++w) {
      float ii = input[h * W * K + w * K + k];
      sum += (ii - mm) * (ii - mm);
    }
  }

  output[k] = sum / (H * W);
}

__kernel void batchnorm(
  __global float *input,
  __global float *mean,
  __global float *variance,
  __global float *output,
  __global float *offset,
  __global float *scale,
  int K_mask
) {
  int idx = get_global_id(0);
  int k = idx & K_mask;

  float epsilon = 1e-5;
  output[idx] = offset[k] + (input[idx] - mean[k]) * scale[k] / sqrt(variance[k] + epsilon);
}

__kernel void leakyrelu(
  __global float *input,
  __global float *output,
  float alpha
) {
  int idx = get_global_id(0);
  float x = input[idx];
  output[idx] = x >= 0 ? x : alpha * x;
}

__kernel void concat(
  __global float *input0,
  __global float *input1,
  __global float *output,
  int H,
  int W,
  int C0,
  int C1
) {
  int idx = get_global_id(0);
  int h = idx / W / (C0 + C1);
  int w = idx / (C0 + C1) % W;
  int c = idx % (C0 + C1);
  if (c - C0 < 0) {
    output[idx] = input0[h * W * C0 + w * C0 + c];
  } else {
    output[idx] = input1[h * W * C1 + w * C1 + c - C0];
  }
}

__kernel void elem_tanh(
  __global float *input,
  __global float *output
) {
  int idx = get_global_id(0);
  float x = input[idx];
  output[idx] = tanh(x);
}




// (R, S, C, K) => (K, C * R * S)
__kernel void transform_filter(
  __global float *filters,
  __global float *output,
  int R,
  int S,
  int C,
  int K
) {
  int k = get_global_id(0);
  int c = get_global_id(1);
  int rs = get_global_id(2);
  int r = rs / S;
  int s = rs % S;

  output[k * R * S * C + c * R * S + r * S + s] = filters[r * S * C * K + s * C * K + c * K + k];
}

// (H, W, C) => (C * R * S, OH * OW)
__kernel void transform_in(
  __global float *input,
  __global float *output,
  int H, int W, int C,
  int R, int S, int K,
  int OH,
  int OW
) {
  int c = get_global_id(0);
  int rs = get_global_id(1);
  int r = rs / S;
  int s = rs % S;
  int o = get_global_id(2);
  int oh = o / OW;
  int ow = o % OW;

  int ih = oh * 2 - 1 + r;
  int iw = ow * 2 - 1 + s;
  if (ih < 0 || ih >= H || iw < 0 || iw >= W) return;
  for (int c = 0; c < C; c++) {
    output[c * R * S * OH * OW + r * S * OH * OW + s * OH * OW + oh * OW + ow] = input[ih * W * C + iw * C + c];
  }
}

// (C, H, W) -> (H, W, C)
__kernel void transform_out(
  __global float *input,
  __global float *output,
  int H,
  int W,
  int C
) {
  int c = get_global_id(0);
  int h = get_global_id(1);
  int w = get_global_id(2);
  output[h * W * C + w * C + c] = input[c * H * W + h * W + w];
}

#define TILE_SIZE 16
// a little fast sgemm kernel by inyong
__kernel void sgemm(__global float *A, __global float *B, __global float *C, int M, int N, int K) {
  int i = get_global_id(0); // row index of C
  int j = get_global_id(1); // column index of C

  int ti = get_local_id(0); // row index of tile
  int tj = get_local_id(1); // column index of tile

  int gi = get_group_id(0); // row index of group
  int gj = get_group_id(1); // column index of group

  if (i >= M || j >= N) return; // boundary check

  /*
  __local float Atile[TILE_SIZE][TILE_SIZE];
  __local float Btile[TILE_SIZE][TILE_SIZE];
  */

  float acc = 0.0f;
  /*
  int nt = K / TILE_SIZE + (K % TILE_SIZE != 0);
  int ii = (gi * TILE_SIZE + ti);
  int jj = (gj * TILE_SIZE + tj);
  for (int t = 0; t < nt; t++) {
    int tt = t * TILE_SIZE;
    Atile[ti][tj] = A[ii * K + (tt + tj)];
    Btile[ti][tj] = B[(tt + ti) * N + jj];

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int k = 0; k < TILE_SIZE; k++) {
      acc += Atile[ti][k] * Btile[k][tj];
    }

    barrier(CLK_LOCAL_MEM_FENCE);
  }
  */
  for (int k = 0; k < K; k++) {
    C[i * N + j] += A[i * K + k] * B[k * N + j];
  }

  C[i * N + j] = acc;
}
