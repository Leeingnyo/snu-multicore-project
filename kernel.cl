__kernel void postprocess_one_image(
  __global float *input, __global float *out, int H, int W, int C
) {
  int i = get_global_id(0); // H
  int j = get_global_id(1); // W
  int k = get_global_id(2); // C

  int ti = get_local_id(0); // H
  int tj = get_local_id(1); // W
  int tk = get_local_id(2); // C

  int gi = get_group_id(0); // H
  int gj = get_group_id(1); // W
  int gk = get_group_id(2); // C

  int idx = i * H * W + j * W + k;
  out[idx] = (input[idx] + 1) / 2 * 255;
}

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

__kernel void conv2d_leakyrelu(
  __global float *input, __global float *filter, __global float *bias,
  __global float *output, 
  int H, int W, int C,
  int R, int S, int K, // filter 크기 // 10
  int OH, int OW, // 아웃풋 크기
  int stride, // 스트라이드
  int pad, // 패드 // 14
  int K_p, int OW_p,
  int K_mask, int OW_mask,
  float alpha
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
  int i = idx >> (OW_p + K_p);
  int j = (idx >> K_p) & OW_mask;
  int k = idx & K_mask;

  // conv2d
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

  // relu
  output[idx] = x >= 0 ? x : alpha * x;
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

inline float atomicadd(volatile __global float* address, const float value){
  float old = value;
  while ((old = atomic_xchg(address, atomic_xchg(address, 0.0f)+old))!=0.0f);
  return old;
}

__kernel void conv2d_batchnorm_leakyrelu(
  __global float *input, __global float *filter, __global float *bias,
  __global float *output, 
  int H, int W, int C,
  int R, int S, int K, // filter 크기 // 10
  int OH, int OW, // 아웃풋 크기
  int stride, // 스트라이드
  int pad, // 패드 // 14
  int K_p, int OW_p,
  int K_mask, int OW_mask, // 18
  float alpha,
  __global float *scale,
  __global float *offset,
  __local volatile float *local_buffer,
  __global float *sum,
  __global float *sqsum
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
  int i = idx >> (OW_p + K_p);
  int j = (idx >> K_p) & OW_mask;
  int k = idx & K_mask;

  // conv2d
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

  // batchnorm
  /*
  float HW = H * W;
  sum[k] = 0.0f;
  sqsum[k] = 0.0f;
  barrier(CLK_GLOBAL_MEM_FENCE);
  if (idx < K) {
    for (int i = 0; i < OH; ++i) {
      for (int i = 0; i < OH; ++j) {
        sum[idx] += output[i * OW * K + j * K + idx];
      }
    }
    float mean = sum[idx] / HW;
    for (int i = 0; i < OH; ++i) {
      for (int i = 0; i < OH; ++j) {
        float x = output[i * OW * K + j * K + idx];
        sqsum[idx] += (x - mean) * (x - mean);
      }
    }
  }
  barrier(CLK_GLOBAL_MEM_FENCE);
  float mean = sum[k] / HW;
  float variance = sqsum[k] / HW;

  float epsilon = 1e-5;
  float xx = offset[k] + (x - mean) * scale[k] / sqrt(variance + epsilon);
  */
  float xx = x;

  // relu
  output[idx] = xx >= 0 ? xx : alpha * xx;
}


  /*
  int lid = get_local_id(0);
  int group_size = get_local_size(0);
  float HW = H * W;

  float res = x;
  local_buffer[lid] = res;
  barrier(CLK_LOCAL_MEM_FENCE);
  int lop = group_size/2;
  for (; lop > 1; lop >>= 1) {
    if (lid < 1) {
      local_buffer[lid] = res = res + local_buffer[lid + 1];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lid == 0) {
    atomicadd(sum, res);
  }
  barrier(CLK_GLOBAL_MEM_FENCE);
  float mean = *sum / HW;
  float delta = (x - mean);

  res = delta * delta;
  local_buffer[lid] = res;
  barrier(CLK_LOCAL_MEM_FENCE);
  lop = group_size/2;
  for (; lop > 1; lop >>= 1) {
    if (lid < 1) {
      local_buffer[lid] = res = res + local_buffer[lid + 1];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lid == 0) {
    atomicadd(sqsum, res);
  }
  barrier(CLK_GLOBAL_MEM_FENCE);
  float variance = *sqsum / HW;
  */
