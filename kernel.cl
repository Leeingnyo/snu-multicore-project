


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
  int pad // 패드
) {
  // 아웃이 잘 해서 잘 한다
  int i = get_global_id(0); // OH
  int j = get_global_id(1); // OW
  int k = get_global_id(2); // K

  int ti = get_local_id(0); // OH
  int tj = get_local_id(1); // OW
  int tk = get_local_id(2); // K

  int gi = get_group_id(0); // OH
  int gj = get_group_id(1); // OW
  int gk = get_group_id(2); // K

  int idx = i * OW * K + j * K + k;
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

__kernel void im2col() {
}
