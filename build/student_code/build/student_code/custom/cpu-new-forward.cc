#include "cpu-new-forward.h"

void conv_forward_cpu(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    The code in 16 is for a single image.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct, not fast (this is the CPU implementation.)

    Function paramters:
    y - output
    x - input
    k - kernel
    B - batch_size (number of images in x)
    M - number of output feature maps
    C - number of input feature maps
    H - input height dimension
    W - input width dimension
    K - kernel height and width (K x K)
    */

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

  // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

  // Insert your CPU convolution kernel code here

  for (int b = 0; b < B; b++) {                   // for each image in the batch 
    for (int m = 0; m < M; m++) {                 // for each output feature maps
      for (int h = 0; h < H_out; h++) {           // for each output element
        for (int w = 0; w < W_out; w++) {
          y4d(b, m, h, w) = 0;

          for (int c = 0; c < C; c++) {         // sum over all input feature maps/channels
            for (int p = 0; p < K; p++) {       // KxK filter
              for (int q = 0; q < K; q++) {
                y4d(b, m, h, w) += x4d(b, c, h + p, w + q) * k4d(m, c, p, q);
              }
            }
          }
        }
      }
    }
  }

#undef y4d
#undef x4d
#undef k4d

}