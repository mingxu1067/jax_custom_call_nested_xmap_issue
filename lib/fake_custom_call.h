#ifndef _LIB_KERNELS_H_
#define _LIB_KERNELS_H_

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace fake_custom_call_ext {
struct FakeDescriptor {
  std::int64_t n;
  std::int64_t hiddien;
  float epsilon;
};

void fake_custom_call_fwd(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len);

void fake_custom_call_bwd(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len);
}  // namespace my_jax_ext

#endif
