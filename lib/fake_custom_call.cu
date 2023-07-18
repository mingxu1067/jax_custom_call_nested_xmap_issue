#include <iostream>
#include "fake_custom_call.h"


namespace fake_custom_call_ext{

void fake_custom_call_fwd(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len) {
    std::cout << "Fake Custom Call FWD..." << std::endl;
}

void fake_custom_call_bwd(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len) {
    std::cout << "Fake Custom Call BWD..." << std::endl;
}
}  // namespace fake_custom_call_ext
