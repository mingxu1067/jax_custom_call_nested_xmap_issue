#include "fake_custom_call.h"
#include "pybind11_kernel_helpers.h"

using namespace fake_custom_call_ext;

namespace {
pybind11::dict Registrations() {
  pybind11::dict dict;
  dict["fake_custom_call_fwd"] = EncapsulateFunction(fake_custom_call_fwd);
  dict["fake_custom_call_bwd"] = EncapsulateFunction(fake_custom_call_bwd);
  return dict;
}

PYBIND11_MODULE(gpu_ops, m) {
  m.def("registrations", &Registrations);
  m.def("build_fake_custom_call_descriptor",
        [](std::int64_t n, std::int64_t hidden, float epsilon) { return PackDescriptor(FakeDescriptor{n, hidden, epsilon}); });
}
}  // namespace
