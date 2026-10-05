// Native-torch migration gap: upstream's torch/csrc/api/include/torch/version.h
// is this thin redirect to the generated torch/headeronly/version.h. Staged at
// the torch/ root so <torch/version.h> resolves against NATIVE_TORCH_ROOT.
#pragma once

#include <torch/headeronly/version.h>
