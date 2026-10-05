#include <torch/library.h>

// SonicBoom ADAPT: dispatcher fallthrough for keys v0 does not implement.
//
// v0 is an inference runtime with autograd deferred (CLAUDE.md §4.1.5). Three
// families of dispatch keys would otherwise have no kernel and make the
// dispatcher report "Could not run 'aten::...' from the '...' backend":
//
// 1. BackendSelect + ADInplaceOrView — c10::default_included_set
//    (c10/core/DispatchKeySet.h) places both in every thread's default TLS, so
//    they are in the dispatch key set of every op. Upstream registers a
//    fallthrough for each (BackendSelectFallbackKernel.cpp and
//    VariableFallbackKernel.cpp). Without them, every factory op fails before
//    reaching its backend kernel.
//
// 2. Autograd* — TensorImpl::TensorImpl (c10/core/TensorImpl.cpp) adds
//    getAutogradRelatedKeySetFromBackend(k) (e.g. AutogradCPU) to every
//    non-inference tensor's key set. Upstream resolves these via the
//    VariableType kernels (autograd). v0 has no VariableType, so it registers
//    the same fallthrough that upstream's own no-autograd builds use
//    (VariableFallbackKernel.cpp under C10_MOBILE: makeFallthrough).
//
// Both upstream source files are excluded from the v0 build (CMakeLists.txt):
// BackendSelectFallbackKernel.cpp is a trivial one-liner, and
// VariableFallbackKernel.cpp's non-mobile autograd_fallback references
// VariableHooksInterface (autograd machinery). v0 folds the no-autograd subset
// of both into this one stub.
//
// makeFallthrough() makes the dispatcher skip the key (the DispatchKeyExtractor
// clears the key's nonFallthroughKeys_ bit and removes it from the dispatch key
// set before lookup), so dispatch lands on the real backend key (CPU), whose
// entry is resolved from the composite registration (CompositeExplicitAutograd
// etc.).

TORCH_LIBRARY_IMPL(_, BackendSelect, m) {
  m.fallback(torch::CppFunction::makeFallthrough());
}

TORCH_LIBRARY_IMPL(_, ADInplaceOrView, m) {
  m.fallback(torch::CppFunction::makeFallthrough());
}

// Autograd* fallthroughs — mirror upstream VariableFallbackKernel.cpp's
// no-autograd (C10_MOBILE) key list.
TORCH_LIBRARY_IMPL(_, AutogradOther, m) {
  m.fallback(torch::CppFunction::makeFallthrough());
}
TORCH_LIBRARY_IMPL(_, AutogradCPU, m) {
  m.fallback(torch::CppFunction::makeFallthrough());
}
TORCH_LIBRARY_IMPL(_, AutogradXPU, m) {
  m.fallback(torch::CppFunction::makeFallthrough());
}
TORCH_LIBRARY_IMPL(_, AutogradCUDA, m) {
  m.fallback(torch::CppFunction::makeFallthrough());
}
TORCH_LIBRARY_IMPL(_, AutogradMTIA, m) {
  m.fallback(torch::CppFunction::makeFallthrough());
}
TORCH_LIBRARY_IMPL(_, AutogradMAIA, m) {
  m.fallback(torch::CppFunction::makeFallthrough());
}
TORCH_LIBRARY_IMPL(_, AutogradXLA, m) {
  m.fallback(torch::CppFunction::makeFallthrough());
}
TORCH_LIBRARY_IMPL(_, AutogradLazy, m) {
  m.fallback(torch::CppFunction::makeFallthrough());
}
TORCH_LIBRARY_IMPL(_, AutogradMPS, m) {
  m.fallback(torch::CppFunction::makeFallthrough());
}
TORCH_LIBRARY_IMPL(_, AutogradMeta, m) {
  m.fallback(torch::CppFunction::makeFallthrough());
}
TORCH_LIBRARY_IMPL(_, AutogradHPU, m) {
  m.fallback(torch::CppFunction::makeFallthrough());
}
TORCH_LIBRARY_IMPL(_, AutogradPrivateUse1, m) {
  m.fallback(torch::CppFunction::makeFallthrough());
}
