// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

// MOSS-Image VAE decoder, a direct transcription of diffusers'
// AutoencoderKLFlux2.decode: post_quant_conv -> decoder (conv_in, mid block
// with one attention, four up blocks of three resnets, GroupNorm/SiLU/conv_out).
// The convolution, GroupNorm, SiLU, nearest upsample, GEMM and softmax kernels
// are native-torch CUDA kernels called through the Layer 1 adapter
// (layer1/adapter/aten_vae.*). Weights are f32 and stay on the device.
//
// The reference applies the attention GroupNorm to [B,C,HW] and then moves to
// [B,HW,C]; here that transpose is explicit, and the SDPA scale
// 1/sqrt(head_dim) is folded into the q projection's weight and bias at load
// time (head_dim = C, one head), so the score GEMM is the reference's
// (q @ k^T) * scale to within f32 rounding.

#include <sonicboom/model/moss_vae.h>

#include "../../layer1/adapter/aten_vae.h"
#include "../safetensors/json.h"

#include <sonicboom/safetensors/reader.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>
#include <utility>

#if defined(SONICBOOM_USE_CUDA)
#include <cuda_runtime.h>
#endif

namespace sonicboom::moss {

namespace {

std::expected<std::string, std::string> read_text(
    const std::filesystem::path& p) {
  std::ifstream in(p, std::ios::binary);
  if (!in) return std::unexpected("cannot open " + p.string());
  std::ostringstream ss;
  ss << in.rdbuf();
  if (!in.good() && !in.eof()) return std::unexpected("cannot read " + p.string());
  return ss.str();
}

} // namespace

struct VaeDecoder::Impl {
  VaeConfig cfg;
  safetensors::Reader reader;
  int64_t bytes = 0;

  struct D {
    float* p = nullptr;
    int64_t count = 0;
    int64_t c = 0, h = 0, w = 0;  // set for activations only
  };

  struct ResNet {
    D n1w, n1b, c1w, c1b, n2w, n2b, c2w, c2b;
    bool shortcut = false;
    D sw, sb;
    int64_t cin = 0, cout = 0;
  };

  struct Attn {
    D gw, gb, qw, qb, kw, kb, vw, vb, ow, ob;
    int64_t c = 0;
    float scale = 1.0f;
  };

  struct UpBlock {
    std::vector<ResNet> resnets;
    bool upsample = false;
    D uw, ub;
    int64_t cin = 0, cout = 0;
  };

  VaeBatchNorm bn;
  D postq_w, postq_b, conv_in_w, conv_in_b;
  ResNet mid0, mid1;
  Attn attn;
  std::vector<UpBlock> up;
  D out_norm_w, out_norm_b, conv_out_w, conv_out_b;

  explicit Impl(safetensors::Reader&& r) : reader(std::move(r)) {}

  ~Impl() { free_all(); }

  void free_all() {
    for (float* p : all_) cudaFree(p);
    all_.clear();
  }

  std::vector<float*> all_;

#if defined(SONICBOOM_USE_CUDA)

  std::expected<D, std::string> load(const std::string& name,
                                     std::vector<int64_t> shape,
                                     float scale = 1.0f) {
    const safetensors::TensorInfo* info = reader.tensor(name);
    if (info == nullptr) return std::unexpected("missing tensor " + name);
    if (info->dtype != safetensors::DType::F32)
      return std::unexpected(name + ": expected f32");
    if (info->shape.size() != shape.size())
      return std::unexpected(name + ": rank mismatch");
    for (size_t i = 0; i < shape.size(); ++i) {
      if (static_cast<int64_t>(info->shape[i]) != shape[i])
        return std::unexpected(name + ": shape mismatch");
    }
    auto span = reader.tensor_data(name);
    if (!span) return std::unexpected("cannot read " + name);
    const int64_t count = info->n_elements;
    D d;
    d.count = count;
    d.c = shape[0];
    d.h = shape.size() > 2 ? shape[2] : 1;
    d.w = shape.size() > 3 ? shape[3] : 1;
    if (cudaMalloc(reinterpret_cast<void**>(&d.p),
                   static_cast<size_t>(count) * sizeof(float)) != cudaSuccess) {
      return std::unexpected("cudaMalloc failed for " + name);
    }
    all_.push_back(d.p);
    if (scale == 1.0f) {
      if (cudaMemcpy(d.p, span->data(), static_cast<size_t>(count) * sizeof(float),
                     cudaMemcpyHostToDevice) != cudaSuccess) {
        return std::unexpected("cudaMemcpy failed for " + name);
      }
    } else {
      const float* src = reinterpret_cast<const float*>(span->data());
      std::vector<float> scaled(static_cast<size_t>(count));
      for (int64_t i = 0; i < count; ++i) scaled[static_cast<size_t>(i)] = src[i] * scale;
      if (cudaMemcpy(d.p, scaled.data(),
                     static_cast<size_t>(count) * sizeof(float),
                     cudaMemcpyHostToDevice) != cudaSuccess) {
        return std::unexpected("cudaMemcpy failed for " + name);
      }
    }
    bytes += count * 4;
    return d;
  }

  std::expected<ResNet, std::string> load_resnet(const std::string& prefix,
                                                 int64_t cin, int64_t cout) {
    ResNet r;
    r.cin = cin;
    r.cout = cout;
    auto get = [&](D& dst, const std::string& name,
                   std::vector<int64_t> shape) -> std::expected<void, std::string> {
      auto d = load(name, std::move(shape));
      if (!d) return std::unexpected(d.error());
      dst = *d;
      return {};
    };
    const std::string p = prefix + ".";
    if (auto e = get(r.n1w, p + "norm1.weight", {cin}); !e) return std::unexpected(e.error());
    if (auto e = get(r.n1b, p + "norm1.bias", {cin}); !e) return std::unexpected(e.error());
    if (auto e = get(r.c1w, p + "conv1.weight", {cout, cin, 3, 3}); !e) return std::unexpected(e.error());
    if (auto e = get(r.c1b, p + "conv1.bias", {cout}); !e) return std::unexpected(e.error());
    if (auto e = get(r.n2w, p + "norm2.weight", {cout}); !e) return std::unexpected(e.error());
    if (auto e = get(r.n2b, p + "norm2.bias", {cout}); !e) return std::unexpected(e.error());
    if (auto e = get(r.c2w, p + "conv2.weight", {cout, cout, 3, 3}); !e) return std::unexpected(e.error());
    if (auto e = get(r.c2b, p + "conv2.bias", {cout}); !e) return std::unexpected(e.error());
    if (cin != cout) {
      r.shortcut = true;
      if (auto e = get(r.sw, p + "conv_shortcut.weight", {cout, cin, 1, 1}); !e)
        return std::unexpected(e.error());
      if (auto e = get(r.sb, p + "conv_shortcut.bias", {cout}); !e)
        return std::unexpected(e.error());
    }
    return r;
  }

  std::expected<void, std::string> load_weights() {
    const float scale = 1.0f / std::sqrt(static_cast<float>(cfg.channels[0]));
    auto get = [&](D& dst, const std::string& name,
                   std::vector<int64_t> shape, float s = 1.0f)
        -> std::expected<void, std::string> {
      auto d = load(name, std::move(shape), s);
      if (!d) return std::unexpected(d.error());
      dst = *d;
      return {};
    };
    const int64_t L = cfg.latent_channels;
    // BatchNorm statistics stay on the host: the pipeline applies them as a
    // per-channel affine transform before the pixel-unshuffle.
    {
      // These are indexed by the DiT's latent channel (128), not by the VAE's
      // 32-channel input: the unnormalization runs before the unshuffle.
      auto host = [&](const char* name,
                      std::vector<float>& dst) -> std::expected<void, std::string> {
        const safetensors::TensorInfo* info = reader.tensor(name);
        if (info == nullptr) return std::unexpected(std::string("missing tensor ") + name);
        if (info->dtype != safetensors::DType::F32)
          return std::unexpected(std::string(name) + ": expected f32");
        auto span = reader.tensor_data(name);
        if (!span) return std::unexpected(std::string("cannot read ") + name);
        const float* src = reinterpret_cast<const float*>(span->data());
        dst.assign(src, src + info->n_elements);
        return {};
      };
      if (auto e = host("bn.running_mean", bn.running_mean); !e)
        return std::unexpected(e.error());
      if (auto e = host("bn.running_var", bn.running_var); !e)
        return std::unexpected(e.error());
      if (bn.running_mean.size() != bn.running_var.size())
        return std::unexpected("VAE BatchNorm statistics size mismatch");
    }
    const int64_t C0 = cfg.channels[0];
    const int64_t OC = cfg.out_channels;

    if (auto e = get(postq_w, "post_quant_conv.weight", {L, L, 1, 1}); !e) return e;
    if (auto e = get(postq_b, "post_quant_conv.bias", {L}); !e) return e;
    if (auto e = get(conv_in_w, "decoder.conv_in.weight", {C0, L, 3, 3}); !e) return e;
    if (auto e = get(conv_in_b, "decoder.conv_in.bias", {C0}); !e) return e;

    {
      auto r0 = load_resnet("decoder.mid_block.resnets.0", C0, C0);
      if (!r0) return std::unexpected(r0.error());
      mid0 = *r0;
      auto r1 = load_resnet("decoder.mid_block.resnets.1", C0, C0);
      if (!r1) return std::unexpected(r1.error());
      mid1 = *r1;
    }
    {
      const std::string p = "decoder.mid_block.attentions.0.";
      attn.c = C0;
      attn.scale = scale;
      if (auto e = get(attn.gw, p + "group_norm.weight", {C0}); !e) return e;
      if (auto e = get(attn.gb, p + "group_norm.bias", {C0}); !e) return e;
      if (auto e = get(attn.qw, p + "to_q.weight", {C0, C0}, scale); !e) return e;
      if (auto e = get(attn.qb, p + "to_q.bias", {C0}, scale); !e) return e;
      if (auto e = get(attn.kw, p + "to_k.weight", {C0, C0}); !e) return e;
      if (auto e = get(attn.kb, p + "to_k.bias", {C0}); !e) return e;
      if (auto e = get(attn.vw, p + "to_v.weight", {C0, C0}); !e) return e;
      if (auto e = get(attn.vb, p + "to_v.bias", {C0}); !e) return e;
      if (auto e = get(attn.ow, p + "to_out.0.weight", {C0, C0}); !e) return e;
      if (auto e = get(attn.ob, p + "to_out.0.bias", {C0}); !e) return e;
    }

    int64_t cin = C0;
    for (size_t i = 0; i < cfg.channels.size(); ++i) {
      UpBlock b;
      b.cin = cin;
      b.cout = cfg.channels[i];
      for (int64_t j = 0; j <= cfg.layers_per_block; ++j) {
        const std::string p = "decoder.up_blocks." + std::to_string(i) +
                              ".resnets." + std::to_string(j);
        auto rn = load_resnet(p, j == 0 ? cin : b.cout, b.cout);
        if (!rn) return std::unexpected(rn.error());
        b.resnets.push_back(*rn);
      }
      if (i + 1 < cfg.channels.size()) {
        b.upsample = true;
        const std::string p = "decoder.up_blocks." + std::to_string(i) +
                              ".upsamplers.0.conv.";
        if (auto e = get(b.uw, p + "weight", {b.cout, b.cout, 3, 3}); !e) return e;
        if (auto e = get(b.ub, p + "bias", {b.cout}); !e) return e;
      }
      up.push_back(std::move(b));
      cin = cfg.channels[i];
    }

    const int64_t CL = cfg.channels.back();
    if (auto e = get(out_norm_w, "decoder.conv_norm_out.weight", {CL}); !e) return e;
    if (auto e = get(out_norm_b, "decoder.conv_norm_out.bias", {CL}); !e) return e;
    if (auto e = get(conv_out_w, "decoder.conv_out.weight", {OC, CL, 3, 3}); !e) return e;
    if (auto e = get(conv_out_b, "decoder.conv_out.bias", {OC}); !e) return e;
    return {};
  }

#endif // SONICBOOM_USE_CUDA
};

namespace {

#if defined(SONICBOOM_USE_CUDA)

using Buf = sonicboom::vae::detail::VaeDevBuf;
using sonicboom::vae::detail::aten_vae_add;
using sonicboom::vae::detail::aten_vae_add_row;
using sonicboom::vae::detail::aten_vae_bmm;
using sonicboom::vae::detail::aten_vae_bmm_nt;
using sonicboom::vae::detail::aten_vae_conv2d;
using sonicboom::vae::detail::aten_vae_group_norm;
using sonicboom::vae::detail::aten_vae_silu;
using sonicboom::vae::detail::aten_vae_softmax;
using sonicboom::vae::detail::aten_vae_transpose;
using sonicboom::vae::detail::aten_vae_upsample2x;

// A decode-scoped set of device allocations, freed on scope exit.
struct Arena {
  std::vector<float*> ptrs;

  float* alloc(int64_t n) {
    void* p = nullptr;
    if (cudaMalloc(&p, static_cast<size_t>(n) * sizeof(float)) != cudaSuccess)
      return nullptr;
    ptrs.push_back(static_cast<float*>(p));
    return static_cast<float*>(p);
  }
  ~Arena() {
    for (float* p : ptrs) cudaFree(p);
  }
};

std::expected<void, std::string> check(bool ok, const char* what) {
  if (!ok) return std::unexpected(std::string("VAE ") + what + " failed");
  return {};
}

std::expected<void, std::string> dump(std::vector<VaeStage>* stages,
                                      const char* name, float* src, int64_t c,
                                      int64_t h, int64_t w) {
  if (stages == nullptr) return {};
  VaeStage s;
  s.name = name;
  s.n = 1;
  s.c = c;
  s.h = h;
  s.w = w;
  s.values.resize(static_cast<size_t>(c * h * w));
  if (cudaMemcpy(s.values.data(), src,
                 s.values.size() * sizeof(float),
                 cudaMemcpyDeviceToHost) != cudaSuccess) {
    return std::unexpected(std::string("VAE dump ") + name + " failed");
  }
  stages->push_back(std::move(s));
  return {};
}

template <class ResNet>
std::expected<void, std::string> resnet(const ResNet& rn,
                                        float* x, float* out, float* t1,
                                        float* t2, int64_t H, int64_t W,
                                        int64_t groups, double eps,
                                        float* gn_mean, float* gn_rstd) {
  const int64_t cin = rn.cin, cout = rn.cout, hw = H * W;
  if (auto e = check(aten_vae_group_norm(t1, gn_mean, gn_rstd, x, rn.n1w.p,
                                         rn.n1b.p, 1, cin, hw, groups, eps),
                     "norm1"); !e) return e;
  if (auto e = check(aten_vae_silu(t2, t1, cin * hw), "silu1"); !e) return e;
  if (auto e = check(aten_vae_conv2d(out, t2, rn.c1w.p, rn.c1b.p, 1, cin, H, W,
                                     cout, 3, 3, 1, 1, 1, 1, H, W),
                     "conv1"); !e) return e;
  if (auto e = check(aten_vae_group_norm(t1, gn_mean, gn_rstd, out, rn.n2w.p,
                                         rn.n2b.p, 1, cout, hw, groups, eps),
                     "norm2"); !e) return e;
  if (auto e = check(aten_vae_silu(t2, t1, cout * hw), "silu2"); !e) return e;
  if (auto e = check(aten_vae_conv2d(t1, t2, rn.c2w.p, rn.c2b.p, 1, cout, H, W,
                                     cout, 3, 3, 1, 1, 1, 1, H, W),
                     "conv2"); !e) return e;
  if (rn.shortcut) {
    if (auto e = check(aten_vae_conv2d(t2, x, rn.sw.p, rn.sb.p, 1, cin, H, W,
                                       cout, 1, 1, 1, 1, 0, 0, H, W),
                       "conv_shortcut"); !e) return e;
    return check(aten_vae_add(out, t1, t2, cout * hw), "residual");
  }
  return check(aten_vae_add(out, t1, x, cout * hw), "residual");
}

// Query-row band for the [S][S] attention planes. A full score plane is S^2
// floats, which at the 1024x1024 latent size (S = 16384) is 1 GiB each; a band
// of query rows produces the same softmax without materializing the full
// plane. Sizes within the budget use one band, i.e. the unbanded path.
constexpr int64_t kAttnPlaneBudget = int64_t(16) << 20;  // floats, 64 MiB
int64_t attn_band(int64_t s) {
  if (s <= 0) return 1;
  return std::max<int64_t>(1, std::min<int64_t>(s, kAttnPlaneBudget / s));
}

template <class Attn>
std::expected<void, std::string> attention(
    const Attn& a, float* x, float* out, int64_t H, int64_t W,
    int64_t groups, double eps, float* scratch, float* gn_mean,
    float* gn_rstd) {
  const int64_t C = a.c, S = H * W, BQ = attn_band(S);
  float* gn = scratch;              // [C][S]
  float* q = gn + C * S;            // [S][C]
  float* k = q + C * S;             // [S][C]
  float* v = k + C * S;             // [S][C]
  float* u = v + C * S;             // [S][C]
  float* prod = u + C * S;          // [BQ][S] or [BQ][C]
  float* probs = prod + BQ * S;     // [BQ][S]

  if (auto e = check(aten_vae_group_norm(gn, gn_mean, gn_rstd, x, a.gw.p,
                                         a.gb.p, 1, C, S, groups, eps),
                     "attn norm"); !e) return e;
  if (auto e = check(aten_vae_transpose(u, gn, C, S), "attn transpose"); !e)
    return e;
  if (auto e = check(aten_vae_bmm_nt(q, u, a.qw.p, 1, S, C, C), "attn q"); !e)
    return e;
  if (auto e = check(aten_vae_add_row(q, q, a.qb.p, S, C), "attn q bias"); !e)
    return e;
  if (auto e = check(aten_vae_bmm_nt(k, u, a.kw.p, 1, S, C, C), "attn k"); !e)
    return e;
  if (auto e = check(aten_vae_add_row(k, k, a.kb.p, S, C), "attn k bias"); !e)
    return e;
  if (auto e = check(aten_vae_bmm_nt(v, u, a.vw.p, 1, S, C, C), "attn v"); !e)
    return e;
  if (auto e = check(aten_vae_add_row(v, v, a.vb.p, S, C), "attn v bias"); !e)
    return e;
  for (int64_t q0 = 0; q0 < S; q0 += BQ) {
    const int64_t b = std::min<int64_t>(BQ, S - q0);
    if (auto e = check(aten_vae_bmm_nt(prod, q + q0 * C, k, 1, b, C, S),
                       "attn scores"); !e) return e;
    if (auto e = check(aten_vae_softmax(probs, prod, b, S), "attn softmax"); !e)
      return e;
    if (auto e = check(aten_vae_bmm(prod, probs, v, 1, b, S, C), "attn pv"); !e)
      return e;
    if (auto e = check(aten_vae_bmm_nt(u + q0 * C, prod, a.ow.p, 1, b, C, C),
                       "attn out"); !e) return e;
    if (auto e = check(aten_vae_add_row(u + q0 * C, u + q0 * C, a.ob.p, b, C),
                       "attn out bias"); !e) return e;
  }
  if (auto e = check(aten_vae_transpose(gn, u, S, C), "attn transpose back"); !e)
    return e;
  return check(aten_vae_add(out, gn, x, C * S), "attn residual");
}

#endif // SONICBOOM_USE_CUDA

} // namespace

VaeDecoder::VaeDecoder(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
VaeDecoder::~VaeDecoder() = default;

const VaeConfig& VaeDecoder::config() const noexcept { return impl_->cfg; }
const VaeBatchNorm& VaeDecoder::batch_norm() const noexcept { return impl_->bn; }
int64_t VaeDecoder::weight_bytes() const noexcept { return impl_->bytes; }

void VaeDecoder::release_cached_memory() const {
  vae::detail::aten_vae_release_cached_memory();
}

std::expected<std::unique_ptr<VaeDecoder>, std::string> VaeDecoder::load(
    const std::filesystem::path& dir) {
  auto text = read_text(dir / "config.json");
  if (!text) return std::unexpected(text.error());
  std::string error;
  auto root = safetensors::detail::parse_json(*text, &error);
  if (!root) return std::unexpected("config.json: " + error);

  VaeConfig cfg;
  auto int_of = [&](const char* key) -> std::expected<int64_t, std::string> {
    const auto* v = root->find(key);
    if (!v || !v->is_integer)
      return std::unexpected(std::string("config.json: ") + key + " missing");
    return v->integer;
  };
  if (const auto* eps = root->find("batch_norm_eps"); eps && eps->type !=
          safetensors::detail::JsonValue::Type::Null) {
    cfg.bn_eps = eps->number;
  }
  for (auto [key, dest] : {std::pair{"latent_channels", &cfg.latent_channels},
                           std::pair{"out_channels", &cfg.out_channels},
                           std::pair{"layers_per_block", &cfg.layers_per_block},
                           std::pair{"norm_num_groups", &cfg.norm_groups}}) {
    auto got = int_of(key);
    if (!got) return std::unexpected(got.error());
    *dest = *got;
  }
  // The decoder reverses the configured block widths: the config lists them
  // from the output end (FLUX.2: 96,192,384,384).
  const auto* block_out = root->find("decoder_block_out_channels");
  if (block_out == nullptr) block_out = root->find("block_out_channels");
  if (block_out == nullptr || block_out->type !=
                                  safetensors::detail::JsonValue::Type::Array)
    return std::unexpected("config.json: no block_out_channels");
  cfg.channels.clear();
  for (const auto& v : block_out->array) {
    if (!v.is_integer)
      return std::unexpected("config.json: block_out_channels not integers");
    cfg.channels.push_back(v.integer);
  }
  std::reverse(cfg.channels.begin(), cfg.channels.end());
  if (cfg.channels.empty() || cfg.channels.size() != 4)
    return std::unexpected("config.json: expected four decoder block widths");

  auto reader = safetensors::Reader::load(
      dir / "diffusion_pytorch_model.safetensors");
  if (!reader) return std::unexpected("VAE weights: " + reader.error().message);

  auto impl = std::make_unique<Impl>(std::move(*reader));
  impl->cfg = cfg;
  impl->bn.eps = cfg.bn_eps;

#if defined(SONICBOOM_USE_CUDA)
  if (auto ok = impl->load_weights(); !ok)
    return std::unexpected(ok.error());
#else
  return std::unexpected("MOSS-Image VAE decode requires CUDA");
#endif
  return std::unique_ptr<VaeDecoder>(new VaeDecoder(std::move(impl)));
}

#if !defined(SONICBOOM_USE_CUDA)

std::expected<std::vector<float>, std::string> VaeDecoder::decode(
    std::span<const float>, int64_t, int64_t, std::vector<VaeStage>*) const {
  return std::unexpected("MOSS-Image VAE decode requires CUDA");
}

#else

std::expected<std::vector<float>, std::string> VaeDecoder::decode(
    std::span<const float> latent, int64_t height, int64_t width,
    std::vector<VaeStage>* stages) const {
  const VaeConfig& cfg = impl_->cfg;
  const int64_t L = cfg.latent_channels;
  const int64_t s = height * width;
  if (static_cast<int64_t>(latent.size()) != L * s)
    return std::unexpected("VAE decode: latent size does not match shape");

  // Every activation level fits in this bound: 384 ch at 16x the latent area,
  // 192 ch at 64x, and 96 ch at 64x are all <= 12288 samples per latent pixel.
  const int64_t kLevel = 12288 * s;
  Arena arena;
  float* buf[4];
  for (float*& b : buf) {
    b = arena.alloc(kLevel);
    if (b == nullptr) return std::unexpected("VAE decode: cudaMalloc failed");
  }
  // gn + q/k/v/u (five [C][S] buffers) plus the banded scores and softmax.
  const int64_t attn_scratch =
      5 * cfg.channels[0] * s + 2 * attn_band(s) * s;
  float* scratch = arena.alloc(attn_scratch);
  float* gn_mean = arena.alloc(cfg.norm_groups);
  float* gn_rstd = arena.alloc(cfg.norm_groups);
  float* z = arena.alloc(L * s);
  // conv_out runs after three 2x upsamplers: the image is 64x the latent area.
  float* image = arena.alloc(cfg.out_channels * 64 * s);
  if (scratch == nullptr || gn_mean == nullptr || gn_rstd == nullptr ||
      z == nullptr || image == nullptr)
    return std::unexpected("VAE decode: cudaMalloc failed");

  if (cudaMemcpy(z, latent.data(), static_cast<size_t>(L * s) * sizeof(float),
                 cudaMemcpyHostToDevice) != cudaSuccess)
    return std::unexpected("VAE decode: latent upload failed");

  const int64_t G = cfg.norm_groups;
  const double eps = cfg.eps;
  const int64_t C0 = cfg.channels[0];

  // post_quant_conv: 1x1, keeps the latent shape.
  if (auto e = check(aten_vae_conv2d(buf[1], z, impl_->postq_w.p,
                                     impl_->postq_b.p, 1, L, height, width, L,
                                     1, 1, 1, 1, 0, 0, height, width),
                     "post_quant_conv"); !e) return std::unexpected(e.error());
  // conv_in
  if (auto e = check(aten_vae_conv2d(buf[0], buf[1], impl_->conv_in_w.p,
                                     impl_->conv_in_b.p, 1, L, height, width,
                                     C0, 3, 3, 1, 1, 1, 1, height, width),
                     "conv_in"); !e) return std::unexpected(e.error());
  if (auto e = dump(stages, "conv_in", buf[0], C0, height, width); !e)
    return std::unexpected(e.error());

  int64_t H = height, W = width;
  int64_t Cin = C0;

  // Mid block: resnet -> attention -> resnet, all at the latent resolution.
  if (auto e = resnet(impl_->mid0, buf[0], buf[1], buf[2], buf[3], H, W, G,
                      eps, gn_mean, gn_rstd); !e)
    return std::unexpected(e.error());
  if (auto e = dump(stages, "mid_block.resnets.0", buf[1], Cin, H, W); !e)
    return std::unexpected(e.error());
  if (auto e = attention(impl_->attn, buf[1], buf[0], H, W, G, eps, scratch,
                         gn_mean, gn_rstd); !e)
    return std::unexpected(e.error());
  if (auto e = dump(stages, "mid_block.attentions.0", buf[0], Cin, H, W); !e)
    return std::unexpected(e.error());
  if (auto e = resnet(impl_->mid1, buf[0], buf[1], buf[2], buf[3], H, W, G,
                      eps, gn_mean, gn_rstd); !e)
    return std::unexpected(e.error());
  if (auto e = dump(stages, "mid_block.resnets.1", buf[1], Cin, H, W); !e)
    return std::unexpected(e.error());
  std::swap(buf[0], buf[1]);

  for (size_t i = 0; i < impl_->up.size(); ++i) {
    const auto& blk = impl_->up[i];
    for (size_t j = 0; j < blk.resnets.size(); ++j) {
      const auto& rn = blk.resnets[j];
      if (auto e = resnet(rn, buf[0], buf[1], buf[2], buf[3], H, W, G, eps,
                          gn_mean, gn_rstd); !e)
        return std::unexpected(e.error());
      std::swap(buf[0], buf[1]);
      Cin = rn.cout;
      if (auto e = dump(stages,
                        ("up_blocks." + std::to_string(i) + ".resnets." +
                         std::to_string(j)).c_str(),
                        buf[0], Cin, H, W); !e)
        return std::unexpected(e.error());
    }
    if (blk.upsample) {
      if (auto e = check(aten_vae_upsample2x(buf[1], buf[0], 1, Cin, H, W),
                         "upsample"); !e) return std::unexpected(e.error());
      if (auto e = check(aten_vae_conv2d(buf[0], buf[1], blk.uw.p, blk.ub.p, 1,
                                         Cin, 2 * H, 2 * W, blk.cout, 3, 3, 1,
                                         1, 1, 1, 2 * H, 2 * W),
                         "upsampler conv"); !e)
        return std::unexpected(e.error());
      H *= 2;
      W *= 2;
      Cin = blk.cout;
      if (auto e = dump(stages,
                        ("up_blocks." + std::to_string(i) + ".upsamplers.0").c_str(),
                        buf[0], Cin, H, W); !e)
        return std::unexpected(e.error());
    }
  }

  // conv_norm_out (GroupNorm, dumped before the SiLU) -> SiLU -> conv_out.
  if (auto e = check(aten_vae_group_norm(buf[1], gn_mean, gn_rstd, buf[0],
                                         impl_->out_norm_w.p, impl_->out_norm_b.p,
                                         1, Cin, H * W, G, eps),
                     "conv_norm_out"); !e) return std::unexpected(e.error());
  if (auto e = dump(stages, "conv_norm_out", buf[1], Cin, H, W); !e)
    return std::unexpected(e.error());
  if (auto e = check(aten_vae_silu(buf[2], buf[1], Cin * H * W), "conv_act"); !e)
    return std::unexpected(e.error());
  if (auto e = check(aten_vae_conv2d(image, buf[2], impl_->conv_out_w.p,
                                     impl_->conv_out_b.p, 1, Cin, H, W,
                                     cfg.out_channels, 3, 3, 1, 1, 1, 1, H, W),
                     "conv_out"); !e) return std::unexpected(e.error());
  if (auto e = dump(stages, "conv_out", image, cfg.out_channels, H, W); !e)
    return std::unexpected(e.error());

  if (cudaDeviceSynchronize() != cudaSuccess)
    return std::unexpected("VAE decode: device execution failed");

  std::vector<float> out(static_cast<size_t>(cfg.out_channels * H * W));
  if (cudaMemcpy(out.data(), image, out.size() * sizeof(float),
                 cudaMemcpyDeviceToHost) != cudaSuccess)
    return std::unexpected("VAE decode: output download failed");
  return out;
}

#endif

}  // namespace sonicboom::moss
