// Copyright (c) 2023-2026 The ggml authors
//
// This source code is licensed under the MIT license found in the
// LICENSES/ggml-MIT.txt file in the root directory of this source tree.
//
// SPDX-License-Identifier: MIT

#include "quantize_q5_k.h"

#include <algorithm>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace sonicboom::quant {
namespace {
int nearest_int(float x) {
  const float v = x + 12582912.0f;
  return (std::bit_cast<uint32_t>(v) & 0x007fffff) - 0x00400000;
}

uint16_t fp32_to_fp16(float value) {
  uint32_t x = std::bit_cast<uint32_t>(value);
  uint32_t sign = (x >> 16) & 0x8000;
  int exp = int((x >> 23) & 0xff) - 127 + 15;
  uint32_t mant = x & 0x7fffff;
  if (exp <= 0) {
    if (exp < -10) return uint16_t(sign);
    mant |= 0x800000;
    const int shift = 14 - exp;
    uint32_t result = mant >> shift;
    const uint32_t rem = mant & ((uint32_t(1) << shift)-1);
    const uint32_t halfway = uint32_t(1) << (shift-1);
    if (rem > halfway || (rem == halfway && (result & 1))) ++result;
    return uint16_t(sign | result);
  }
  if (exp >= 31) return uint16_t(sign | 0x7c00 | (mant ? 0x0200 : 0));
  uint32_t result = (uint32_t(exp) << 10) | (mant >> 13);
  const uint32_t rem = mant & 0x1fff;
  if (rem > 0x1000 || (rem == 0x1000 && (result & 1))) ++result;
  return uint16_t(sign | result);
}

float fp16_to_fp32(uint16_t h) {
  uint32_t sign = uint32_t(h & 0x8000) << 16;
  uint32_t exp = (h >> 10) & 31, mant = h & 1023, bits;
  if (!exp) {
    if (!mant) bits = sign;
    else {
      exp = 113;
      while (!(mant & 0x400)) { mant <<= 1; --exp; }
      bits = sign | (exp << 23) | ((mant & 1023) << 13);
    }
  } else if (exp == 31) bits = sign | 0x7f800000 | (mant << 13);
  else bits = sign | ((exp + 112) << 23) | (mant << 13);
  return std::bit_cast<float>(bits);
}

float make_qkx2_quants(int n, int nmax, const float* x, const float* weights,
                       uint8_t* L, float* the_min, uint8_t* Laux,
                       float rmin, float rdelta, int nstep) {
  float min=x[0], max=x[0], sum_w=weights[0], sum_x=sum_w*x[0];
  for (int i=1;i<n;++i) {
    min=std::min(min,x[i]); max=std::max(max,x[i]);
    sum_w+=weights[i]; sum_x+=weights[i]*x[i];
  }
  if (min>0) min=0;
  if (max==min) { std::fill_n(L,n,0); *the_min=-min; return 0; }
  float iscale=nmax/(max-min), scale=1/iscale, best_error=0;
  for (int i=0;i<n;++i) {
    int q=std::clamp(nearest_int(iscale*(x[i]-min)),0,nmax);
    L[i]=uint8_t(q);
    float diff=scale*q+min-x[i]; best_error+=weights[i]*diff*diff;
  }
  for (int step=0;step<=nstep;++step) {
    iscale=(rmin+rdelta*step+nmax)/(max-min);
    float sum_l=0,sum_l2=0,sum_xl=0;
    for(int i=0;i<n;++i) {
      int q=std::clamp(nearest_int(iscale*(x[i]-min)),0,nmax);
      Laux[i]=uint8_t(q); float w=weights[i];
      sum_l+=w*q; sum_l2+=w*q*q; sum_xl+=w*q*x[i];
    }
    const float D=sum_w*sum_l2-sum_l*sum_l;
    if(D>0) {
      float this_scale=(sum_w*sum_xl-sum_x*sum_l)/D;
      float this_min=(sum_l2*sum_x-sum_l*sum_xl)/D;
      if(this_min>0) { this_min=0; this_scale=sum_xl/sum_l2; }
      float error=0;
      for(int i=0;i<n;++i) { float diff=this_scale*Laux[i]+this_min-x[i]; error+=weights[i]*diff*diff; }
      if(error<best_error) { std::copy_n(Laux,n,L); best_error=error; scale=this_scale; min=this_min; }
    }
  }
  *the_min=-min;
  return scale;
}

void get_scale_min(int j,const uint8_t* q,uint8_t& d,uint8_t& m) {
  if(j<4) { d=q[j]&63; m=q[j+4]&63; }
  else { d=(q[j+4]&15)|((q[j-4]>>6)<<4); m=(q[j+4]>>4)|((q[j]>>6)<<4); }
}

void quantize_block(const float* x,uint8_t* y) {
  uint8_t L[256]{}, Laux[32]{};
  float mins[8]{},scales[8]{},weights[32];
  float max_scale=0,max_min=0;
  for(int j=0;j<8;++j) {
    float sum_x2=0;
    for(int l=0;l<32;++l) sum_x2+=x[32*j+l]*x[32*j+l];
    float av_x=std::sqrt(sum_x2/32);
    for(int l=0;l<32;++l) weights[l]=av_x+std::fabs(x[32*j+l]);
    scales[j]=make_qkx2_quants(32,31,x+32*j,weights,L+32*j,&mins[j],Laux,-0.5f,0.1f,15);
    max_scale=std::max(max_scale,scales[j]); max_min=std::max(max_min,mins[j]);
  }
  uint8_t* packed_scales=y+4;
  std::fill_n(packed_scales,12,0);
  const float inv_scale=max_scale>0?63.f/max_scale:0;
  const float inv_min=max_min>0?63.f/max_min:0;
  for(int j=0;j<8;++j) {
    uint8_t ls=uint8_t(std::min(63,nearest_int(inv_scale*scales[j])));
    uint8_t lm=uint8_t(std::min(63,nearest_int(inv_min*mins[j])));
    if(j<4) { packed_scales[j]=ls; packed_scales[j+4]=lm; }
    else {
      packed_scales[j+4]=(ls&15)|((lm&15)<<4);
      packed_scales[j-4]|=(ls>>4)<<6; packed_scales[j]|=(lm>>4)<<6;
    }
  }
  const uint16_t dh=fp32_to_fp16(max_scale/63.f), dmh=fp32_to_fp16(max_min/63.f);
  y[0]=dh&255; y[1]=dh>>8; y[2]=dmh&255; y[3]=dmh>>8;
  for(int j=0;j<8;++j) {
    uint8_t sc,m; get_scale_min(j,packed_scales,sc,m);
    float d=fp16_to_fp32(dh)*sc, dm=fp16_to_fp32(dmh)*m;
    if(!d) continue;
    for(int i=0;i<32;++i) L[32*j+i]=uint8_t(std::clamp(nearest_int((x[32*j+i]+dm)/d),0,31));
  }
  uint8_t* qh=y+16; uint8_t* ql=y+48;
  std::fill_n(qh,32,0);
  uint8_t m1=1,m2=2;
  for(int n=0;n<256;n+=64) {
    for(int j=0;j<32;++j) {
      int l1=L[n+j],l2=L[n+j+32];
      if(l1>15) { l1-=16; qh[j]|=m1; }
      if(l2>15) { l2-=16; qh[j]|=m2; }
      ql[j]=uint8_t(l1|(l2<<4));
    }
    m1<<=2; m2<<=2; ql+=32;
  }
}
} // namespace

std::vector<uint8_t> quantize_f32_to_q5_k(const float* src,uint64_t n_elems) {
  assert(n_elems%256==0);
  std::vector<uint8_t> out((n_elems/256)*176);
  for(uint64_t i=0;i<n_elems/256;++i) quantize_block(src+i*256,out.data()+i*176);
  return out;
}
} // namespace sonicboom::quant
