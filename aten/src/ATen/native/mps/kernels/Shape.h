#pragma once
#include <c10/metal/common.h>

template <typename idx_type_t = int64_t, unsigned N = c10::metal::max_ndim>
struct CatSharedParams {
  int32_t ndim;
  int32_t cat_dim;
  ::c10::metal::array<idx_type_t, N> output_strides;
  ::c10::metal::array<idx_type_t, N> output_sizes;
};

template <typename idx_type_t = int64_t, unsigned N = c10::metal::max_ndim>
struct CatInputParams {
  idx_type_t cat_dim_offset;
  idx_type_t input_element_offset;
  ::c10::metal::array<idx_type_t, N> input_strides;
  ::c10::metal::array<idx_type_t, N> input_sizes;
};

// Byte geometry for one dispatch of the contiguous `cat` fast path.
template <typename idx_type_t = uint32_t>
struct CatCopyParams {
  idx_type_t chunk_base;
  idx_type_t slice_bytes;
  idx_type_t out_stride_bytes;
  idx_type_t cat_off_bytes;
  uint32_t nbytes;
};
