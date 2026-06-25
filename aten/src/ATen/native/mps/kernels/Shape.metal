#include <ATen/native/mps/kernels/Shape.h>
#include <c10/metal/utils.h>
#include <metal_array>
#include <metal_stdlib>

using namespace metal;
using namespace c10::metal;

// Each thread copies a 16-byte chunk of an input into the output, using the
// widest aligned vector store its address allows, and a per-byte copy for runs
// that cross a slice boundary. See cat_out_mps_contiguous_impl for the
// geometry.
template <typename I>
kernel void cat_copy_contiguous(
    constant uchar* input [[buffer(0)]],
    device uchar* output [[buffer(1)]],
    constant CatCopyParams<I>& params [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
  uint pos = tid * 16;
  if (pos >= params.nbytes) {
    return;
  }
  I g = params.chunk_base + pos;
  I o = g / params.slice_bytes;
  I j = g - o * params.slice_bytes;
  constant uchar* inp = input + g;

  // A run that crosses a slice boundary (or the dispatch tail) maps to
  // discontiguous output, so copy it byte by byte, recomputing the destination.
  if (params.slice_bytes - j < 16 || pos + 16 > params.nbytes) {
    uint stop = min(pos + 16u, params.nbytes) - pos;
    for (uint i = 0; i < stop; i++) {
      I gg = g + i;
      I oo = gg / params.slice_bytes;
      I jj = gg - oo * params.slice_bytes;
      output[oo * params.out_stride_bytes + params.cat_off_bytes + jj] = inp[i];
    }
    return;
  }

  // Whole 16-byte run: copy with the widest store both endpoints are aligned
  // for.
  device uchar* out =
      output + o * params.out_stride_bytes + params.cat_off_bytes + j;
  uint align = (reinterpret_cast<device ulong>(out) |
                reinterpret_cast<constant ulong>(inp)) &
      15;
  if (align == 0) {
    *reinterpret_cast<device uint4*>(out) =
        *reinterpret_cast<constant uint4*>(inp);
  } else if ((align & 7) == 0) {
    for (uint k = 0; k < 2; k++) {
      reinterpret_cast<device uint2*>(out)[k] =
          reinterpret_cast<constant uint2*>(inp)[k];
    }
  } else if ((align & 3) == 0) {
    for (uint k = 0; k < 4; k++) {
      reinterpret_cast<device uint*>(out)[k] =
          reinterpret_cast<constant uint*>(inp)[k];
    }
  } else if ((align & 1) == 0) {
    for (uint k = 0; k < 8; k++) {
      reinterpret_cast<device ushort*>(out)[k] =
          reinterpret_cast<constant ushort*>(inp)[k];
    }
  } else {
    for (uint k = 0; k < 16; k++) {
      out[k] = inp[k];
    }
  }
}

#define REGISTER_CAT_COPY_OP(I, SUFFIX)                  \
  template [[host_name("cat_copy_contiguous_" #SUFFIX)]] \
  kernel void cat_copy_contiguous<I>(                    \
      constant uchar * input [[buffer(0)]],              \
      device uchar * output [[buffer(1)]],               \
      constant CatCopyParams<I> & params [[buffer(2)]],  \
      uint tid [[thread_position_in_grid]]);

REGISTER_CAT_COPY_OP(uint, u32);
REGISTER_CAT_COPY_OP(ulong, u64);

template <typename I, typename T_in, typename T_out>
kernel void cat(
    constant T_in* input [[buffer(0)]],
    device T_out* output [[buffer(1)]],
    constant CatSharedParams<I>& shared_params [[buffer(2)]],
    constant CatInputParams<I>& input_params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  auto ndim = shared_params.ndim;
  auto cat_dim = shared_params.cat_dim;
  constant auto& output_strides = shared_params.output_strides;

  auto cat_dim_offset = input_params.cat_dim_offset;
  auto input_element_offset = input_params.input_element_offset;
  constant auto& input_strides = input_params.input_strides;
  constant auto& input_sizes = input_params.input_sizes;

  auto input_element_idx = static_cast<I>(tid) + input_element_offset;
  I input_offset = 0;
  I output_offset = 0;

  for (auto dim = ndim - 1; dim >= 0; dim--) {
    auto dim_size = input_sizes[dim];
    auto input_dim_idx = input_element_idx % dim_size;
    auto output_dim_idx =
        input_dim_idx + ((dim == cat_dim) ? cat_dim_offset : 0);

    input_offset += input_strides[dim] * input_dim_idx;
    output_offset += output_strides[dim] * output_dim_idx;

    input_element_idx = input_element_idx / dim_size;
  }

  output[output_offset] = static_cast<T_out>(input[input_offset]);
}

#define REGISTER_CAT_OP(I, T_in, T_out)                          \
  template [[host_name("cat_" #I "_" #T_in "_" #T_out)]]         \
  kernel void cat<I, T_in, T_out>(                               \
      constant T_in * input [[buffer(0)]],                       \
      device T_out * output [[buffer(1)]],                       \
      constant CatSharedParams<I> & shared_params [[buffer(2)]], \
      constant CatInputParams<I> & input_params [[buffer(3)]],   \
      uint tid [[thread_position_in_grid]]);

#define REGISTER_CAT_OP_ALL_INPUT_TYPES(I, T_out) \
  REGISTER_CAT_OP(I, float, T_out);               \
  REGISTER_CAT_OP(I, half, T_out);                \
  REGISTER_CAT_OP(I, bfloat, T_out);              \
  REGISTER_CAT_OP(I, int, T_out);                 \
  REGISTER_CAT_OP(I, uint, T_out);                \
  REGISTER_CAT_OP(I, long, T_out);                \
  REGISTER_CAT_OP(I, ulong, T_out);               \
  REGISTER_CAT_OP(I, short, T_out);               \
  REGISTER_CAT_OP(I, ushort, T_out);              \
  REGISTER_CAT_OP(I, char, T_out);                \
  REGISTER_CAT_OP(I, uchar, T_out);               \
  REGISTER_CAT_OP(I, bool, T_out);

#define REGISTER_CAT_FOR_INDEX_TYPE(I)        \
  REGISTER_CAT_OP_ALL_INPUT_TYPES(I, float);  \
  REGISTER_CAT_OP_ALL_INPUT_TYPES(I, half);   \
  REGISTER_CAT_OP_ALL_INPUT_TYPES(I, bfloat); \
  REGISTER_CAT_OP_ALL_INPUT_TYPES(I, int);    \
  REGISTER_CAT_OP_ALL_INPUT_TYPES(I, uint);   \
  REGISTER_CAT_OP_ALL_INPUT_TYPES(I, long);   \
  REGISTER_CAT_OP_ALL_INPUT_TYPES(I, ulong);  \
  REGISTER_CAT_OP_ALL_INPUT_TYPES(I, short);  \
  REGISTER_CAT_OP_ALL_INPUT_TYPES(I, ushort); \
  REGISTER_CAT_OP_ALL_INPUT_TYPES(I, char);   \
  REGISTER_CAT_OP_ALL_INPUT_TYPES(I, uchar);  \
  REGISTER_CAT_OP_ALL_INPUT_TYPES(I, bool);   \
                                              \
  REGISTER_CAT_OP(I, float2, float2);         \
  REGISTER_CAT_OP(I, half2, half2);

REGISTER_CAT_FOR_INDEX_TYPE(int64_t);
REGISTER_CAT_FOR_INDEX_TYPE(int32_t);
