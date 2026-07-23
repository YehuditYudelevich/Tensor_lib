#include "tensor.h"

status_t tensor_broadcast_shape(const Tensor *a,const Tensor *b,size_t *out_shape, size_t out_nidims);
status_t tensor_compute_broadcast_strides(const Tensor *tensor, const size_t *output_shape, size_t output_nidims,ptrdiff_t *out_strides);