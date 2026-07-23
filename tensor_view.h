#ifndef TENSOR_VIEW_H
#define TENSOR_VIEW_H

#include "tensor.h"

Tensor *tensor_reshape(const Tensor *tensor, const size_t *new_shape,
                       size_t new_nidims);
Tensor *tensor_transpose(const Tensor *tensor, size_t dim1, size_t dim2);
Tensor *tensor_slice(const Tensor *tensor, size_t dim, size_t start,
                     size_t end, size_t step);

#endif
