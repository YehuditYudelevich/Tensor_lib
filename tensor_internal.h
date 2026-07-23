#ifndef TENSOR_INTERNAL_H
#define TENSOR_INTERNAL_H

#include "tensor.h"

/* Create a C-order metadata view using existing storage without copying data. */
Tensor *tensor_create_contiguous_view(TensorStorage *storage, size_t offset,
                                      const size_t *shape, size_t nidims);

/* Clone tensor metadata, retain the shared storage, and return a view. */
Tensor *tensor_clone_view_metadata(const Tensor *tensor);

#endif
