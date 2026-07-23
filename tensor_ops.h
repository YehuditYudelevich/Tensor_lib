#ifndef TENSOR_OPS_H
#define TENSOR_OPS_H

#include "tensor.h"

typedef enum {
    TENSOR_OP_ADD,
    TENSOR_OP_SUB,
    TENSOR_OP_MUL,
    TENSOR_OP_DIV
} TensorOp;

/* Return a newly allocated tensor after applying an element-wise operation.
 * Input tensors are broadcast using NumPy-style trailing-dimension rules. */
Tensor *tensor_add(const Tensor *tensor_a, const Tensor *tensor_b);
Tensor *tensor_sub(const Tensor *tensor_a, const Tensor *tensor_b);
Tensor *tensor_mul(const Tensor *tensor_a, const Tensor *tensor_b);
Tensor *tensor_div(const Tensor *tensor_a, const Tensor *tensor_b);
Tensor *tensor_binary_op(const Tensor *tensor_a, const Tensor *tensor_b,
                         TensorOp op);

#endif