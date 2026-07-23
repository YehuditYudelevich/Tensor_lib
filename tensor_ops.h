#include "tensor.h"
#include "tensor_broadcast.h"

#include <stdlib.h>
#include <string.h>
typedef enum {
    ADD,
    SUB,
    MUL,
    DIV
} TensorOp;

Tensor *tensor_add(Tensor *tensor_a, Tensor *tensor_b);
Tensor *tensor_sub(Tensor *tensor_a, Tensor *tensor_b);
Tensor *tensor_mul(Tensor *tensor_a, Tensor *tensor_b);
Tensor *tensor_div(Tensor *tensor_a, Tensor *tensor_b);
Tensor *tensor_binary_op(Tensor *tensor_a, Tensor *tensor_b, TensorOp op);