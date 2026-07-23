#include "tensor_ops.h"
#include "tensor_broadcast.h"

#include <stdlib.h>
#include <string.h>

Tensor *tensor_add(const Tensor *tensor_a, const Tensor *tensor_b) {
    return tensor_binary_op(tensor_a, tensor_b, TENSOR_OP_ADD);
}

Tensor *tensor_sub(const Tensor *tensor_a, const Tensor *tensor_b) {
    return tensor_binary_op(tensor_a, tensor_b, TENSOR_OP_SUB);
}

Tensor *tensor_mul(const Tensor *tensor_a, const Tensor *tensor_b) {
    return tensor_binary_op(tensor_a, tensor_b, TENSOR_OP_MUL);
}

Tensor *tensor_div(const Tensor *tensor_a, const Tensor *tensor_b) {
    return tensor_binary_op(tensor_a, tensor_b, TENSOR_OP_DIV);
}

static size_t ptrdiff_magnitude(ptrdiff_t value) {
    return value >= 0 ? (size_t)value : (size_t)(-(value + 1)) + 1U;
}
/*
 * Move one logical step according to a stride.
 *
 * This helper assumes that the tensor and broadcast strides were already
 * validated, so the resulting offset remains inside the storage.
 */
static size_t tensor_advance_offset(size_t offset, ptrdiff_t stride) {
    if (stride >= 0) {
        return offset + (size_t)stride;
    }

    return offset - ptrdiff_magnitude(stride);
}


/*
 * Move back "count" logical steps.
 *
 * Used when a multidimensional index reaches the end of one dimension
 * and needs to wrap back to zero.
 */
static size_t tensor_rewind_offset(size_t offset,
                                   ptrdiff_t stride,
                                   size_t count) {
    size_t distance;

    if (stride == 0 || count == 0U) {
        return offset;
    }
    distance = ptrdiff_magnitude(stride) * count;

    if (stride > 0) {
        return offset - distance;
    }

    return offset + distance;
}
Tensor *tensor_binary_op(const Tensor *a,const Tensor *b,TensorOp operation) {
    Tensor *result;
    size_t out_nidims;

    size_t shape_inline[TENSOR_INLINE_DIMS];
    size_t indices_inline[TENSOR_INLINE_DIMS];

    ptrdiff_t a_strides_inline[TENSOR_INLINE_DIMS];
    ptrdiff_t b_strides_inline[TENSOR_INLINE_DIMS];

    size_t *out_shape = shape_inline;
    size_t *indices = indices_inline;

    ptrdiff_t *a_strides = a_strides_inline;
    ptrdiff_t *b_strides = b_strides_inline;

    size_t *size_metadata = NULL;
    ptrdiff_t *stride_metadata = NULL;

    size_t a_offset;
    size_t b_offset;

    status_t status;

    if (operation != TENSOR_OP_ADD && operation != TENSOR_OP_SUB &&
        operation != TENSOR_OP_MUL && operation != TENSOR_OP_DIV) {
        return NULL;
    }

    if (tensor_validate(a) != SUCCESS ||
        tensor_validate(b) != SUCCESS) {
        return NULL;
    }

    out_nidims = a->nidims > b->nidims
                     ? a->nidims
                     : b->nidims;

    /*
     * Most tensors have <= TENSOR_INLINE_DIMS dimensions,
     * so they require no temporary heap allocations.
     */
    if (out_nidims > TENSOR_INLINE_DIMS) {
        if (out_nidims > SIZE_MAX / (2U * sizeof(size_t)) ||
            out_nidims > SIZE_MAX / (2U * sizeof(ptrdiff_t))) {
            return NULL;
        }

        size_metadata =
            malloc(out_nidims * 2U * sizeof(*size_metadata));

        if (size_metadata == NULL) {
            return NULL;
        }

        stride_metadata =
            malloc(out_nidims * 2U * sizeof(*stride_metadata));

        if (stride_metadata == NULL) {
            free(size_metadata);
            return NULL;
        }

        out_shape = size_metadata;
        indices = size_metadata + out_nidims;

        a_strides = stride_metadata;
        b_strides = stride_metadata + out_nidims;
    }

    /*
     * Determine the output shape and verify broadcasting compatibility.
     */
    status = tensor_broadcast_shape(
        a,
        b,
        out_shape,
        out_nidims
    );

    if (status != SUCCESS) {
        free(stride_metadata);
        free(size_metadata);
        return NULL;
    }

    /*
     * Compute how A and B should be traversed relative to the
     * broadcasted output shape.
     */
    status = tensor_compute_broadcast_strides(
        a,
        out_shape,
        out_nidims,
        a_strides
    );

    if (status != SUCCESS) {
        free(stride_metadata);
        free(size_metadata);
        return NULL;
    }

    status = tensor_compute_broadcast_strides(
        b,
        out_shape,
        out_nidims,
        b_strides
    );

    if (status != SUCCESS) {
        free(stride_metadata);
        free(size_metadata);
        return NULL;
    }

    /*
     * The operation overwrites every result element,
     * so zero-initializing the result would be unnecessary work.
     */
    result = tensor_create_uninitialized(
        out_shape,
        out_nidims
    );

    if (result == NULL) {
        free(stride_metadata);
        free(size_metadata);
        return NULL;
    }

    memset(indices, 0, out_nidims * sizeof(*indices));

    /*
     * A and B may already be views, so traversal starts at
     * their existing offsets.
     */
    a_offset = a->offset;
    b_offset = b->offset;

    for (size_t linear = 0U;
         linear < result->nelements;
         ++linear) {

        float a_value = a->storage->data[a_offset];
        float b_value = b->storage->data[b_offset];

        switch (operation) {
            case TENSOR_OP_ADD:
                result->storage->data[linear] =
                    a_value + b_value;
                break;

            case TENSOR_OP_SUB:
                result->storage->data[linear] =
                    a_value - b_value;
                break;

            case TENSOR_OP_MUL:
                result->storage->data[linear] =
                    a_value * b_value;
                break;

            case TENSOR_OP_DIV:
                result->storage->data[linear] =
                    a_value / b_value;
                break;

            default:
                destroy_tensor(result);
                free(stride_metadata);
                free(size_metadata);
                return NULL;
        }

        /*
         * Increment the multidimensional output index.
         *
         * At the same time, update A and B offsets incrementally.
         * This avoids recomputing the complete offset from scratch
         * for every output element.
         */
        for (size_t dimension = out_nidims;
             dimension > 0U;
             --dimension) {

            size_t i = dimension - 1U;

            if (indices[i] + 1U < out_shape[i]) {
                ++indices[i];

                a_offset =
                    tensor_advance_offset(
                        a_offset,
                        a_strides[i]
                    );

                b_offset =
                    tensor_advance_offset(
                        b_offset,
                        b_strides[i]
                    );

                break;
            }

            /*
             * This dimension wrapped back to zero.
             */
            a_offset =
                tensor_rewind_offset(
                    a_offset,
                    a_strides[i],
                    indices[i]
                );

            b_offset =
                tensor_rewind_offset(
                    b_offset,
                    b_strides[i],
                    indices[i]
                );

            indices[i] = 0U;
        }
    }

    free(stride_metadata);
    free(size_metadata);

    return result;
}
