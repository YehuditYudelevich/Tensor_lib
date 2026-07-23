#include "tensor_broadcast.h"


status_t tensor_broadcast_shape(const Tensor *a,const Tensor *b,size_t *out_shape, size_t out_nidims) {
    size_t expected_nidims;
    size_t i;

    if (a == NULL || b == NULL || out_shape == NULL) {
        return NO_PARAMS;
    }

    if (tensor_validate(a) != SUCCESS ||
        tensor_validate(b) != SUCCESS) {
        return INVALID_TENSOR;
    }
    expected_nidims = a->nidims > b->nidims ? a->nidims : b->nidims;

    if (out_nidims != expected_nidims) {
        return INVALID_SHAPE;
    }

    for (i = 0U; i < out_nidims; ++i) {
        size_t a_dim = 1U;
        size_t b_dim = 1U;
        size_t out_index = out_nidims - 1U - i;

        if (i < a->nidims) {
            a_dim = a->shape[a->nidims - 1U - i];
        }

        if (i < b->nidims) {
            b_dim = b->shape[b->nidims - 1U - i];
        }

        if (a_dim != b_dim && a_dim != 1U && b_dim != 1U) {
            return INVALID_SHAPE;
        }

        out_shape[out_index] = a_dim > b_dim ? a_dim : b_dim;
    }

    return SUCCESS;
}
status_t tensor_compute_broadcast_strides(const Tensor *tensor,const size_t *output_shape,size_t output_nidims,ptrdiff_t *out_strides) {

    size_t dim_offset;
    size_t i;

    if (tensor == NULL || output_shape == NULL || out_strides == NULL) {
        return NO_PARAMS;
    }
    if (tensor_validate(tensor) != SUCCESS) {
        return INVALID_TENSOR;
    }
    if (output_nidims < tensor->nidims) {
        return INVALID_SHAPE;
    }

    dim_offset = output_nidims - tensor->nidims;

    for (i = 0U; i < output_nidims; ++i) {
        if (output_shape[i] == 0U) {
            return INVALID_SHAPE;
        }
        if (i < dim_offset) {
            /*
             * Leading dimensions that do not exist in the original tensor
             * are implicitly dimensions of size 1.
             */
            out_strides[i] = 0;
            continue;
        }

        {
            size_t tensor_dim = i - dim_offset;
            size_t tensor_extent = tensor->shape[tensor_dim];
            size_t output_extent = output_shape[i];

            if (tensor_extent == output_extent) {
                out_strides[i] = tensor->strides[tensor_dim];
            } else if (tensor_extent == 1U) {
                /*
                 * Broadcasting this dimension means repeatedly reading
                 * the same physical element.
                 */
                out_strides[i] = 0;
            } else {
                return INVALID_SHAPE;
            }
        }
    }

    return SUCCESS;
}