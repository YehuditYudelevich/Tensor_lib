#include "tensor_view.h"
#include "tensor_internal.h"

Tensor *tensor_reshape(const Tensor *tensor, const size_t *new_shape,
                       size_t new_nidims) {
    size_t new_nelements = 1U;

    if (tensor == NULL || new_shape == NULL || new_nidims == 0U || !tensor_is_contiguous(tensor)) {
        return NULL;
    }
    for (size_t i = 0U; i < new_nidims; ++i) {
        if (new_shape[i] == 0U ||
            size_mul_overflows(new_nelements, new_shape[i])) {
            return NULL;
        }
        new_nelements *= new_shape[i];
    }
    if (new_nelements != tensor->nelements) {
        return NULL;
    }
    return tensor_create_contiguous_view(tensor->storage, tensor->offset,
                                         new_shape, new_nidims);
}

Tensor *tensor_transpose(const Tensor *tensor, size_t dim1, size_t dim2) {
    if(tensor == NULL || dim1 >= tensor->nidims || dim2 >= tensor->nidims) {
        return NULL;
    }
    Tensor *view;
    size_t temp_shape;
    ptrdiff_t temp_strides;
    view=tensor_clone_view_metadata(tensor);
    if(view == NULL) {
        return NULL;
    }
    temp_shape = view->shape[dim1];
    view->shape[dim1] = view->shape[dim2];
    view->shape[dim2] = temp_shape;
    temp_strides = view->strides[dim1];
    view->strides[dim1] = view->strides[dim2];
    view->strides[dim2] = temp_strides;
    return view;
}

static size_t stride_magnitude(ptrdiff_t stride) {
    return stride >= 0 ? (size_t)stride : (size_t)(-(stride + 1)) + 1U;
}

Tensor *tensor_slice(const Tensor *tensor, size_t dim, size_t start,
                     size_t end, size_t step) {
    Tensor *view;
    size_t magnitude;
    size_t offset_delta;
    size_t scaled_magnitude;
    size_t new_extent;
    size_t new_offset;
    size_t new_nelements;
    ptrdiff_t new_stride;

    if (tensor_validate(tensor) != SUCCESS || dim >= tensor->nidims ||
        step == 0U || start >= end || end > tensor->shape[dim]) {
        return NULL;
    }
    magnitude = stride_magnitude(tensor->strides[dim]);
    if (size_mul_overflows(start, magnitude) ||
        size_mul_overflows(magnitude, step)) {
        return NULL;
    }
    offset_delta = start * magnitude;
    scaled_magnitude = magnitude * step;
    if ((tensor->strides[dim] >= 0 && scaled_magnitude > (size_t)PTRDIFF_MAX) ||
        (tensor->strides[dim] < 0 &&
         scaled_magnitude > (size_t)PTRDIFF_MAX + 1U)) {
        return NULL;
    }

    if (tensor->strides[dim] < 0) {
        if (offset_delta > tensor->offset) {
            return NULL;
        }
        new_offset = tensor->offset - offset_delta;
        new_stride = scaled_magnitude == (size_t)PTRDIFF_MAX + 1U
                         ? PTRDIFF_MIN
                         : -(ptrdiff_t)scaled_magnitude;
    } else {
        if (size_add_overflows(tensor->offset, offset_delta)) {
            return NULL;
        }
        new_offset = tensor->offset + offset_delta;
        new_stride = (ptrdiff_t)scaled_magnitude;
    }

    new_extent = 1U + (end - start - 1U) / step;
    new_nelements = (tensor->nelements / tensor->shape[dim]) * new_extent;
    view = tensor_clone_view_metadata(tensor);
    if (view == NULL) {
        return NULL;
    }
    view->offset = new_offset;
    view->nelements = new_nelements;
    view->shape[dim] = new_extent;
    view->strides[dim] = new_stride;
    if (tensor_validate(view) != SUCCESS) {
        (void)destroy_tensor(view);
        return NULL;
    }
    return view;
}
