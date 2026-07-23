#include "tensor.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(_MSC_VER)
#include <malloc.h>
#endif

static size_t ptrdiff_magnitude(ptrdiff_t value) {
    return value >= 0 ? (size_t)value : (size_t)(-(value + 1)) + 1U;
}

static status_t tensor_metadata_heap_layout(size_t nidims, size_t *stride_offset,
                                            size_t *total_bytes) {

    const size_t stride_alignment = _Alignof(ptrdiff_t);
    size_t shape_bytes;
    size_t stride_bytes;
    size_t padding;

    if (nidims > SIZE_MAX / sizeof(size_t) || nidims > SIZE_MAX / sizeof(ptrdiff_t)) {
        return OVERFLOW;
    }
    shape_bytes = nidims * sizeof(size_t);
    stride_bytes = nidims * sizeof(ptrdiff_t);
    padding = (stride_alignment - (shape_bytes % stride_alignment)) % stride_alignment;
    if (size_add_overflows(shape_bytes, padding)) {
        return OVERFLOW;
    }
    *stride_offset = shape_bytes + padding;
    if (size_add_overflows(*stride_offset, stride_bytes)) {
        return OVERFLOW;
    }
    *total_bytes = *stride_offset + stride_bytes;
    return SUCCESS;
}

static void tensor_reset_metadata(Tensor *tensor) {
    tensor->shape = tensor->shape_inline;
    tensor->strides = tensor->strides_inline;
    tensor->metadata_heap = NULL;
}

static status_t tensor_assign_heap_metadata(Tensor *tensor, size_t nidims) {
    size_t stride_offset;
    size_t total_bytes;
    status_t status = tensor_metadata_heap_layout(nidims, &stride_offset, &total_bytes);

    if (status != SUCCESS) {
        return status;
    }
    tensor->metadata_heap = malloc(total_bytes);
    if (tensor->metadata_heap == NULL) {
        return ALLOCATION_FAILURE;
    }
    tensor->shape = tensor->metadata_heap;
    tensor->strides = (ptrdiff_t *)((unsigned char *)tensor->metadata_heap + stride_offset);
    return SUCCESS;
}

void *tensor_aligned_alloc(size_t alignment, size_t size) {
    size_t rounded_size;

    if (alignment == 0U || (alignment & (alignment - 1U)) != 0U || size == 0U) {
        return NULL;
    }
    if (size_add_overflows(size, alignment - 1U)) {
        return NULL;
    }
    rounded_size = ((size + alignment - 1U) / alignment) * alignment;
#if defined(_MSC_VER)
    return _aligned_malloc(rounded_size, alignment);
#else
    return aligned_alloc(alignment, rounded_size);
#endif
}

void tensor_aligned_free(void *ptr) {
#if defined(_MSC_VER)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

TensorStorage *storage_create(size_t nelements, bool zero_initialize) {
    TensorStorage *storage;
    size_t bytes;

    if (nelements == 0U || nelements > (size_t)PTRDIFF_MAX ||
        size_mul_overflows(nelements, sizeof(float))) {
        return NULL;
    }
    bytes = nelements * sizeof(float);
    storage = malloc(sizeof(*storage));
    if (storage == NULL) {
        return NULL;
    }
    storage->data = tensor_aligned_alloc(TENSOR_ALIGNMENT, bytes);
    if (storage->data == NULL) {
        free(storage);
        return NULL;
    }
    storage->nelements = nelements;
    storage->refcount = 1U;
    if (zero_initialize) {
        memset(storage->data, 0, bytes);
    }
    return storage;
}

status_t storage_retain(TensorStorage *storage) {
    if (storage == NULL || storage->data == NULL || storage->nelements == 0U ||
        storage->refcount == 0U) {
        return INVALID_TENSOR;
    }
    if (storage->refcount == SIZE_MAX) {
        return OVERFLOW;
    }
    ++storage->refcount;
    return SUCCESS;
}

status_t storage_release(TensorStorage *storage) {
    if (storage == NULL) {
        return NO_PARAMS;
    }
    if (storage->refcount == 0U) {
        return INVALID_TENSOR;
    }
    --storage->refcount;
    if (storage->refcount == 0U) {
        tensor_aligned_free(storage->data);
        free(storage);
    }
    return SUCCESS;
}

static Tensor *tensor_allocate_metadata(const size_t *shape, size_t nidims) {
    Tensor *tensor;
    size_t elements = 1U;
    size_t contiguous_stride = 1U;

    if (shape == NULL || nidims == 0U) {
        return NULL;
    }
    tensor = calloc(1U, sizeof(*tensor));
    if (tensor == NULL) {
        return NULL;
    }
    tensor_reset_metadata(tensor);
    if (nidims > TENSOR_INLINE_DIMS &&
        tensor_assign_heap_metadata(tensor, nidims) != SUCCESS) {
        free(tensor);
        return NULL;
    }
    for (size_t i = 0U; i < nidims; ++i) {
        if (shape[i] == 0U || size_mul_overflows(elements, shape[i])) {
            free(tensor->metadata_heap);
            free(tensor);
            return NULL;
        }
        tensor->shape[i] = shape[i];
        elements *= shape[i];
    }
    if (elements > (size_t)PTRDIFF_MAX) {
        free(tensor->metadata_heap);
        free(tensor);
        return NULL;
    }
    for (size_t dimension = nidims; dimension > 0U; --dimension) {
        size_t i = dimension - 1U;
        tensor->strides[i] = (ptrdiff_t)contiguous_stride;
        if (size_mul_overflows(contiguous_stride, tensor->shape[i])) {
            free(tensor->metadata_heap);
            free(tensor);
            return NULL;
        }
        contiguous_stride *= tensor->shape[i];
    }
    tensor->nidims = nidims;
    tensor->nelements = elements;
    return tensor;
}

Tensor *tensor_create_contiguous_view(TensorStorage *storage, size_t offset,
                                      const size_t *shape, size_t nidims) {
    Tensor *view;

    if (storage == NULL || storage->data == NULL || storage->nelements == 0U ||
        shape == NULL || nidims == 0U || offset >= storage->nelements) {
        return NULL;
    }
    view = tensor_allocate_metadata(shape, nidims);
    if (view == NULL) {
        return NULL;
    }
    view->storage = storage;
    view->offset = offset;
    if (tensor_validate(view) != SUCCESS || storage_retain(storage) != SUCCESS) {
        free(view->metadata_heap);
        free(view);
        return NULL;
    }
    return view;
}
Tensor *tensor_clone_view_metadata(const Tensor *tensor){
    Tensor *view;

    if (tensor_validate(tensor) != SUCCESS) {
        return NULL;
    }

    view = tensor_allocate_metadata(tensor->shape, tensor->nidims);
    if (view == NULL) {
        return NULL;
    }
    //no need to check for overflow here because we already validated the tensor
    memcpy_secure(
        view->strides,
        tensor->nidims * sizeof(*view->strides),
        tensor->strides,
        tensor->nidims * sizeof(*tensor->strides)
    );

    view->storage = tensor->storage;
    view->offset = tensor->offset;

    if (storage_retain(view->storage) != SUCCESS) {
        free(view->metadata_heap);
        free(view);
        return NULL;
    }

    return view;
}
static status_t tensor_relative_bounds(const Tensor *tensor, size_t *negative,
                                       size_t *positive) {
    size_t negative_bound = 0U;
    size_t positive_bound = 0U;

    for (size_t i = 0U; i < tensor->nidims; ++i) {
        size_t contribution;
        size_t magnitude;

        if (tensor->shape[i] == 0U) {
            return INVALID_SHAPE;
        }
        magnitude = ptrdiff_magnitude(tensor->strides[i]);
        if (size_mul_overflows(tensor->shape[i] - 1U, magnitude)) {
            return OVERFLOW;
        }
        contribution = (tensor->shape[i] - 1U) * magnitude;
        if (tensor->strides[i] < 0) {
            if (size_add_overflows(negative_bound, contribution)) {
                return OVERFLOW;
            }
            negative_bound += contribution;
        } else {
            if (size_add_overflows(positive_bound, contribution)) {
                return OVERFLOW;
            }
            positive_bound += contribution;
        }
    }
    if (negative_bound > (size_t)PTRDIFF_MAX || positive_bound > (size_t)PTRDIFF_MAX) {
        return OVERFLOW;
    }
    *negative = negative_bound;
    *positive = positive_bound;
    return SUCCESS;
}

status_t tensor_validate(const Tensor *tensor) {
    size_t elements = 1U;
    size_t negative_bound;
    size_t positive_bound;
    status_t status;

    if (tensor == NULL) {
        return NO_PARAMS;
    }
    if (tensor->storage == NULL || tensor->storage->data == NULL ||
        tensor->storage->nelements == 0U ||
        tensor->storage->nelements > (size_t)PTRDIFF_MAX ||
        tensor->storage->refcount == 0U || tensor->nidims == 0U ||
        tensor->shape == NULL || tensor->strides == NULL) {
        return INVALID_TENSOR;
    }
    if (tensor->metadata_heap == NULL) {
        if (tensor->shape != tensor->shape_inline || tensor->strides != tensor->strides_inline) {
            return INVALID_TENSOR;
        }
    } else {
        size_t stride_offset;
        size_t total_bytes;

        status = tensor_metadata_heap_layout(tensor->nidims, &stride_offset, &total_bytes);
        if (status != SUCCESS || tensor->shape != tensor->metadata_heap ||
            tensor->strides != (ptrdiff_t *)((unsigned char *)tensor->metadata_heap + stride_offset)) {
            return INVALID_TENSOR;
        }
    }
    for (size_t i = 0U; i < tensor->nidims; ++i) {
        if (tensor->shape[i] == 0U) {
            return INVALID_SHAPE;
        }
        if (size_mul_overflows(elements, tensor->shape[i])) {
            return OVERFLOW;
        }
        elements *= tensor->shape[i];
    }
    if (elements != tensor->nelements || elements > (size_t)PTRDIFF_MAX ||
        tensor->offset >= tensor->storage->nelements) {
        return INVALID_TENSOR;
    }
    status = tensor_relative_bounds(tensor, &negative_bound, &positive_bound);
    if (status != SUCCESS) {
        return status;
    }
    if (negative_bound > tensor->offset ||
        positive_bound > tensor->storage->nelements - 1U - tensor->offset) {
        return OUT_OF_BOUNDS;
    }
    return SUCCESS;
}

status_t tensor_validate_indices(const Tensor *tensor, const size_t *indices) {
    status_t status;

    if (indices == NULL) {
        return NO_PARAMS;
    }
    status = tensor_validate(tensor);
    if (status != SUCCESS) {
        return status;
    }
    for (size_t i = 0U; i < tensor->nidims; ++i) {
        if (indices[i] >= tensor->shape[i]) {
            return OUT_OF_BOUNDS;
        }
    }
    return SUCCESS;
}

static ptrdiff_t tensor_compute_relative_offset_validated(const Tensor *tensor,
                                                           const size_t *indices) {
    size_t negative = 0U;
    size_t positive = 0U;

    for (size_t i = 0U; i < tensor->nidims; ++i) {
        size_t contribution = indices[i] * ptrdiff_magnitude(tensor->strides[i]);
        if (tensor->strides[i] < 0) {
            negative += contribution;
        } else {
            positive += contribution;
        }
    }
    if (positive >= negative) {
        return (ptrdiff_t)(positive - negative);
    }
    return -(ptrdiff_t)(negative - positive);
}

static size_t tensor_absolute_offset_validated(const Tensor *tensor,
                                               ptrdiff_t relative_offset) {
    if (relative_offset >= 0) {
        return tensor->offset + (size_t)relative_offset;
    }
    return tensor->offset - ptrdiff_magnitude(relative_offset);
}

status_t tensor_compute_relative_offset(const Tensor *tensor, const size_t *indices,
                                        ptrdiff_t *relative_offset) {
    status_t status;

    if (relative_offset == NULL) {
        return NO_PARAMS;
    }
    status = tensor_validate_indices(tensor, indices);
    if (status != SUCCESS) {
        return status;
    }
    *relative_offset = tensor_compute_relative_offset_validated(tensor, indices);
    return SUCCESS;
}

bool tensor_is_contiguous(const Tensor *tensor) {
    ptrdiff_t expected_stride = 1;

    if (tensor_validate(tensor) != SUCCESS) {
        return false;
    }
    for (size_t dimension = tensor->nidims; dimension > 0U; --dimension) {
        size_t i = dimension - 1U;
        if (tensor->shape[i] != 1U && tensor->strides[i] != expected_stride) {
            return false;
        }
        expected_stride *= (ptrdiff_t)tensor->shape[i];
    }
    return true;
}

static Tensor *tensor_create_contiguous(const size_t *shape, size_t nidims,
                                        bool zero_initialize) {
    Tensor *tensor = tensor_allocate_metadata(shape, nidims);

    if (tensor == NULL) {
        return NULL;
    }
    tensor->storage = storage_create(tensor->nelements, zero_initialize);
    if (tensor->storage == NULL) {
        free(tensor->metadata_heap);
        free(tensor);
        return NULL;
    }
    return tensor;
}

Tensor *create_tensor(const size_t *shape, size_t nidims) {
    return tensor_create_contiguous(shape, nidims, true);
}

Tensor *tensor_create_uninitialized(const size_t *shape, size_t nidims) {
    return tensor_create_contiguous(shape, nidims, false);
}

static void tensor_fill_contiguous(float *restrict data, size_t nelements, float value) {
    for (size_t i = 0U; i < nelements; ++i) {
        data[i] = value;
    }
}

Tensor *init_tensor(const size_t *shape, size_t nidims, float init_val) {
    Tensor *tensor = tensor_create_contiguous(shape, nidims, init_val == 0.0f);

    if (tensor != NULL && init_val != 0.0f) {
        tensor_fill_contiguous(tensor->storage->data, tensor->nelements, init_val);
    }
    return tensor;
}

Tensor *tensor_from_data(const size_t *shape, size_t nidims, const float *data) {
    Tensor *tensor;
    size_t bytes;

    if (data == NULL) {
        return NULL;
    }
    tensor = tensor_create_contiguous(shape, nidims, false);
    if (tensor == NULL) {
        return NULL;
    }
    bytes = tensor->nelements * sizeof(*data);
    memcpy_secure(tensor->storage->data, tensor->nelements * sizeof(*tensor->storage->data), data, bytes);
    return tensor;
}

Tensor *tensor_create_view(TensorStorage *storage, size_t offset, const size_t *shape,
                           const ptrdiff_t *strides, size_t nidims) {
    Tensor *view;

    if (storage == NULL || storage->data == NULL || storage->nelements == 0U ||
        shape == NULL || strides == NULL || offset >= storage->nelements) {
        return NULL;
    }
    view = tensor_allocate_metadata(shape, nidims);
    if (view == NULL) {
        return NULL;
    }
    memcpy_secure(view->strides, nidims * sizeof(*view->strides), strides, nidims * sizeof(*strides));
    view->storage = storage;
    view->offset = offset;
    if (tensor_validate(view) != SUCCESS || storage_retain(storage) != SUCCESS) {
        free(view->metadata_heap);
        free(view);
        return NULL;
    }
    return view;
}

status_t tensor_get(const Tensor *tensor, const size_t *indices, float *value) {
    ptrdiff_t relative_offset;
    status_t status;

    if (value == NULL) {
        return NO_PARAMS;
    }
    status = tensor_compute_relative_offset(tensor, indices, &relative_offset);
    if (status != SUCCESS) {
        return status;
    }
    *value = tensor->storage->data[tensor_absolute_offset_validated(tensor, relative_offset)];
    return SUCCESS;
}

status_t tensor_set(Tensor *tensor, const size_t *indices, float value) {
    ptrdiff_t relative_offset;
    status_t status = tensor_compute_relative_offset(tensor, indices, &relative_offset);

    if (status != SUCCESS) {
        return status;
    }
    tensor->storage->data[tensor_absolute_offset_validated(tensor, relative_offset)] = value;
    return SUCCESS;
}

status_t deinit_tensor(Tensor *tensor) {
    TensorStorage *storage;
    status_t status = SUCCESS;

    if (tensor == NULL) {
        return NO_PARAMS;
    }
    storage = tensor->storage;
    tensor->storage = NULL;
    free(tensor->metadata_heap);
    tensor->nidims = 0U;
    tensor->nelements = 0U;
    tensor->offset = 0U;
    tensor_reset_metadata(tensor);
    if (storage != NULL) {
        status = storage_release(storage);
    }
    return status;
}

status_t destroy_tensor(Tensor *tensor) {
    status_t status;

    if (tensor == NULL) {
        return NO_PARAMS;
    }
    status = deinit_tensor(tensor);
    free(tensor);
    return status;
}

status_t print_tensor(const Tensor *tensor) {
    size_t *indices;
    status_t status = tensor_validate(tensor);

    if (status != SUCCESS) {
        return status;
    }
    indices = calloc(tensor->nidims, sizeof(*indices));
    if (indices == NULL) {
        return ALLOCATION_FAILURE;
    }
    for (size_t linear = 0U; linear < tensor->nelements; ++linear) {
        ptrdiff_t relative_offset = tensor_compute_relative_offset_validated(tensor, indices);
        printf("%f ", tensor->storage->data[
            tensor_absolute_offset_validated(tensor, relative_offset)]);
        for (size_t dimension = tensor->nidims; dimension > 0U; --dimension) {
            size_t i = dimension - 1U;
            if (++indices[i] < tensor->shape[i]) {
                break;
            }
            indices[i] = 0U;
        }
    }
    free(indices);
    printf("\n");
    return SUCCESS;
}
