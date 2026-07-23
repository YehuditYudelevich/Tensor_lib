#ifndef TENSOR_H
#define TENSOR_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "error_handle.h"
#include "utils.h"

#define TENSOR_INLINE_DIMS 6U
#define TENSOR_ALIGNMENT 64U

typedef struct TensorStorage {
    float *data;
    size_t nelements;
    size_t refcount;
} TensorStorage;

typedef struct Tensor {
    TensorStorage *storage;
    size_t *shape;
    ptrdiff_t *strides;
    size_t nidims;
    size_t nelements;
    size_t offset;

    size_t shape_inline[TENSOR_INLINE_DIMS];
    ptrdiff_t strides_inline[TENSOR_INLINE_DIMS];
    void *metadata_heap;
} Tensor;

/* Allocate and release 64-byte-aligned memory used by tensor storage. */
void *tensor_aligned_alloc(size_t alignment, size_t size);
void tensor_aligned_free(void *ptr);

/* Low-level non-thread-safe shared-buffer API. A new storage has refcount 1. */
TensorStorage *storage_create(size_t nelements, bool zero_initialize);
status_t storage_retain(TensorStorage *storage);
status_t storage_release(TensorStorage *storage);

/* Create a zero-initialized, contiguous C-order tensor. */
Tensor *create_tensor(const size_t *shape, size_t nidims);

/* Create a contiguous tensor whose elements are intentionally uninitialized. */
Tensor *tensor_create_uninitialized(const size_t *shape, size_t nidims);

/* Create a contiguous tensor and fill every element with init_val. */
Tensor *init_tensor(const size_t *shape, size_t nidims, float init_val);

/* Copy external contiguous data into newly owned tensor storage. */
Tensor *tensor_from_data(const size_t *shape, size_t nidims, const float *data);

/* Create a metadata-only view and retain storage; strides are in float elements. */
Tensor *tensor_create_view(TensorStorage *storage, size_t offset,
                           const size_t *shape, const ptrdiff_t *strides,
                           size_t nidims);

/* Release tensor-owned resources while leaving the Tensor object allocated. */
status_t deinit_tensor(Tensor *tensor);

/* Release all tensor resources, including the Tensor object itself. */
status_t destroy_tensor(Tensor *tensor);

/* Validate metadata, ownership, and both reachable strided storage bounds. */
status_t tensor_validate(const Tensor *tensor);

/* Validate one complete logical index tuple for tensor. */
status_t tensor_validate_indices(const Tensor *tensor, const size_t *indices);

/* Safely compute the signed storage-relative offset for a logical index tuple. */
status_t tensor_compute_relative_offset(const Tensor *tensor,
                                        const size_t *indices,
                                        ptrdiff_t *relative_offset);

/* Read or write one logical element after validating the supplied indices. */
status_t tensor_get(const Tensor *tensor, const size_t *indices, float *value);
status_t tensor_set(Tensor *tensor, const size_t *indices, float value);

/* Return true when the logical layout is contiguous in row-major (C) order. */
bool tensor_is_contiguous(const Tensor *tensor);

/* Print logical elements in row-major index order; intended for debugging only. */
status_t print_tensor(const Tensor *tensor);

#endif
