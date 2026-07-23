#include "tensor.h"
#include "tensor_broadcast.h"
#include "tensor_view.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>

static void print_test_passed(const char *test_name) {
    printf("[PASS] %s\n", test_name);
}

static void test_create_destroy_and_inline_metadata(void) {
    const size_t shape[] = {2U, 3U};
    Tensor *tensor = create_tensor(shape, 2U);

    assert(tensor != NULL);
    assert(tensor_validate(tensor) == SUCCESS);
    assert(tensor->shape == tensor->shape_inline);
    assert(tensor->strides == tensor->strides_inline);
    assert(tensor->metadata_heap == NULL);
    assert(tensor->nelements == 6U);
    assert(tensor->strides[0] == 3 && tensor->strides[1] == 1);
    assert(((uintptr_t)tensor->storage->data % TENSOR_ALIGNMENT) == 0U);
    for (size_t i = 0U; i < tensor->nelements; ++i) {
        assert(tensor->storage->data[i] == 0.0f);
    }
    assert(tensor_is_contiguous(tensor));
    assert(destroy_tensor(tensor) == SUCCESS);
    print_test_passed("create/destroy uses typed inline metadata and aligned storage");
}

static void test_init_and_from_data(void) {
    const size_t shape[] = {2U, 2U};
    const float source[] = {1.0f, 2.0f, 3.0f, 4.0f};
    const size_t index[] = {1U, 0U};
    Tensor *filled = init_tensor(shape, 2U, 7.5f);
    Tensor *copied = tensor_from_data(shape, 2U, source);
    float value = 0.0f;

    assert(filled != NULL && copied != NULL);
    for (size_t i = 0U; i < filled->nelements; ++i) {
        assert(filled->storage->data[i] == 7.5f);
    }
    assert(tensor_get(copied, index, &value) == SUCCESS);
    assert(value == 3.0f);
    assert(destroy_tensor(filled) == SUCCESS);
    assert(destroy_tensor(copied) == SUCCESS);
    print_test_passed("init and from_data");
}

static void test_get_set_for_1d_2d_and_3d(void) {
    const size_t shape_1d[] = {4U};
    const size_t shape_2d[] = {2U, 3U};
    const size_t shape_3d[] = {2U, 3U, 4U};
    const size_t index_1d[] = {3U};
    const size_t index_2d[] = {1U, 2U};
    const size_t index_3d[] = {1U, 2U, 3U};
    Tensor *one = create_tensor(shape_1d, 1U);
    Tensor *two = create_tensor(shape_2d, 2U);
    Tensor *three = create_tensor(shape_3d, 3U);
    float value = 0.0f;
    ptrdiff_t offset = 0;

    assert(one != NULL && two != NULL && three != NULL);
    assert(tensor_set(one, index_1d, 1.0f) == SUCCESS);
    assert(tensor_set(two, index_2d, 2.0f) == SUCCESS);
    assert(tensor_set(three, index_3d, 3.0f) == SUCCESS);
    assert(tensor_compute_relative_offset(three, index_3d, &offset) == SUCCESS);
    assert(offset == 23);
    assert(tensor_get(one, index_1d, &value) == SUCCESS && value == 1.0f);
    assert(tensor_get(two, index_2d, &value) == SUCCESS && value == 2.0f);
    assert(tensor_get(three, index_3d, &value) == SUCCESS && value == 3.0f);
    assert(destroy_tensor(one) == SUCCESS);
    assert(destroy_tensor(two) == SUCCESS);
    assert(destroy_tensor(three) == SUCCESS);
    print_test_passed("1D, 2D, and 3D get/set");
}

static void test_out_of_bounds_and_invalid_shapes(void) {
    const size_t shape[] = {2U, 2U};
    const size_t invalid_shape[] = {0U, 2U};
    const size_t bad_index[] = {2U, 0U};
    const size_t overflowing_shape[] = {SIZE_MAX, 2U};
    Tensor *tensor = create_tensor(shape, 2U);
    float value;

    assert(tensor != NULL);
    assert(tensor_get(tensor, bad_index, &value) == OUT_OF_BOUNDS);
    assert(tensor_set(tensor, bad_index, 1.0f) == OUT_OF_BOUNDS);
    assert(create_tensor(invalid_shape, 2U) == NULL);
    assert(create_tensor(overflowing_shape, 2U) == NULL);
    assert(create_tensor(shape, SIZE_MAX) == NULL);
    assert(destroy_tensor(tensor) == SUCCESS);
    print_test_passed("out-of-bounds and overflow inputs");
}

static void test_positive_and_negative_stride_views(void) {
    const size_t base_shape[] = {2U, 3U};
    const size_t row_shape[] = {3U};
    const ptrdiff_t row_strides[] = {1};
    const size_t transpose_shape[] = {3U, 2U};
    const ptrdiff_t transpose_strides[] = {1, 3};
    const size_t reverse_shape[] = {6U};
    const ptrdiff_t reverse_strides[] = {-1};
    const size_t invalid_strided_shape[] = {2U, 2U};
    const ptrdiff_t invalid_strided_strides[] = {6, 1};
    const size_t row_index[] = {1U};
    const size_t transpose_index[] = {2U, 1U};
    const size_t reverse_last[] = {5U};
    Tensor *base = tensor_from_data(base_shape, 2U,
                                    (const float[]){0.0f, 1.0f, 2.0f,
                                                    3.0f, 4.0f, 5.0f});
    Tensor *row;
    Tensor *transpose;
    Tensor *reverse;
    TensorStorage *storage;
    float value = 0.0f;
    ptrdiff_t relative_offset = 0;

    assert(base != NULL);
    storage = base->storage;
    assert(storage->refcount == 1U);
    row = tensor_create_view(storage, 3U, row_shape, row_strides, 1U);
    assert(row != NULL);
    assert(storage->refcount == 2U);
    assert(tensor_is_contiguous(row));
    assert(tensor_get(row, row_index, &value) == SUCCESS && value == 4.0f);
    assert(tensor_set(row, row_index, 40.0f) == SUCCESS);
    assert(base->storage->data[4] == 40.0f);

    transpose = tensor_create_view(storage, 0U, transpose_shape, transpose_strides, 2U);
    assert(transpose != NULL);
    assert(storage->refcount == 3U);
    assert(!tensor_is_contiguous(transpose));
    assert(tensor_get(transpose, transpose_index, &value) == SUCCESS && value == 5.0f);

    reverse = tensor_create_view(storage, 5U, reverse_shape, reverse_strides, 1U);
    assert(reverse != NULL);
    assert(storage->refcount == 4U);
    assert(!tensor_is_contiguous(reverse));
    assert(tensor_compute_relative_offset(reverse, reverse_last, &relative_offset) == SUCCESS);
    assert(relative_offset == -5);
    assert(tensor_get(reverse, reverse_last, &value) == SUCCESS && value == 0.0f);
    assert(tensor_set(reverse, (const size_t[]){1U}, 44.0f) == SUCCESS);
    assert(base->storage->data[4] == 44.0f);

    assert(tensor_create_view(storage, 5U, row_shape, row_strides, 1U) == NULL);
    assert(tensor_create_view(storage, 0U, invalid_strided_shape,
                              invalid_strided_strides, 2U) == NULL);
    assert(destroy_tensor(row) == SUCCESS);
    assert(storage->refcount == 3U);
    assert(destroy_tensor(transpose) == SUCCESS);
    assert(storage->refcount == 2U);
    assert(destroy_tensor(reverse) == SUCCESS);
    assert(storage->refcount == 1U);
    assert(destroy_tensor(base) == SUCCESS);
    print_test_passed("positive and negative stride views share storage safely");
}

static void test_dynamic_metadata(void) {
    const size_t shape[] = {1U, 1U, 1U, 1U, 1U, 1U, 2U};
    Tensor *tensor = create_tensor(shape, 7U);

    assert(tensor != NULL);
    assert(tensor->metadata_heap != NULL);
    assert(tensor->shape != tensor->shape_inline);
    assert(tensor->strides != tensor->strides_inline);
    assert(tensor->nelements == 2U);
    assert(tensor_is_contiguous(tensor));
    assert(destroy_tensor(tensor) == SUCCESS);
    print_test_passed("dynamic metadata for large ndim");
}

static void test_reshape_creates_contiguous_zero_copy_view(void) {
    const size_t source_shape[] = {2U, 3U, 4U};
    const size_t reshaped_shape[] = {4U, 6U};
    const size_t source_index[] = {1U, 2U, 3U};
    const size_t reshaped_index[] = {3U, 5U};
    Tensor *source = create_tensor(source_shape, 3U);
    Tensor *reshaped;
    float value = 0.0f;

    assert(source != NULL);
    assert(tensor_set(source, source_index, 99.0f) == SUCCESS);
    reshaped = tensor_reshape(source, reshaped_shape, 2U);
    assert(reshaped != NULL);
    assert(reshaped->storage == source->storage);
    assert(source->storage->refcount == 2U);
    assert(reshaped->offset == source->offset);
    assert(tensor_is_contiguous(reshaped));
    assert(tensor_get(reshaped, reshaped_index, &value) == SUCCESS);
    assert(value == 99.0f);
    assert(tensor_set(reshaped, (const size_t[]){0U, 0U}, 7.0f) == SUCCESS);
    assert(source->storage->data[0] == 7.0f);
    assert(tensor_reshape(source, (const size_t[]){5U, 5U}, 2U) == NULL);
    assert(destroy_tensor(reshaped) == SUCCESS);
    assert(source->storage->refcount == 1U);
    assert(destroy_tensor(source) == SUCCESS);
    print_test_passed("reshape creates a contiguous zero-copy view");
}

static void test_slice_creates_strided_zero_copy_view(void) {
    const size_t shape[] = {2U, 5U};
    const size_t selected_index[] = {1U, 1U};
    Tensor *source = tensor_from_data(shape, 2U,
                                      (const float[]){0.0f, 1.0f, 2.0f, 3.0f, 4.0f,
                                                      5.0f, 6.0f, 7.0f, 8.0f, 9.0f});
    Tensor *slice;
    float value = 0.0f;

    assert(source != NULL);
    slice = tensor_slice(source, 1U, 1U, 5U, 2U);
    assert(slice != NULL);
    assert(slice->storage == source->storage);
    assert(slice->offset == 1U);
    assert(slice->shape[0] == 2U && slice->shape[1] == 2U);
    assert(slice->strides[0] == 5 && slice->strides[1] == 2);
    assert(tensor_get(slice, selected_index, &value) == SUCCESS && value == 8.0f);
    assert(tensor_set(slice, (const size_t[]){0U, 1U}, 30.0f) == SUCCESS);
    assert(source->storage->data[3] == 30.0f);
    assert(tensor_slice(source, 1U, 3U, 3U, 1U) == NULL);
    assert(destroy_tensor(slice) == SUCCESS);
    assert(destroy_tensor(source) == SUCCESS);
    print_test_passed("slice creates a strided zero-copy view");
}

static void test_broadcast_shape_and_strides(void) {
    const size_t matrix_shape[] = {2U, 3U};
    const size_t vector_shape[] = {3U};
    const size_t incompatible_shape[] = {4U, 3U};
    const size_t expected_shape[] = {2U, 3U};
    const size_t larger_output_shape[] = {4U, 2U, 3U};
    const size_t zero_output_shape[] = {0U, 3U};
    size_t output_shape[2];
    ptrdiff_t matrix_strides[2];
    ptrdiff_t vector_strides[2];
    ptrdiff_t expanded_vector_strides[3];
    Tensor *matrix = create_tensor(matrix_shape, 2U);
    Tensor *vector = create_tensor(vector_shape, 1U);
    Tensor *incompatible = create_tensor(incompatible_shape, 2U);

    assert(matrix != NULL && vector != NULL && incompatible != NULL);
    assert(tensor_broadcast_shape(matrix, vector, output_shape, 2U) == SUCCESS);
    assert(output_shape[0] == expected_shape[0] && output_shape[1] == expected_shape[1]);
    assert(tensor_broadcast_shape(matrix, incompatible, output_shape, 2U) == INVALID_SHAPE);

    assert(tensor_compute_broadcast_strides(matrix, expected_shape, 2U,
                                            matrix_strides) == SUCCESS);
    assert(matrix_strides[0] == 3 && matrix_strides[1] == 1);
    assert(tensor_compute_broadcast_strides(vector, expected_shape, 2U,
                                            vector_strides) == SUCCESS);
    assert(vector_strides[0] == 0 && vector_strides[1] == 1);
    assert(tensor_compute_broadcast_strides(vector, larger_output_shape, 3U,
                                            expanded_vector_strides) == SUCCESS);
    assert(expanded_vector_strides[0] == 0 && expanded_vector_strides[1] == 0 &&
           expanded_vector_strides[2] == 1);
    assert(tensor_compute_broadcast_strides(vector, zero_output_shape, 2U,
                                            vector_strides) == INVALID_SHAPE);

    assert(destroy_tensor(incompatible) == SUCCESS);
    assert(destroy_tensor(vector) == SUCCESS);
    assert(destroy_tensor(matrix) == SUCCESS);
    print_test_passed("broadcast shape and strides");
}

static double elapsed_milliseconds(clock_t start, clock_t end) {
    return ((double)(end - start) * 1000.0) / (double)CLOCKS_PER_SEC;
}

static void print_tensor_summary(const char *name, const Tensor *tensor) {
    printf("%s: shape=[", name);
    for (size_t i = 0U; i < tensor->nidims; ++i) {
        printf("%zu%s", tensor->shape[i], i + 1U == tensor->nidims ? "" : ", ");
    }
    printf("], strides=[");
    for (size_t i = 0U; i < tensor->nidims; ++i) {
        printf("%td%s", tensor->strides[i], i + 1U == tensor->nidims ? "" : ", ");
    }
    printf("], offset=%zu, elements=%zu\n", tensor->offset, tensor->nelements);
    assert(print_tensor(tensor) == SUCCESS);
}

static void demo_live_tensor_workflow(void) {
    const size_t shape[] = {3U, 4U};
    const size_t first_index[] = {0U, 0U};
    const size_t middle_index[] = {1U, 2U};
    const size_t last_index[] = {2U, 3U};
    const size_t reverse_shape[] = {12U};
    const ptrdiff_t reverse_stride[] = {-1};
    const size_t benchmark_shape[] = {1024U, 1024U};
    Tensor *tensor;
    Tensor *reverse;
    Tensor *benchmark;
    float selected_value = 0.0f;
    clock_t start;
    clock_t end;

    printf("\n=== Live tensor runtime demonstration ===\n");
    start = clock();
    tensor = create_tensor(shape, 2U);
    end = clock();
    assert(tensor != NULL);
    printf("Created a zero-initialized 3x4 tensor in %.3f ms\n",
           elapsed_milliseconds(start, end));
    print_tensor_summary("Initial tensor", tensor);

    printf("Writing selected values: [0,0]=10.0, [1,2]=-3.5, [2,3]=42.0\n");
    assert(tensor_set(tensor, first_index, 10.0f) == SUCCESS);
    assert(tensor_set(tensor, middle_index, -3.5f) == SUCCESS);
    assert(tensor_set(tensor, last_index, 42.0f) == SUCCESS);
    print_tensor_summary("After tensor_set", tensor);

    assert(tensor_get(tensor, middle_index, &selected_value) == SUCCESS);
    printf("tensor_get([1,2]) returned %.1f\n", selected_value);

    reverse = tensor_create_view(tensor->storage, tensor->nelements - 1U,
                                 reverse_shape, reverse_stride, 1U);
    assert(reverse != NULL);
    printf("Reverse view: shares storage, uses stride -1, and does not copy data\n");
    print_tensor_summary("Reverse view", reverse);

    start = clock();
    benchmark = init_tensor(benchmark_shape, 2U, 1.0f);
    end = clock();
    assert(benchmark != NULL);
    printf("Initialized 1,048,576 float32 elements in %.3f ms\n",
           elapsed_milliseconds(start, end));

    assert(destroy_tensor(benchmark) == SUCCESS);
    assert(destroy_tensor(reverse) == SUCCESS);
    assert(destroy_tensor(tensor) == SUCCESS);
    printf("=== End live demonstration ===\n\n");
}

int main(void) {
    test_create_destroy_and_inline_metadata();
    test_init_and_from_data();
    test_get_set_for_1d_2d_and_3d();
    test_out_of_bounds_and_invalid_shapes();
    test_positive_and_negative_stride_views();
    test_dynamic_metadata();
    test_reshape_creates_contiguous_zero_copy_view();
    test_slice_creates_strided_zero_copy_view();
    test_broadcast_shape_and_strides();
    demo_live_tensor_workflow();
    printf("All tests passed\n");
    return 0;
}
