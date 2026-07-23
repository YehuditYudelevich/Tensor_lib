#include "utils.h"


bool size_add_overflows(size_t left, size_t right) {
    return left > SIZE_MAX - right;
}

bool size_mul_overflows(size_t left, size_t right) {
    return left != 0U && right > SIZE_MAX / left;
}
void memcpy_secure(void *dest, size_t dest_len, const void *src, size_t src_len) {
    if (dest == NULL || src == NULL) {
        return;
    }
    fatal_if(dest_len < src_len);
    memcpy(dest, src, src_len);
}

void fatal_if(bool condition) {
    if (condition) {
        fprintf(stderr, "Fatal error: condition failed\n");
        exit(EXIT_FAILURE);
    }
}