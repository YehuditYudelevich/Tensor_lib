#ifndef UTILS_H
#define UTILS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>



bool size_add_overflows(size_t left, size_t right);
bool size_mul_overflows(size_t left, size_t right);
void memcpy_secure(void *dest, size_t dest_len, const void *src, size_t src_len);
void fatal_if(bool condition);

#endif // UTILS_H