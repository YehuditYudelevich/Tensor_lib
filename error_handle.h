#ifndef ERROR_HANDLE_H
#define ERROR_HANDLE_H

typedef enum {
    SUCCESS = 0,
    NO_PARAMS,
    ALLOCATION_FAILURE,
    INVALID_SHAPE,
    OUT_OF_BOUNDS,
    OVERFLOW,
    INVALID_TENSOR,
    FAILURE
} status_t;

#endif