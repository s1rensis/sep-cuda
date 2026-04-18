#ifndef SEPCUDA_SEPCORE_H
#define SEPCUDA_SEPCORE_H

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "sep_cuda.h"

#define BIG 1e+30
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#define PI M_PI

typedef unsigned char BYTE;
typedef float PIXTYPE;
#define PIXDTYPE SEP_TFLOAT

typedef PIXTYPE (*converter)(const void *ptr);
typedef void (*array_converter)(const void *ptr, int64_t n, PIXTYPE *target);
typedef void (*array_writer)(const float *ptr, int64_t n, void *target);

#define QCALLOC(ptr, typ, nel, status)                        \
  {                                                           \
    if (!(ptr = (typ *)calloc((size_t)(nel), sizeof(typ)))) { \
      char errtext[160];                                      \
      snprintf(                                               \
          errtext,                                            \
          sizeof(errtext),                                    \
          #ptr " allocation failed at line %d in %s",         \
          __LINE__,                                           \
          __FILE__);                                          \
      put_errdetail(errtext);                                 \
      status = MEMORY_ALLOC_ERROR;                            \
      goto exit;                                              \
    }                                                         \
  }

#define QMALLOC(ptr, typ, nel, status)                     \
  {                                                        \
    if (!(ptr = (typ *)malloc((size_t)(nel) * sizeof(typ)))) { \
      char errtext[160];                                   \
      snprintf(                                            \
          errtext,                                         \
          sizeof(errtext),                                 \
          #ptr " allocation failed at line %d in %s",      \
          __LINE__,                                        \
          __FILE__);                                       \
      put_errdetail(errtext);                              \
      status = MEMORY_ALLOC_ERROR;                         \
      goto exit;                                           \
    }                                                      \
  }

float fqmedian(float *ra, int64_t n);
void put_errdetail(const char *errtext);

int get_converter(int dtype, converter *f, int64_t *size);
int get_array_converter(int dtype, array_converter *f, int64_t *size);
int get_array_writer(int dtype, array_writer *f, int64_t *size);
int get_array_subtractor(int dtype, array_writer *f, int64_t *size);

#endif
