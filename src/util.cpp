#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sep_cuda.h"
#include "sepcore.h"

#define DETAILSIZE 512

static thread_local char errdetail_buffer[DETAILSIZE] = "";

PIXTYPE convert_dbl(const void *ptr) {
  return *(const double *)ptr;
}

PIXTYPE convert_flt(const void *ptr) {
  return *(const float *)ptr;
}

PIXTYPE convert_int(const void *ptr) {
  return *(const int *)ptr;
}

PIXTYPE convert_byt(const void *ptr) {
  return *(const BYTE *)ptr;
}

PIXTYPE convert_ushort(const void *ptr) {
  return *(const uint16_t *)ptr;
}

int get_converter(int dtype, converter *f, int64_t *size) {
  int status = RETURN_OK;

  if (dtype == SEP_TFLOAT) {
    *f = convert_flt;
    *size = sizeof(float);
  } else if (dtype == SEP_TINT) {
    *f = convert_int;
    *size = sizeof(int);
  } else if (dtype == SEP_TDOUBLE) {
    *f = convert_dbl;
    *size = sizeof(double);
  } else if (dtype == SEP_TBYTE) {
    *f = convert_byt;
    *size = sizeof(BYTE);
  } else if (dtype == SEP_TUSHORT) {
    *f = convert_ushort;
    *size = sizeof(uint16_t);
  } else {
    *f = NULL;
    *size = 0;
    status = ILLEGAL_DTYPE;
  }

  return status;
}

void convert_array_flt(const void *ptr, int64_t n, PIXTYPE *target) {
  const float *source = (const float *)ptr;
  int64_t i;
  for (i = 0; i < n; i++) {
    target[i] = source[i];
  }
}

void convert_array_dbl(const void *ptr, int64_t n, PIXTYPE *target) {
  const double *source = (const double *)ptr;
  int64_t i;
  for (i = 0; i < n; i++) {
    target[i] = (float)source[i];
  }
}

void convert_array_int(const void *ptr, int64_t n, PIXTYPE *target) {
  const int *source = (const int *)ptr;
  int64_t i;
  for (i = 0; i < n; i++) {
    target[i] = (float)source[i];
  }
}

void convert_array_byt(const void *ptr, int64_t n, PIXTYPE *target) {
  const BYTE *source = (const BYTE *)ptr;
  int64_t i;
  for (i = 0; i < n; i++) {
    target[i] = (float)source[i];
  }
}

void convert_array_ushort(const void *ptr, int64_t n, PIXTYPE *target) {
  const uint16_t *source = (const uint16_t *)ptr;
  int64_t i;
  for (i = 0; i < n; i++) {
    target[i] = (float)source[i];
  }
}

int get_array_converter(int dtype, array_converter *f, int64_t *size) {
  int status = RETURN_OK;

  if (dtype == SEP_TFLOAT) {
    *f = convert_array_flt;
    *size = sizeof(float);
  } else if (dtype == SEP_TINT) {
    *f = convert_array_int;
    *size = sizeof(int);
  } else if (dtype == SEP_TDOUBLE) {
    *f = convert_array_dbl;
    *size = sizeof(double);
  } else if (dtype == SEP_TBYTE) {
    *f = convert_array_byt;
    *size = sizeof(BYTE);
  } else if (dtype == SEP_TUSHORT) {
    *f = convert_array_ushort;
    *size = sizeof(uint16_t);
  } else {
    *f = NULL;
    *size = 0;
    status = ILLEGAL_DTYPE;
  }

  return status;
}

void write_array_dbl(const float *ptr, int64_t n, void *target) {
  double *out = (double *)target;
  int64_t i;
  for (i = 0; i < n; i++) {
    out[i] = (double)ptr[i];
  }
}

void write_array_int(const float *ptr, int64_t n, void *target) {
  int *out = (int *)target;
  int64_t i;
  for (i = 0; i < n; i++) {
    out[i] = (int)(ptr[i] + 0.5f);
  }
}

void write_array_ushort(const float *ptr, int64_t n, void *target) {
  uint16_t *out = (uint16_t *)target;
  int64_t i;
  for (i = 0; i < n; i++) {
    int value = (int)(ptr[i] + 0.5f);
    if (value < 0) {
      value = 0;
    } else if (value > 65535) {
      value = 65535;
    }
    out[i] = (uint16_t)value;
  }
}

int get_array_writer(int dtype, array_writer *f, int64_t *size) {
  int status = RETURN_OK;

  if (dtype == SEP_TINT) {
    *f = write_array_int;
    *size = sizeof(int);
  } else if (dtype == SEP_TUSHORT) {
    *f = write_array_ushort;
    *size = sizeof(uint16_t);
  } else if (dtype == SEP_TDOUBLE) {
    *f = write_array_dbl;
    *size = sizeof(double);
  } else {
    *f = NULL;
    *size = 0;
    status = ILLEGAL_DTYPE;
  }

  return status;
}

void subtract_array_dbl(const float *ptr, int64_t n, void *target) {
  double *out = (double *)target;
  int64_t i;
  for (i = 0; i < n; i++) {
    out[i] -= (double)ptr[i];
  }
}

void subtract_array_flt(const float *ptr, int64_t n, void *target) {
  float *out = (float *)target;
  int64_t i;
  for (i = 0; i < n; i++) {
    out[i] -= ptr[i];
  }
}

void subtract_array_int(const float *ptr, int64_t n, void *target) {
  int *out = (int *)target;
  int64_t i;
  for (i = 0; i < n; i++) {
    out[i] -= (int)(ptr[i] + 0.5f);
  }
}

void subtract_array_ushort(const float *ptr, int64_t n, void *target) {
  uint16_t *out = (uint16_t *)target;
  int64_t i;
  for (i = 0; i < n; i++) {
    int value = (int)out[i] - (int)(ptr[i] + 0.5f);
    if (value < 0) {
      value = 0;
    } else if (value > 65535) {
      value = 65535;
    }
    out[i] = (uint16_t)value;
  }
}

int get_array_subtractor(int dtype, array_writer *f, int64_t *size) {
  int status = RETURN_OK;

  if (dtype == SEP_TFLOAT) {
    *f = subtract_array_flt;
    *size = sizeof(float);
  } else if (dtype == SEP_TINT) {
    *f = subtract_array_int;
    *size = sizeof(int);
  } else if (dtype == SEP_TUSHORT) {
    *f = subtract_array_ushort;
    *size = sizeof(uint16_t);
  } else if (dtype == SEP_TDOUBLE) {
    *f = subtract_array_dbl;
    *size = sizeof(double);
  } else {
    *f = NULL;
    *size = 0;
    status = ILLEGAL_DTYPE;
  }

  return status;
}

extern "C" SEP_API void sep_get_errmsg(int status, char *errtext) {
  errtext[0] = '\0';

  switch (status) {
    case RETURN_OK:
      strcpy(errtext, "OK - no error");
      break;
    case MEMORY_ALLOC_ERROR:
      strcpy(errtext, "memory allocation");
      break;
    case ILLEGAL_DTYPE:
      strcpy(errtext, "dtype not recognized/unsupported");
      break;
    case ILLEGAL_SUBPIX:
      strcpy(errtext, "subpix value must be nonnegative");
      break;
    case NON_ELLIPSE_PARAMS:
      strcpy(errtext, "parameters do not describe ellipse");
      break;
    case ILLEGAL_APER_PARAMS:
      strcpy(errtext, "invalid parameters");
      break;
    case LINE_NOT_IN_BUF:
      strcpy(errtext, "array line out of buffer");
      break;
    case RELTHRESH_NO_NOISE:
      strcpy(errtext, "relative threshold but image has noise_type of NONE");
      break;
    case UNKNOWN_NOISE_TYPE:
      strcpy(errtext, "image has unknown noise_type");
      break;
    case SEP_CUDA_UNAVAILABLE:
      strcpy(errtext, "cuda unavailable");
      break;
    case SEP_CUDA_RUNTIME_ERROR:
      strcpy(errtext, "cuda runtime error");
      break;
    default:
      strcpy(errtext, "unknown error status");
      break;
  }
}

extern "C" SEP_API void sep_get_errdetail(char *errtext) {
  strcpy(errtext, errdetail_buffer);
  memset(errdetail_buffer, 0, DETAILSIZE);
}

void put_errdetail(const char *errtext) {
  strncpy(errdetail_buffer, errtext, DETAILSIZE - 1);
  errdetail_buffer[DETAILSIZE - 1] = '\0';
}

static int fqcmp(const void *p1, const void *p2) {
  const double f1 = *((const float *)p1);
  const double f2 = *((const float *)p2);
  return f1 > f2 ? 1 : (f1 < f2 ? -1 : 0);
}

float fqmedian(float *ra, int64_t n) {
  qsort(ra, (size_t)n, sizeof(float), fqcmp);
  if (n < 2) {
    return *ra;
  }
  return (n & 1) ? ra[n / 2] : (ra[n / 2 - 1] + ra[n / 2]) / 2.0f;
}
