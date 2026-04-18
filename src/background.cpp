#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <chrono>

#include "sep_cuda.h"
#include "sep_internal.h"
#include "sepcore.h"

#define BACK_MINGOODFRAC 0.5
#define QUANTIF_NSIGMA 5
#define QUANTIF_NMAXLEVELS 4096
#define QUANTIF_AMIN 4

typedef struct {
  float mode, mean, sigma;
  int64_t *histo;
  int nlevels;
  float qzero, qscale;
  float lcut, hcut;
  int64_t npix;
} backstruct;

static int filterback(sep_bkg *bkg, int64_t fw, int64_t fh, double fthresh);
static int makebackspline(const sep_bkg *bkg, float *map, float *dmap);
static int bkg_line_flt_internal(
    const sep_bkg *bkg, float *values, float *dvalues, int64_t y, float *line);
static int sep_bkg_line_flt(const sep_bkg *bkg, int64_t y, float *line);
static int sep_bkg_rmsline_flt(const sep_bkg *bkg, int64_t y, float *line);

static double now_ms(void) {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

static int convert_to_float_buffer(
    const void *source, int dtype, int64_t count, float **out_buf) {
  int status;
  int64_t elsize;
  array_converter convert;

  status = get_array_converter(dtype, &convert, &elsize);
  if (status != RETURN_OK) {
    return status;
  }

  *out_buf = NULL;
  if (count == 0) {
    return RETURN_OK;
  }

  *out_buf = (float *)malloc((size_t)count * sizeof(float));
  if (*out_buf == NULL) {
    put_errdetail("failed to allocate float staging buffer");
    return MEMORY_ALLOC_ERROR;
  }

  convert(source, count, *out_buf);
  return RETURN_OK;
}

extern "C" SEP_API int sep_background(
    const sep_image *image,
    int64_t bw,
    int64_t bh,
    int64_t fw,
    int64_t fh,
    double fthresh,
    sep_bkg **bkg) {
  int status;
  int64_t nx, ny, nb, npix;
  float *imgbuf, *maskbuf;
  float maskthresh;
  sep_bkg *bkgout;
  sep_cuda_background_profile profile;
  double total_start_ms, phase_start_ms;
  const int profile_enabled = sepcuda_profile_enabled();

  status = RETURN_OK;
  imgbuf = NULL;
  maskbuf = NULL;
  bkgout = NULL;
  sepcuda_profile_reset_background(&profile);
  total_start_ms = profile_enabled ? now_ms() : 0.0;

  if (bkg == NULL || image == NULL || image->data == NULL) {
    put_errdetail("sep_background received a null pointer");
    if (profile_enabled) {
      profile.total_background_ms = now_ms() - total_start_ms;
      sepcuda_profile_commit_background(&profile);
    }
    return ILLEGAL_APER_PARAMS;
  }

  *bkg = NULL;

  if (image->w <= 0 || image->h <= 0 || bw <= 0 || bh <= 0 || fw <= 0 || fh <= 0) {
    put_errdetail("sep_background received non-positive dimensions");
    if (profile_enabled) {
      profile.total_background_ms = now_ms() - total_start_ms;
      sepcuda_profile_commit_background(&profile);
    }
    return ILLEGAL_APER_PARAMS;
  }

  npix = image->w * image->h;
  maskthresh = image->mask ? (float)image->maskthresh : 0.0f;

  phase_start_ms = profile_enabled ? now_ms() : 0.0;
  status = convert_to_float_buffer(image->data, image->dtype, npix, &imgbuf);
  if (status != RETURN_OK) {
    goto exit;
  }

  if (image->mask != NULL) {
    status = convert_to_float_buffer(image->mask, image->mdtype, npix, &maskbuf);
    if (status != RETURN_OK) {
      goto exit;
    }
  }
  if (profile_enabled) {
    profile.staging_ms = now_ms() - phase_start_ms;
  }

  nx = (image->w - 1) / bw + 1;
  ny = (image->h - 1) / bh + 1;
  nb = nx * ny;

  QMALLOC(bkgout, sep_bkg, 1, status);
  bkgout->w = image->w;
  bkgout->h = image->h;
  bkgout->bw = bw;
  bkgout->bh = bh;
  bkgout->nx = nx;
  bkgout->ny = ny;
  bkgout->n = nb;
  bkgout->global = 0.0f;
  bkgout->globalrms = 0.0f;
  bkgout->back = NULL;
  bkgout->sigma = NULL;
  bkgout->dback = NULL;
  bkgout->dsigma = NULL;

  QMALLOC(bkgout->back, float, nb, status);
  QMALLOC(bkgout->sigma, float, nb, status);
  QMALLOC(bkgout->dback, float, nb, status);
  QMALLOC(bkgout->dsigma, float, nb, status);

  phase_start_ms = profile_enabled ? now_ms() : 0.0;
  status = sep_cuda_compute_meshes(
      imgbuf,
      maskbuf,
      image->w,
      image->h,
      bw,
      bh,
      maskthresh,
      bkgout->back,
      bkgout->sigma,
      profile_enabled ? &profile : NULL);
  if (status != RETURN_OK) {
    goto exit;
  }
  if (profile_enabled) {
    profile.mesh_total_ms = now_ms() - phase_start_ms;
  }

  phase_start_ms = profile_enabled ? now_ms() : 0.0;
  status = filterback(bkgout, fw, fh, fthresh);
  if (status != RETURN_OK) {
    goto exit;
  }
  if (profile_enabled) {
    profile.filter_ms = now_ms() - phase_start_ms;
  }

  phase_start_ms = profile_enabled ? now_ms() : 0.0;
  status = makebackspline(bkgout, bkgout->back, bkgout->dback);
  if (status != RETURN_OK) {
    goto exit;
  }
  if (profile_enabled) {
    profile.spline_back_ms = now_ms() - phase_start_ms;
  }

  phase_start_ms = profile_enabled ? now_ms() : 0.0;
  status = makebackspline(bkgout, bkgout->sigma, bkgout->dsigma);
  if (status != RETURN_OK) {
    goto exit;
  }
  if (profile_enabled) {
    profile.spline_sigma_ms = now_ms() - phase_start_ms;
    profile.total_background_ms = now_ms() - total_start_ms;
  }

  free(imgbuf);
  free(maskbuf);
  if (profile_enabled) {
    sepcuda_profile_commit_background(&profile);
  }
  *bkg = bkgout;
  return RETURN_OK;

exit:
  if (profile_enabled) {
    profile.total_background_ms = now_ms() - total_start_ms;
    sepcuda_profile_commit_background(&profile);
  }
  free(imgbuf);
  free(maskbuf);
  sep_bkg_free(bkgout);
  *bkg = NULL;
  return status;
}

extern "C" SEP_API float sep_bkg_global(const sep_bkg *bkg) {
  return bkg->global;
}

extern "C" SEP_API float sep_bkg_globalrms(const sep_bkg *bkg) {
  return bkg->globalrms;
}

extern "C" SEP_API float sep_bkg_pix(const sep_bkg *bkg, int64_t x, int64_t y) {
  int64_t nx, ny, xl, yl, pos;
  double dx, dy, cdx;
  float *bp;
  float b0, b1, b2, b3;

  bp = bkg->back;
  nx = bkg->nx;
  ny = bkg->ny;

  dx = (double)x / bkg->bw - 0.5;
  dy = (double)y / bkg->bh - 0.5;
  dx -= (xl = (int64_t)dx);
  dy -= (yl = (int64_t)dy);

  if (xl < 0) {
    xl = 0;
    dx -= 1.0;
  } else if (xl >= nx - 1) {
    xl = nx < 2 ? 0 : nx - 2;
    dx += 1.0;
  }

  if (yl < 0) {
    yl = 0;
    dy -= 1.0;
  } else if (yl >= ny - 1) {
    yl = ny < 2 ? 0 : ny - 2;
    dy += 1.0;
  }

  pos = yl * nx + xl;
  cdx = 1.0 - dx;

  b0 = *(bp += pos);
  b1 = nx < 2 ? b0 : *(++bp);
  b2 = ny < 2 ? *bp : *(bp += nx);
  b3 = nx < 2 ? *bp : *(--bp);

  return (float)((1.0 - dy) * (cdx * b0 + dx * b1) + dy * (dx * b2 + cdx * b3));
}

static int bkg_line_flt_internal(
    const sep_bkg *bkg, float *values, float *dvalues, int64_t y, float *line) {
  int64_t x, i, j, yl, nbx, nbxm1, nby, nx, width, ystep, changepoint;
  int status;
  float dx, dx0, dy, dy3, cdx, cdy, cdy3, temp, xstep;
  float *nodebuf, *dnodebuf, *u;
  float *node, *nodep, *dnode, *blo, *bhi, *dblo, *dbhi;

  status = RETURN_OK;
  nodebuf = NULL;
  dnodebuf = NULL;
  u = NULL;
  node = NULL;
  dnode = NULL;

  width = bkg->w;
  nbx = bkg->nx;
  nbxm1 = nbx - 1;
  nby = bkg->ny;

  if (nby > 1) {
    dy = (float)y / bkg->bh - 0.5f;
    dy -= (yl = (int64_t)dy);
    if (yl < 0) {
      yl = 0;
      dy -= 1.0f;
    } else if (yl >= nby - 1) {
      yl = nby < 2 ? 0 : nby - 2;
      dy += 1.0f;
    }

    cdy = 1.0f - dy;
    dy3 = (dy * dy * dy - dy);
    cdy3 = (cdy * cdy * cdy - cdy);
    ystep = nbx * yl;
    blo = values + ystep;
    bhi = blo + nbx;
    dblo = dvalues + ystep;
    dbhi = dblo + nbx;

    QMALLOC(nodebuf, float, nbx, status);
    nodep = node = nodebuf;
    for (x = nbx; x--;) {
      *(nodep++) = cdy * *(blo++) + dy * *(bhi++) + cdy3 * *(dblo++) + dy3 * *(dbhi++);
    }

    QMALLOC(dnodebuf, float, nbx, status);
    dnode = dnodebuf;
    if (nbx > 1) {
      QMALLOC(u, float, nbxm1, status);
      *dnode = *u = 0.0f;
      nodep = node + 1;
      for (x = nbxm1; --x; nodep++) {
        temp = -1.0f / (*(dnode++) + 4.0f);
        *dnode = temp;
        temp *= *(u++) - 6.0f * (*(nodep + 1) + *(nodep - 1) - 2.0f * *nodep);
        *u = temp;
      }
      *(++dnode) = 0.0f;
      for (x = nbx - 2; x--;) {
        temp = *(dnode--);
        *dnode = (*dnode * temp + *(u--)) / 6.0f;
      }
      dnode--;
    }
  } else {
    node = values;
    dnode = dvalues;
  }

  if (nbx > 1) {
    nx = bkg->bw;
    xstep = 1.0f / (float)nx;
    changepoint = nx / 2;
    dx = (xstep - 1.0f) / 2.0f;
    dx0 = ((nx + 1) % 2) * xstep / 2.0f;
    blo = node;
    bhi = node + 1;
    dblo = dnode;
    dbhi = dnode + 1;
    for (x = i = 0, j = width; j--; i++, dx += xstep) {
      if (i == changepoint && x > 0 && x < nbxm1) {
        blo++;
        bhi++;
        dblo++;
        dbhi++;
        dx = dx0;
      }
      cdx = 1.0f - dx;
      *(line++) = cdx * (*blo + (cdx * cdx - 1.0f) * *dblo) +
                  dx * (*bhi + (dx * dx - 1.0f) * *dbhi);
      if (i == nx) {
        x++;
        i = 0;
      }
    }
  } else {
    for (j = width; j--;) {
      *(line++) = *node;
    }
  }

exit:
  free(nodebuf);
  free(dnodebuf);
  free(u);
  return status;
}

static int sep_bkg_line_flt(const sep_bkg *bkg, int64_t y, float *line) {
  return bkg_line_flt_internal(bkg, bkg->back, bkg->dback, y, line);
}

static int sep_bkg_rmsline_flt(const sep_bkg *bkg, int64_t y, float *line) {
  return bkg_line_flt_internal(bkg, bkg->sigma, bkg->dsigma, y, line);
}

extern "C" SEP_API int sep_bkg_line(const sep_bkg *bkg, int64_t y, void *line, int dtype) {
  array_writer write_array;
  int64_t size;
  int status;
  float *tmpline;

  if (dtype == SEP_TFLOAT) {
    return sep_bkg_line_flt(bkg, y, (float *)line);
  }

  tmpline = NULL;
  status = get_array_writer(dtype, &write_array, &size);
  if (status != RETURN_OK) {
    return status;
  }

  tmpline = (float *)malloc((size_t)bkg->w * sizeof(float));
  if (tmpline == NULL) {
    put_errdetail("failed to allocate temporary line buffer");
    return MEMORY_ALLOC_ERROR;
  }

  status = sep_bkg_line_flt(bkg, y, tmpline);
  if (status == RETURN_OK) {
    write_array(tmpline, bkg->w, line);
  }

  free(tmpline);
  return status;
}

extern "C" SEP_API int sep_bkg_rmsline(const sep_bkg *bkg, int64_t y, void *line, int dtype) {
  array_writer write_array;
  int64_t size;
  int status;
  float *tmpline;

  if (dtype == SEP_TFLOAT) {
    return sep_bkg_rmsline_flt(bkg, y, (float *)line);
  }

  tmpline = NULL;
  status = get_array_writer(dtype, &write_array, &size);
  if (status != RETURN_OK) {
    return status;
  }

  tmpline = (float *)malloc((size_t)bkg->w * sizeof(float));
  if (tmpline == NULL) {
    put_errdetail("failed to allocate temporary rms line buffer");
    return MEMORY_ALLOC_ERROR;
  }

  status = sep_bkg_rmsline_flt(bkg, y, tmpline);
  if (status == RETURN_OK) {
    write_array(tmpline, bkg->w, line);
  }

  free(tmpline);
  return status;
}

extern "C" SEP_API int sep_bkg_array(const sep_bkg *bkg, void *arr, int dtype) {
  int64_t y, width, size;
  int status;
  array_writer write_array;
  float *tmpline;
  BYTE *line;

  width = bkg->w;
  if (dtype == SEP_TFLOAT) {
    tmpline = (float *)arr;
    for (y = 0; y < bkg->h; y++, tmpline += width) {
      status = sep_bkg_line_flt(bkg, y, tmpline);
      if (status != RETURN_OK) {
        return status;
      }
    }
    return RETURN_OK;
  }

  status = get_array_writer(dtype, &write_array, &size);
  if (status != RETURN_OK) {
    return status;
  }

  tmpline = (float *)malloc((size_t)width * sizeof(float));
  if (tmpline == NULL) {
    put_errdetail("failed to allocate temporary array buffer");
    return MEMORY_ALLOC_ERROR;
  }

  line = (BYTE *)arr;
  for (y = 0; y < bkg->h; y++, line += size * width) {
    status = sep_bkg_line_flt(bkg, y, tmpline);
    if (status != RETURN_OK) {
      free(tmpline);
      return status;
    }
    write_array(tmpline, width, line);
  }

  free(tmpline);
  return RETURN_OK;
}

extern "C" SEP_API int sep_bkg_rmsarray(const sep_bkg *bkg, void *arr, int dtype) {
  int64_t y, width, size;
  int status;
  array_writer write_array;
  float *tmpline;
  BYTE *line;

  width = bkg->w;
  if (dtype == SEP_TFLOAT) {
    tmpline = (float *)arr;
    for (y = 0; y < bkg->h; y++, tmpline += width) {
      status = sep_bkg_rmsline_flt(bkg, y, tmpline);
      if (status != RETURN_OK) {
        return status;
      }
    }
    return RETURN_OK;
  }

  status = get_array_writer(dtype, &write_array, &size);
  if (status != RETURN_OK) {
    return status;
  }

  tmpline = (float *)malloc((size_t)width * sizeof(float));
  if (tmpline == NULL) {
    put_errdetail("failed to allocate temporary rms array buffer");
    return MEMORY_ALLOC_ERROR;
  }

  line = (BYTE *)arr;
  for (y = 0; y < bkg->h; y++, line += size * width) {
    status = sep_bkg_rmsline_flt(bkg, y, tmpline);
    if (status != RETURN_OK) {
      free(tmpline);
      return status;
    }
    write_array(tmpline, width, line);
  }

  free(tmpline);
  return RETURN_OK;
}

extern "C" SEP_API int sep_bkg_subline(const sep_bkg *bkg, int64_t y, void *line, int dtype) {
  array_writer subtract_array;
  int64_t size;
  int status;
  PIXTYPE *tmpline;

  status = RETURN_OK;
  tmpline = NULL;
  QMALLOC(tmpline, PIXTYPE, bkg->w, status);

  status = sep_bkg_line_flt(bkg, y, tmpline);
  if (status != RETURN_OK) {
    goto exit;
  }

  status = get_array_subtractor(dtype, &subtract_array, &size);
  if (status != RETURN_OK) {
    goto exit;
  }

  subtract_array(tmpline, bkg->w, line);

exit:
  free(tmpline);
  return status;
}

extern "C" SEP_API int sep_bkg_subarray(const sep_bkg *bkg, void *arr, int dtype) {
  array_writer subtract_array;
  int64_t y, size, width;
  int status;
  PIXTYPE *tmpline;
  BYTE *arrt;

  status = RETURN_OK;
  width = bkg->w;
  tmpline = NULL;
  arrt = (BYTE *)arr;

  QMALLOC(tmpline, PIXTYPE, width, status);

  status = get_array_subtractor(dtype, &subtract_array, &size);
  if (status != RETURN_OK) {
    goto exit;
  }

  for (y = 0; y < bkg->h; y++, arrt += width * size) {
    status = sep_bkg_line_flt(bkg, y, tmpline);
    if (status != RETURN_OK) {
      goto exit;
    }
    subtract_array(tmpline, width, arrt);
  }

exit:
  free(tmpline);
  return status;
}

extern "C" SEP_API void sep_bkg_free(sep_bkg *bkg) {
  if (bkg != NULL) {
    free(bkg->back);
    free(bkg->dback);
    free(bkg->sigma);
    free(bkg->dsigma);
  }
  free(bkg);
}

static int filterback(sep_bkg *bkg, int64_t fw, int64_t fh, double fthresh) {
  float *back, *sigma, *back2, *sigma2, *bmask, *smask, *sigmat;
  float d2, d2min, med, val, sval;
  int64_t i, j, px, py, np, nx, ny, npx, npx2, npy, npy2, dpx, dpy, x, y, nmin;
  int status;

  status = RETURN_OK;
  bmask = NULL;
  smask = NULL;
  back2 = NULL;
  sigma2 = NULL;

  nx = bkg->nx;
  ny = bkg->ny;
  np = bkg->n;
  npx = fw / 2;
  npy = fh / 2;
  npy *= nx;

  QMALLOC(bmask, float, (2 * npx + 1) * (2 * npy + 1), status);
  QMALLOC(smask, float, (2 * npx + 1) * (2 * npy + 1), status);
  QMALLOC(back2, float, np, status);
  QMALLOC(sigma2, float, np, status);

  back = bkg->back;
  sigma = bkg->sigma;
  val = 0.0f;
  sval = 0.0f;

  for (i = 0, py = 0; py < ny; py++) {
    for (px = 0; px < nx; px++, i++) {
      if ((back2[i] = back[i]) <= -BIG) {
        d2min = BIG;
        nmin = 0;
        for (j = 0, y = 0; y < ny; y++) {
          for (x = 0; x < nx; x++, j++) {
            if (back[j] > -BIG) {
              d2 = (float)((x - px) * (x - px) + (y - py) * (y - py));
              if (d2 < d2min) {
                val = back[j];
                sval = sigma[j];
                nmin = 1;
                d2min = d2;
              } else if (d2 == d2min) {
                val += back[j];
                sval += sigma[j];
                nmin++;
              }
            }
          }
        }
        back2[i] = nmin ? val / (float)nmin : 0.0f;
        sigma[i] = nmin ? sval / (float)nmin : 1.0f;
      }
    }
  }
  memcpy(back, back2, (size_t)np * sizeof(float));

  for (py = 0; py < np; py += nx) {
    npy2 = np - py - nx;
    if (npy2 > npy) {
      npy2 = npy;
    }
    if (npy2 > py) {
      npy2 = py;
    }
    for (px = 0; px < nx; px++) {
      npx2 = nx - px - 1;
      if (npx2 > npx) {
        npx2 = npx;
      }
      if (npx2 > px) {
        npx2 = px;
      }
      i = 0;
      for (dpy = -npy2; dpy <= npy2; dpy += nx) {
        y = py + dpy;
        for (dpx = -npx2; dpx <= npx2; dpx++) {
          x = px + dpx;
          bmask[i] = back[x + y];
          smask[i++] = sigma[x + y];
        }
      }
      med = fqmedian(bmask, i);
      if (fabs((double)(med - back[px + py])) >= fthresh) {
        back2[px + py] = med;
        sigma2[px + py] = fqmedian(smask, i);
      } else {
        back2[px + py] = back[px + py];
        sigma2[px + py] = sigma[px + py];
      }
    }
  }

  memcpy(back, back2, (size_t)np * sizeof(float));
  bkg->global = fqmedian(back2, np);
  memcpy(sigma, sigma2, (size_t)np * sizeof(float));
  bkg->globalrms = fqmedian(sigma2, np);

  if (bkg->globalrms <= 0.0f) {
    sigmat = sigma2 + np;
    for (i = np; i-- && *(--sigmat) > 0.0f;) {
    }
    if (i >= 0 && i < np - 1) {
      bkg->globalrms = fqmedian(sigmat + 1, np - 1 - i);
    } else {
      bkg->globalrms = 1.0f;
    }
  }

exit:
  free(bmask);
  free(smask);
  free(back2);
  free(sigma2);
  return status;
}

static int makebackspline(const sep_bkg *bkg, float *map, float *dmap) {
  int64_t x, y, nbx, nby, nbym1;
  int status;
  float *dmapt, *mapt, *u, temp;

  status = RETURN_OK;
  u = NULL;
  nbx = bkg->nx;
  nby = bkg->ny;
  nbym1 = nby - 1;

  for (x = 0; x < nbx; x++) {
    mapt = map + x;
    dmapt = dmap + x;
    if (nby > 1) {
      QMALLOC(u, float, nbym1, status);
      *dmapt = *u = 0.0f;
      mapt += nbx;
      for (y = 1; y < nbym1; y++, mapt += nbx) {
        temp = -1.0f / (*dmapt + 4.0f);
        *(dmapt += nbx) = temp;
        temp *= *(u++) - 6.0f * (*(mapt + nbx) + *(mapt - nbx) - 2.0f * *mapt);
        *u = temp;
      }
      *(dmapt += nbx) = 0.0f;
      for (y = nby - 2; y--;) {
        temp = *dmapt;
        dmapt -= nbx;
        *dmapt = (*dmapt * temp + *(u--)) / 6.0f;
      }
      free(u);
      u = NULL;
    } else {
      *dmapt = 0.0f;
    }
  }

  return RETURN_OK;

exit:
  free(u);
  return status;
}
