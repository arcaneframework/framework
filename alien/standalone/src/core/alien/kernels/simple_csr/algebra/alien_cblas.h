#pragma once

#ifdef ALIEN_USE_MKL
#include <mkl_cblas.h>
#else
#include <clas.h>
#endif

namespace cblas
{

inline float dot(const int n, const float* x, const int incx, const float* y,
          const int incy)
{
  return cblas_sdot(n, x, incx, y, incy);
}

inline double dot(const int n, const double* x, const int incx, const double* y, const int incy)
{
  return cblas_ddot(n, x, incx, y, incy);
}

inline void axpy(const int n, const float alpha, const float* x, const int incx, float* y, const int incy)
{
  return cblas_saxpy(n, alpha, x, incx, y, incy);
}

inline void axpy(const int n, const double alpha, const double* x, const int incx, double* y, const int incy)
{
  return cblas_daxpy(n, alpha, x, incx, y, incy);
}

inline void copy(const int n, const float* x, const int incx, float* y, const int incy)
{
  return cblas_scopy(n, x, incx, y, incy);
}

inline void copy(const int n, const double* x, const int incx, double* y, const int incy)
{
  return cblas_dcopy(n, x, incx, y, incy);
}

inline void scal(const int n, const float alpha, float* x, const int incx)
{
  return cblas_sscal(n, alpha, x, incx);
}

inline void scal(const int n, const double alpha, double* x, const int incx)
{
  return cblas_dscal(n, alpha, x, incx);
}

inline double nrm1(const int n, const double* x, const int incx)
{
  return cblas_dasum(n, x, incx);
}

inline double nrm2(const int n, const double* x, const int incx)
{
  return cblas_dnrm2(n, x, incx);
}

} // namespace cblas
