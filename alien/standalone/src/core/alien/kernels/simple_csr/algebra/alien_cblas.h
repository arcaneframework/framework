#pragma once

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

float cblas_sdot(const int n, const float* x, const int incx, const float* y,
                 const int incy);
double cblas_ddot(const int n, const double* x, const int incx, const double* y,
                  const int incy);
void cblas_saxpy(const int n, const float alpha, const float* x, const int incx,
                 float* y, const int incy);
void cblas_daxpy(const int n, const double alpha, const double* x,
                 const int incx, double* y, const int incy);
void cblas_scopy(const int n, const float* x, const int incx, float* y,
                 const int incy);
void cblas_dcopy(const int n, const double* x, const int incx, double* y,
                 const int incy);
void cblas_sscal(const int n, const float alpha, float* x, const int incx);
void cblas_dscal(const int n, const double alpha, double* x, const int incx);
double cblas_dasum(const int n, const double* x, const int incx);
double cblas_dnrm2(const int n, const double* x, const int incx);

#ifdef __cplusplus
}
#endif // __cplusplus

namespace cblas
{

float dot(const int n, const float* x, const int incx, const float* y,
          const int incy)
{
  return cblas_sdot(n, x, incx, y, incy);
}

double dot(const int n, const double* x, const int incx, const double* y, const int incy)
{
  return cblas_ddot(n, x, incx, y, incy);
}

void axpy(const int n, const float alpha, const float* x, const int incx, float* y, const int incy)
{
  return cblas_saxpy(n, alpha, x, incx, y, incy);
}

void axpy(const int n, const double alpha, const double* x, const int incx, double* y, const int incy)
{
  return cblas_daxpy(n, alpha, x, incx, y, incy);
}

void copy(const int n, const float* x, const int incx, float* y, const int incy)
{
  return cblas_scopy(n, x, incx, y, incy);
}

void copy(const int n, const double* x, const int incx, double* y, const int incy)
{
  return cblas_dcopy(n, x, incx, y, incy);
}

void scal(const int n, const float alpha, float* x, const int incx)
{
  return cblas_sscal(n, alpha, x, incx);
}

void scal(const int n, const double alpha, double* x, const int incx)
{
  return cblas_dscal(n, alpha, x, incx);
}

double nrm1(const int n, const double* x, const int incx)
{
  return cblas_dasum(n, x, incx);
}

double nrm2(const int n, const double* x, const int incx)
{
  return cblas_dnrm2(n, x, incx);
}

} // namespace cblas
