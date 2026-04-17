// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* relaxation_cusparse_ilu0.h                                  (C) 2000-2026 */
/*                                                                           */
/* Implementation of ILU0 smoother for CUDA backend.                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_ALINA_RELAXATION_CUSPARSEILU0_H
#define ARCCORE_ALINA_RELAXATION_CUSPARSEILU0_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*
 * This file is based on the work on AMGCL library (version march 2026)
 * which can be found at https://github.com/ddemidov/amgcl.
 *
 * Copyright (c) 2012-2022 Denis Demidov <dennis.demidov@gmail.com>
 * SPDX-License-Identifier: MIT
 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include <type_traits>

#include <thrust/device_vector.h>
#include <cusparse_v2.h>

#include "arccore/alina/CudaBackend.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Alina::relaxation
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Backend> struct ilu0;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Implementation of ILU0 smoother for CUDA backend.
 */
template <typename real>
struct ilu0<backend::cuda<real>>
{
  typedef real value_type;
  typedef backend::cuda<real> Backend;

  struct params
  {
    /// Damping factor.
    float damping = 1.0;

    params() = default;

    params(const PropertyTree& p)
    : ARCCORE_ALINA_PARAMS_IMPORT_VALUE(p, damping)
    {
      p.check_params({ "damping" });
    }

    void get(Alina::PropertyTree& p, const std::string& path) const
    {
      ARCCORE_ALINA_PARAMS_EXPORT_VALUE(p, path, damping);
    }
  } prm;

  template <class Matrix>
  ilu0(const Matrix& A, const params& prm, const typename Backend::params& bprm)
  : prm(prm)
  , handle(bprm.cusparse_handle)
  , n(backend::nbRow(A))
  , nnz(backend::nonzeros(A))
  , ptr(A.ptr, A.ptr + n + 1)
  , col(A.col, A.col + nnz)
  , val(A.val, A.val + nnz)
  , y(n)
  {
    // LU decomposition
    std::shared_ptr<std::remove_pointer<cusparseMatDescr_t>::type> descr_M;
    std::shared_ptr<std::remove_pointer<csrilu02Info_t>::type> info_M;

    {
      cusparseMatDescr_t descr;
      csrilu02Info_t info;

      ARCCORE_ALINA_CALL_CUDA(cusparseCreateMatDescr(&descr));
      ARCCORE_ALINA_CALL_CUDA(cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO));
      ARCCORE_ALINA_CALL_CUDA(cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL));

      ARCCORE_ALINA_CALL_CUDA(cusparseCreateCsrilu02Info(&info));

      descr_M.reset(descr, backend::detail::cuda_deleter());
      info_M.reset(info, backend::detail::cuda_deleter());

      int buf_size;

      ARCCORE_ALINA_CALL_CUDA(
      cusparseXcsrilu02_bufferSize(handle, n, nnz, descr_M.get(),
                                   thrust::raw_pointer_cast(&val[0]),
                                   thrust::raw_pointer_cast(&ptr[0]),
                                   thrust::raw_pointer_cast(&col[0]),
                                   info_M.get(), &buf_size));

      thrust::device_vector<char> bufLU(buf_size);

      // Analysis and incomplete factorization of the system matrix.
      int structural_zero;
      int numerical_zero;

      ARCCORE_ALINA_CALL_CUDA(
      cusparseXcsrilu02_analysis(handle,
                                 n,
                                 nnz,
                                 descr_M.get(),
                                 thrust::raw_pointer_cast(&val[0]),
                                 thrust::raw_pointer_cast(&ptr[0]),
                                 thrust::raw_pointer_cast(&col[0]),
                                 info_M.get(),
                                 CUSPARSE_SOLVE_POLICY_USE_LEVEL,
                                 thrust::raw_pointer_cast(&bufLU[0])));

      precondition(
      CUSPARSE_STATUS_ZERO_PIVOT != cusparseXcsrilu02_zeroPivot(handle, info_M.get(), &structural_zero),
      "Zero pivot in cuSPARSE ILU0");

      ARCCORE_ALINA_CALL_CUDA(
      cusparseXcsrilu02(handle,
                        n,
                        nnz,
                        descr_M.get(),
                        thrust::raw_pointer_cast(&val[0]),
                        thrust::raw_pointer_cast(&ptr[0]),
                        thrust::raw_pointer_cast(&col[0]),
                        info_M.get(),
                        CUSPARSE_SOLVE_POLICY_USE_LEVEL,
                        thrust::raw_pointer_cast(&bufLU[0])));
      precondition(
      CUSPARSE_STATUS_ZERO_PIVOT != cusparseXcsrilu02_zeroPivot(handle, info_M.get(), &numerical_zero),
      "Zero pivot in cuSPARSE ILU0");
    }

    // Triangular solvers
#if CUDART_VERSION >= 11000
    const real alpha = 1;
    thrust::device_vector<value_type> t(n);

    descr_y.reset(
    backend::detail::cuda_vector_description(y),
    backend::detail::cuda_deleter());

    std::shared_ptr<std::remove_pointer<cusparseDnVecDescr_t>::type> descr_t(
    backend::detail::cuda_vector_description(t),
    backend::detail::cuda_deleter());

    cusparseFillMode_t fill_lower = CUSPARSE_FILL_MODE_LOWER;
    cusparseFillMode_t fill_upper = CUSPARSE_FILL_MODE_UPPER;
    cusparseDiagType_t diag_unit = CUSPARSE_DIAG_TYPE_UNIT;
    cusparseDiagType_t diag_non_unit = CUSPARSE_DIAG_TYPE_NON_UNIT;

    // Triangular solver for L
    {
      descr_L.reset(
      backend::detail::cuda_matrix_description(n, n, nnz, ptr, col, val),
      backend::detail::cuda_deleter());

      ARCCORE_ALINA_CALL_CUDA(
      cusparseSpMatSetAttribute(descr_L.get(),
                                CUSPARSE_SPMAT_FILL_MODE,
                                &fill_lower,
                                sizeof(fill_lower)));

      ARCCORE_ALINA_CALL_CUDA(
      cusparseSpMatSetAttribute(descr_L.get(),
                                CUSPARSE_SPMAT_DIAG_TYPE,
                                &diag_unit,
                                sizeof(diag_unit)));

      size_t buf_size;

      cusparseSpSVDescr_t desc;
      ARCCORE_ALINA_CALL_CUDA(cusparseSpSV_createDescr(&desc));
      descr_SL.reset(desc, backend::detail::cuda_deleter());

      ARCCORE_ALINA_CALL_CUDA(
      cusparseSpSV_bufferSize(handle,
                              CUSPARSE_OPERATION_NON_TRANSPOSE,
                              &alpha,
                              descr_L.get(),
                              descr_t.get(),
                              descr_y.get(),
                              backend::detail::cuda_datatype<real>(),
                              CUSPARSE_SPSV_ALG_DEFAULT,
                              descr_SL.get(),
                              &buf_size));

      bufL.resize(buf_size);

      ARCCORE_ALINA_CALL_CUDA(
      cusparseSpSV_analysis(handle,
                            CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &alpha,
                            descr_L.get(),
                            descr_t.get(),
                            descr_y.get(),
                            backend::detail::cuda_datatype<real>(),
                            CUSPARSE_SPSV_ALG_DEFAULT,
                            descr_SL.get(),
                            thrust::raw_pointer_cast(&bufL[0])));
    }

    // Triangular solver for U
    {
      descr_U.reset(
      backend::detail::cuda_matrix_description(n, n, nnz, ptr, col, val),
      backend::detail::cuda_deleter());

      ARCCORE_ALINA_CALL_CUDA(
      cusparseSpMatSetAttribute(descr_U.get(),
                                CUSPARSE_SPMAT_FILL_MODE,
                                &fill_upper,
                                sizeof(fill_upper)));

      ARCCORE_ALINA_CALL_CUDA(
      cusparseSpMatSetAttribute(descr_U.get(),
                                CUSPARSE_SPMAT_DIAG_TYPE,
                                &diag_non_unit,
                                sizeof(diag_non_unit)));

      size_t buf_size;

      cusparseSpSVDescr_t desc;
      ARCCORE_ALINA_CALL_CUDA(cusparseSpSV_createDescr(&desc));
      descr_SU.reset(desc, backend::detail::cuda_deleter());

      ARCCORE_ALINA_CALL_CUDA(
      cusparseSpSV_bufferSize(handle,
                              CUSPARSE_OPERATION_NON_TRANSPOSE,
                              &alpha,
                              descr_U.get(),
                              descr_y.get(),
                              descr_t.get(),
                              backend::detail::cuda_datatype<real>(),
                              CUSPARSE_SPSV_ALG_DEFAULT,
                              descr_SU.get(),
                              &buf_size));

      bufU.resize(buf_size);

      ARCCORE_ALINA_CALL_CUDA(
      cusparseSpSV_analysis(handle,
                            CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &alpha,
                            descr_U.get(),
                            descr_y.get(),
                            descr_t.get(),
                            backend::detail::cuda_datatype<real>(),
                            CUSPARSE_SPSV_ALG_DEFAULT,
                            descr_SU.get(),
                            thrust::raw_pointer_cast(&bufU[0])));
    }
#else // CUDART_VERSION >= 11000
    {
      cusparseMatDescr_t descr;

      ARCCORE_ALINA_CALL_CUDA(cusparseCreateMatDescr(&descr));
      ARCCORE_ALINA_CALL_CUDA(cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO));
      ARCCORE_ALINA_CALL_CUDA(cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL));
      ARCCORE_ALINA_CALL_CUDA(cusparseSetMatFillMode(descr, CUSPARSE_FILL_MODE_LOWER));
      ARCCORE_ALINA_CALL_CUDA(cusparseSetMatDiagType(descr, CUSPARSE_DIAG_TYPE_UNIT));

      descr_L.reset(descr, backend::detail::cuda_deleter());
    }
    {
      cusparseMatDescr_t descr;

      ARCCORE_ALINA_CALL_CUDA(cusparseCreateMatDescr(&descr));
      ARCCORE_ALINA_CALL_CUDA(cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO));
      ARCCORE_ALINA_CALL_CUDA(cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL));
      ARCCORE_ALINA_CALL_CUDA(cusparseSetMatFillMode(descr, CUSPARSE_FILL_MODE_UPPER));
      ARCCORE_ALINA_CALL_CUDA(cusparseSetMatDiagType(descr, CUSPARSE_DIAG_TYPE_NON_UNIT));

      descr_U.reset(descr, backend::detail::cuda_deleter());
    }

    // Create info structures.
    {
      csrsv2Info_t info;
      ARCCORE_ALINA_CALL_CUDA(cusparseCreateCsrsv2Info(&info));
      info_L.reset(info, backend::detail::cuda_deleter());
    }
    {
      csrsv2Info_t info;
      ARCCORE_ALINA_CALL_CUDA(cusparseCreateCsrsv2Info(&info));
      info_U.reset(info, backend::detail::cuda_deleter());
    }

    // Allocate scratch buffer.
    {
      int buf_size_L;
      int buf_size_U;

      ARCCORE_ALINA_CALL_CUDA(
      cusparseXcsrsv2_bufferSize(handle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 n,
                                 nnz,
                                 descr_L.get(),
                                 thrust::raw_pointer_cast(&val[0]),
                                 thrust::raw_pointer_cast(&ptr[0]),
                                 thrust::raw_pointer_cast(&col[0]),
                                 info_L.get(), &buf_size_L));

      ARCCORE_ALINA_CALL_CUDA(
      cusparseXcsrsv2_bufferSize(handle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 n,
                                 nnz,
                                 descr_U.get(),
                                 thrust::raw_pointer_cast(&val[0]),
                                 thrust::raw_pointer_cast(&ptr[0]),
                                 thrust::raw_pointer_cast(&col[0]),
                                 info_U.get(), &buf_size_U));

      buf.resize(std::max(buf_size_L, buf_size_U));
    }

    ARCCORE_ALINA_CALL_CUDA(
    cusparseXcsrsv2_analysis(handle,
                             CUSPARSE_OPERATION_NON_TRANSPOSE,
                             n,
                             nnz,
                             descr_L.get(),
                             thrust::raw_pointer_cast(&val[0]),
                             thrust::raw_pointer_cast(&ptr[0]),
                             thrust::raw_pointer_cast(&col[0]),
                             info_L.get(), CUSPARSE_SOLVE_POLICY_USE_LEVEL,
                             thrust::raw_pointer_cast(&buf[0])));

    ARCCORE_ALINA_CALL_CUDA(
    cusparseXcsrsv2_analysis(handle,
                             CUSPARSE_OPERATION_NON_TRANSPOSE,
                             n,
                             nnz,
                             descr_U.get(),
                             thrust::raw_pointer_cast(&val[0]),
                             thrust::raw_pointer_cast(&ptr[0]),
                             thrust::raw_pointer_cast(&col[0]),
                             info_U.get(), CUSPARSE_SOLVE_POLICY_USE_LEVEL,
                             thrust::raw_pointer_cast(&buf[0])));
#endif // CUDART_VERSION >= 11000
  }

  template <class Matrix, class VectorRHS, class VectorX, class VectorTMP>
  void apply_pre(const Matrix& A, const VectorRHS& rhs, VectorX& x, VectorTMP& tmp) const
  {
    backend::residual(rhs, A, x, tmp);
    solve(tmp);
    backend::axpby(prm.damping, tmp, 1, x);
  }

  template <class Matrix, class VectorRHS, class VectorX, class VectorTMP>
  void apply_post(const Matrix& A, const VectorRHS& rhs, VectorX& x, VectorTMP& tmp) const
  {
    backend::residual(rhs, A, x, tmp);
    solve(tmp);
    backend::axpby(prm.damping, tmp, 1, x);
  }

  template <class Matrix, class VectorRHS, class VectorX>
  void apply(const Matrix& A, const VectorRHS& rhs, VectorX& x) const
  {
    backend::copy(rhs, x);
    solve(x);
  }

  size_t bytes() const
  {
    // This is incomplete, as cusparse structs are opaque.
    return backend::bytes(ptr) +
    backend::bytes(col) +
    backend::bytes(val) +
    backend::bytes(y) +
#if CUDART_VERSION >= 11000
    backend::bytes(bufL) +
    backend::bytes(bufU)
#else
    backend::bytes(buf)
#endif
    ;
  }

 private:

  cusparseHandle_t handle;
  int n, nnz;

  thrust::device_vector<int> ptr, col;
  thrust::device_vector<value_type> val;
  mutable thrust::device_vector<value_type> y;

#if CUDART_VERSION >= 11000
  std::shared_ptr<std::remove_pointer<cusparseSpMatDescr_t>::type> descr_L, descr_U;
  std::shared_ptr<std::remove_pointer<cusparseSpSVDescr_t>::type> descr_SL, descr_SU;
  std::shared_ptr<std::remove_pointer<cusparseDnVecDescr_t>::type> descr_y;
  mutable thrust::device_vector<char> bufL, bufU;
#else
  std::shared_ptr<std::remove_pointer<cusparseMatDescr_t>::type> descr_L, descr_U;
  std::shared_ptr<std::remove_pointer<csrsv2Info_t>::type> info_L, info_U;
  mutable thrust::device_vector<char> buf;
#endif

  template <class VectorX>
  void solve(VectorX& x) const
  {
    value_type alpha = 1;

#if CUDART_VERSION >= 11000
    std::shared_ptr<std::remove_pointer<cusparseDnVecDescr_t>::type> descr_x(
    backend::detail::cuda_vector_description(x),
    backend::detail::cuda_deleter());

    // Solve L * y = x
    ARCCORE_ALINA_CALL_CUDA(
    cusparseSpSV_solve(handle,
                       CUSPARSE_OPERATION_NON_TRANSPOSE,
                       &alpha,
                       descr_L.get(),
                       descr_x.get(),
                       descr_y.get(),
                       backend::detail::cuda_datatype<real>(),
                       CUSPARSE_SPSV_ALG_DEFAULT,
                       descr_SL.get()));

    // Solve U * x = y
    ARCCORE_ALINA_CALL_CUDA(
    cusparseSpSV_solve(handle,
                       CUSPARSE_OPERATION_NON_TRANSPOSE,
                       &alpha,
                       descr_U.get(),
                       descr_y.get(),
                       descr_x.get(),
                       backend::detail::cuda_datatype<real>(),
                       CUSPARSE_SPSV_ALG_DEFAULT,
                       descr_SU.get()));
#else // CUDART_VERSION >= 11000
    // Solve L * y = x
    ARCCORE_ALINA_CALL_CUDA(
    cusparseXcsrsv2_solve(handle,
                          CUSPARSE_OPERATION_NON_TRANSPOSE,
                          n,
                          nnz,
                          &alpha,
                          descr_L.get(),
                          thrust::raw_pointer_cast(&val[0]),
                          thrust::raw_pointer_cast(&ptr[0]),
                          thrust::raw_pointer_cast(&col[0]),
                          info_L.get(),
                          thrust::raw_pointer_cast(&x[0]),
                          thrust::raw_pointer_cast(&y[0]),
                          CUSPARSE_SOLVE_POLICY_USE_LEVEL,
                          thrust::raw_pointer_cast(&buf[0])));

    // Solve U * x = y
    ARCCORE_ALINA_CALL_CUDA(
    cusparseXcsrsv2_solve(handle,
                          CUSPARSE_OPERATION_NON_TRANSPOSE,
                          n,
                          nnz,
                          &alpha,
                          descr_U.get(),
                          thrust::raw_pointer_cast(&val[0]),
                          thrust::raw_pointer_cast(&ptr[0]),
                          thrust::raw_pointer_cast(&col[0]),
                          info_U.get(),
                          thrust::raw_pointer_cast(&y[0]),
                          thrust::raw_pointer_cast(&x[0]),
                          CUSPARSE_SOLVE_POLICY_USE_LEVEL,
                          thrust::raw_pointer_cast(&buf[0])));
#endif // CUDART_VERSION >= 11000
  }

  static cusparseStatus_t
  cusparseXcsrilu02_bufferSize(cusparseHandle_t handle,
                               int m,
                               int nnz,
                               const cusparseMatDescr_t descrA,
                               double* csrSortedValA,
                               const int* csrSortedRowPtrA,
                               const int* csrSortedColIndA,
                               csrilu02Info_t info,
                               int* pBufferSizeInBytes)
  {
    return cusparseDcsrilu02_bufferSize(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA,
                                        csrSortedColIndA, info, pBufferSizeInBytes);
  }

  static cusparseStatus_t
  cusparseXcsrilu02_bufferSize(cusparseHandle_t handle,
                               int m,
                               int nnz,
                               const cusparseMatDescr_t descrA,
                               float* csrSortedValA,
                               const int* csrSortedRowPtrA,
                               const int* csrSortedColIndA,
                               csrilu02Info_t info,
                               int* pBufferSizeInBytes)
  {
    return cusparseScsrilu02_bufferSize(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA,
                                        csrSortedColIndA, info, pBufferSizeInBytes);
  }

  static cusparseStatus_t
  cusparseXcsrilu02_analysis(cusparseHandle_t handle,
                             int m,
                             int nnz,
                             const cusparseMatDescr_t descrA,
                             const double* csrSortedValA,
                             const int* csrSortedRowPtrA,
                             const int* csrSortedColIndA,
                             csrilu02Info_t info,
                             cusparseSolvePolicy_t policy,
                             void* pBuffer)
  {
    return cusparseDcsrilu02_analysis(handle, m, nnz, descrA, csrSortedValA,
                                      csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer);
  }

  static cusparseStatus_t
  cusparseXcsrilu02_analysis(cusparseHandle_t handle,
                             int m,
                             int nnz,
                             const cusparseMatDescr_t descrA,
                             const float* csrSortedValA,
                             const int* csrSortedRowPtrA,
                             const int* csrSortedColIndA,
                             csrilu02Info_t info,
                             cusparseSolvePolicy_t policy,
                             void* pBuffer)
  {
    return cusparseScsrilu02_analysis(handle, m, nnz, descrA, csrSortedValA,
                                      csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer);
  }

  static cusparseStatus_t
  cusparseXcsrilu02(cusparseHandle_t handle,
                    int m,
                    int nnz,
                    const cusparseMatDescr_t descrA,
                    double* csrSortedValA_valM,
                    const int* csrSortedRowPtrA,
                    const int* csrSortedColIndA,
                    csrilu02Info_t info,
                    cusparseSolvePolicy_t policy,
                    void* pBuffer)
  {
    return cusparseDcsrilu02(handle, m, nnz, descrA,
                             csrSortedValA_valM, csrSortedRowPtrA, csrSortedColIndA,
                             info, policy, pBuffer);
  }

  static cusparseStatus_t
  cusparseXcsrilu02(cusparseHandle_t handle,
                    int m,
                    int nnz,
                    const cusparseMatDescr_t descrA,
                    float* csrSortedValA_valM,
                    const int* csrSortedRowPtrA,
                    const int* csrSortedColIndA,
                    csrilu02Info_t info,
                    cusparseSolvePolicy_t policy,
                    void* pBuffer)
  {
    return cusparseScsrilu02(handle, m, nnz, descrA,
                             csrSortedValA_valM, csrSortedRowPtrA, csrSortedColIndA,
                             info, policy, pBuffer);
  }

#if CUDART_VERSION < 11000
  static cusparseStatus_t
  cusparseXcsrsv2_bufferSize(cusparseHandle_t handle,
                             cusparseOperation_t transA,
                             int m,
                             int nnz,
                             const cusparseMatDescr_t descrA,
                             double* csrSortedValA,
                             const int* csrSortedRowPtrA,
                             const int* csrSortedColIndA,
                             csrsv2Info_t info,
                             int* pBufferSizeInBytes)
  {
    return cusparseDcsrsv2_bufferSize(handle, transA, m, nnz, descrA, csrSortedValA,
                                      csrSortedRowPtrA, csrSortedColIndA, info, pBufferSizeInBytes);
  }

  static cusparseStatus_t
  cusparseXcsrsv2_bufferSize(cusparseHandle_t handle,
                             cusparseOperation_t transA,
                             int m,
                             int nnz,
                             const cusparseMatDescr_t descrA,
                             float* csrSortedValA,
                             const int* csrSortedRowPtrA,
                             const int* csrSortedColIndA,
                             csrsv2Info_t info,
                             int* pBufferSizeInBytes)
  {
    return cusparseScsrsv2_bufferSize(handle, transA, m, nnz, descrA, csrSortedValA,
                                      csrSortedRowPtrA, csrSortedColIndA, info, pBufferSizeInBytes);
  }

  static cusparseStatus_t
  cusparseXcsrsv2_analysis(cusparseHandle_t handle,
                           cusparseOperation_t transA,
                           int m,
                           int nnz,
                           const cusparseMatDescr_t descrA,
                           const double* csrSortedValA,
                           const int* csrSortedRowPtrA,
                           const int* csrSortedColIndA,
                           csrsv2Info_t info,
                           cusparseSolvePolicy_t policy,
                           void* pBuffer)
  {
    return cusparseDcsrsv2_analysis(handle, transA, m, nnz, descrA, csrSortedValA,
                                    csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer);
  }

  static cusparseStatus_t
  cusparseXcsrsv2_analysis(cusparseHandle_t handle,
                           cusparseOperation_t transA,
                           int m,
                           int nnz,
                           const cusparseMatDescr_t descrA,
                           const float* csrSortedValA,
                           const int* csrSortedRowPtrA,
                           const int* csrSortedColIndA,
                           csrsv2Info_t info,
                           cusparseSolvePolicy_t policy,
                           void* pBuffer)
  {
    return cusparseScsrsv2_analysis(handle, transA, m, nnz, descrA, csrSortedValA,
                                    csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer);
  }

  static cusparseStatus_t
  cusparseXcsrsv2_solve(cusparseHandle_t handle,
                        cusparseOperation_t transA,
                        int m,
                        int nnz,
                        const double* alpha,
                        const cusparseMatDescr_t descrA,
                        const double* csrSortedValA,
                        const int* csrSortedRowPtrA,
                        const int* csrSortedColIndA,
                        csrsv2Info_t info,
                        const double* f,
                        double* x,
                        cusparseSolvePolicy_t policy,
                        void* pBuffer)
  {
    return cusparseDcsrsv2_solve(handle, transA, m,
                                 nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA,
                                 csrSortedColIndA, info, f, x, policy, pBuffer);
  }

  static cusparseStatus_t
  cusparseXcsrsv2_solve(cusparseHandle_t handle,
                        cusparseOperation_t transA,
                        int m,
                        int nnz,
                        const float* alpha,
                        const cusparseMatDescr_t descrA,
                        const float* csrSortedValA,
                        const int* csrSortedRowPtrA,
                        const int* csrSortedColIndA,
                        csrsv2Info_t info,
                        const float* f,
                        float* x,
                        cusparseSolvePolicy_t policy,
                        void* pBuffer)
  {
    return cusparseScsrsv2_solve(handle, transA, m,
                                 nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA,
                                 csrSortedColIndA, info, f, x, policy, pBuffer);
  }
#endif // CUDART_VERSION < 11000
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Alina::relaxation

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
