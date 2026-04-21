// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CudaBackend.h                                               (C) 2000-2026 */
/*                                                                           */
/* backend using Cuda runtime.                                               */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_ALINA_CUDABACKEND_H
#define ARCCORE_ALINA_CUDABACKEND_H
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
#include <memory>

#include "arccore/alina/BuiltinBackend.h"
#include "arccore/alina/SkylineLUSolver.h"
#include "arccore/alina/AlinaUtils.h"

#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/copy.h>
#include <thrust/gather.h>
#include <thrust/scatter.h>
#include <thrust/for_each.h>
#include <thrust/inner_product.h>
#include <cusparse_v2.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Alina::solver
{
/** Wrapper around solver::skyline_lu for use with the CUDA backend.
 * Copies the rhs to the host memory, solves the problem using the host CPU,
 * then copies the solution back to the compute device(s).
 */
template <class T>
struct cuda_skyline_lu : SkylineLUSolver<T>
{
  typedef SkylineLUSolver<T> Base;

  mutable std::vector<T> _rhs, _x;

  template <class Matrix, class Params>
  cuda_skyline_lu(const Matrix& A, const Params&)
  : Base(*A)
  , _rhs(backend::nbRow(*A))
  , _x(backend::nbRow(*A))
  {}

  template <class Vec1, class Vec2>
  void operator()(const Vec1& rhs, Vec2& x) const
  {
    thrust::copy(rhs.begin(), rhs.end(), _rhs.begin());
    static_cast<const Base*>(this)->operator()(_rhs, _x);
    thrust::copy(_x.begin(), _x.end(), x.begin());
  }

  size_t bytes() const
  {
    return backend::bytes(*static_cast<const Base*>(this)) +
    backend::bytes(_rhs) +
    backend::bytes(_x);
  }
};

}

namespace Arcane::Alina::backend::detail
{

inline void
cuda_check(cusparseStatus_t rc, const char* file, int line)
{
  if (rc != CUSPARSE_STATUS_SUCCESS) {
    std::ostringstream msg;
    msg << "CUDA error " << rc << " at \"" << file << ":" << line;
    precondition(false, msg.str());
  }
}

inline void
cuda_check(cudaError_t rc, const char* file, int line)
{
  if (rc != cudaSuccess) {
    std::ostringstream msg;
    msg << "CUDA error " << rc << " at \"" << file << ":" << line;
    precondition(false, msg.str());
  }
}

#define ARCCORE_ALINA_CALL_CUDA(rc) \
  Arcane::Alina::backend::detail::cuda_check(rc, __FILE__, __LINE__)

struct cuda_deleter
{
  void operator()(cusparseMatDescr_t handle)
  {
    ARCCORE_ALINA_CALL_CUDA(cusparseDestroyMatDescr(handle));
  }

  void operator()(cusparseSpMatDescr_t handle)
  {
    ARCCORE_ALINA_CALL_CUDA(cusparseDestroySpMat(handle));
  }

  void operator()(cusparseDnVecDescr_t handle)
  {
    ARCCORE_ALINA_CALL_CUDA(cusparseDestroyDnVec(handle));
  }

  void operator()(cudaEvent_t handle)
  {
    ARCCORE_ALINA_CALL_CUDA(cudaEventDestroy(handle));
  }

  void operator()(csrilu02Info_t handle)
  {
    ARCCORE_ALINA_CALL_CUDA(cusparseDestroyCsrilu02Info(handle));
  }

  void operator()(cusparseSpSVDescr_t handle)
  {
    ARCCORE_ALINA_CALL_CUDA(cusparseSpSV_destroyDescr(handle));
  }
};

template <typename real>
cudaDataType cuda_datatype()
{
  if (sizeof(real) == sizeof(float))
    return CUDA_R_32F;
  else
    return CUDA_R_64F;
}

template <typename real>
cusparseDnVecDescr_t cuda_vector_description(thrust::device_vector<real>& x)
{
  cusparseDnVecDescr_t desc;
  ARCCORE_ALINA_CALL_CUDA(cusparseCreateDnVec(&desc,
                                             x.size(),
                                             thrust::raw_pointer_cast(&x[0]),
                                             cuda_datatype<real>()));
  return desc;
}

template <typename real> cusparseDnVecDescr_t
cuda_vector_description(const thrust::device_vector<real>&& x)
{
  cusparseDnVecDescr_t desc;
  ARCCORE_ALINA_CALL_CUDA(
  cusparseCreateDnVec(&desc,
                      x.size(),
                      thrust::raw_pointer_cast(&x[0]),
                      cuda_datatype<real>()));
  return desc;
}

template <typename real> cusparseSpMatDescr_t
cuda_matrix_description(size_t nrows,
                        size_t ncols,
                        size_t nnz,
                        thrust::device_vector<int>& ptr,
                        thrust::device_vector<int>& col,
                        thrust::device_vector<real>& val)
{
  cusparseSpMatDescr_t desc;
  ARCCORE_ALINA_CALL_CUDA(
  cusparseCreateCsr(&desc,
                    nrows,
                    ncols,
                    nnz,
                    thrust::raw_pointer_cast(&ptr[0]),
                    thrust::raw_pointer_cast(&col[0]),
                    thrust::raw_pointer_cast(&val[0]),
                    CUSPARSE_INDEX_32I,
                    CUSPARSE_INDEX_32I,
                    CUSPARSE_INDEX_BASE_ZERO,
                    detail::cuda_datatype<real>()));
  return desc;
}
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Alina::backend
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/// CUSPARSE matrix in CSR format.
template <typename real>
class cuda_matrix
{
 public:

  typedef real value_type;

  cuda_matrix(size_t n, size_t m,
              const ptrdiff_t* p_ptr,
              const ptrdiff_t* p_col,
              const real* p_val,
              cusparseHandle_t handle)
  : nrows(n)
  , ncols(m)
  , nnz(p_ptr[n])
  , handle(handle)
  , ptr(p_ptr, p_ptr + n + 1)
  , col(p_col, p_col + nnz)
  , val(p_val, p_val + nnz)
  {
    desc.reset(detail::cuda_matrix_description(nrows, ncols, nnz, ptr, col, val),
               backend::detail::cuda_deleter());
  }

  void spmv(real alpha, thrust::device_vector<real> const& x,
            real beta, thrust::device_vector<real>& y) const
  {
    std::shared_ptr<std::remove_pointer<cusparseDnVecDescr_t>::type> xdesc(
    detail::cuda_vector_description(const_cast<thrust::device_vector<real>&>(x)),
    backend::detail::cuda_deleter());
    std::shared_ptr<std::remove_pointer<cusparseDnVecDescr_t>::type> ydesc(
    detail::cuda_vector_description(y),
    backend::detail::cuda_deleter());

    size_t buf_size;
    ARCCORE_ALINA_CALL_CUDA(
    cusparseSpMV_bufferSize(handle,
                            CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &alpha,
                            desc.get(),
                            xdesc.get(),
                            &beta,
                            ydesc.get(),
                            detail::cuda_datatype<real>(),
                            CUSPARSE_SPMV_CSR_ALG1,
                            &buf_size));

    if (buf.size() < buf_size)
      buf.resize(buf_size);

    ARCCORE_ALINA_CALL_CUDA(
    cusparseSpMV(handle,
                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                 &alpha,
                 desc.get(),
                 xdesc.get(),
                 &beta,
                 ydesc.get(),
                 detail::cuda_datatype<real>(),
                 CUSPARSE_SPMV_CSR_ALG1,
                 thrust::raw_pointer_cast(&buf[0])));
  }

  size_t nbRow() const { return nrows; }
  size_t nbColumn() const { return ncols; }
  size_t nbNonZero() const { return nnz; }

  size_t rows() const { return nrows; }
  size_t cols() const { return ncols; }
  size_t nonzeros() const { return nnz; }
  size_t bytes() const
  {
    return sizeof(int) * (nrows + 1) +
    sizeof(int) * nnz +
    sizeof(real) * nnz;
  }

 public:

  size_t nrows, ncols, nnz;

  cusparseHandle_t handle;

  std::shared_ptr<std::remove_pointer<cusparseSpMatDescr_t>::type> desc;

  thrust::device_vector<int> ptr;
  thrust::device_vector<int> col;
  thrust::device_vector<real> val;

  mutable thrust::device_vector<char> buf;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/// CUDA backend.
/**
 * Uses CUSPARSE for matrix operations and Thrust for vector operations.
 *
 * \param real Value type.
 * \ingroup backends
 */
template <typename real, class DirectSolver = solver::cuda_skyline_lu<real>>
struct cuda
{
  static_assert(std::is_same<real, float>::value ||
                std::is_same<real, double>::value,
                "Unsupported value type for cuda backend");

  typedef real value_type;
  typedef ptrdiff_t col_type;
  typedef ptrdiff_t ptr_type;
  typedef cuda_matrix<real> matrix;
  typedef thrust::device_vector<real> vector;
  typedef thrust::device_vector<real> matrix_diagonal;
  typedef DirectSolver direct_solver;

  struct provides_row_iterator : std::false_type
  {};

  /// Backend parameters.
  struct params
  {
    /// CUSPARSE handle.
    cusparseHandle_t cusparse_handle = nullptr;

    params(cusparseHandle_t handle = nullptr)
    : cusparse_handle(handle)
    {}

    params(const PropertyTree& p)
    //: ARCCORE_ALINA_PARAMS_IMPORT_VALUE(p, cusparse_handle)
    {
      //check_params(p, { "cusparse_handle" });
    }

    void get(PropertyTree& p, const std::string& path) const
    {
      //ARCCORE_ALINA_PARAMS_EXPORT_VALUE(p, path, cusparse_handle);
    }
  };

  static std::string name() { return "cuda"; }

  /// Copy matrix from builtin backend.
  static std::shared_ptr<matrix>
  copy_matrix(std::shared_ptr<typename BuiltinBackend<real>::matrix> A, const params& prm)
  {
    return std::make_shared<matrix>(backend::nbRow(*A), backend::nbColumn(*A),
                                    A->ptr, A->col, A->val, prm.cusparse_handle);
  }

  /// Copy vector from builtin backend.
  static std::shared_ptr<vector>
  copy_vector(typename BuiltinBackend<real>::vector const& x, const params&)
  {
    return std::make_shared<vector>(x.data(), x.data() + x.size());
  }

  /// Copy vector from builtin backend.
  static std::shared_ptr<vector>
  copy_vector(std::shared_ptr<typename BuiltinBackend<real>::vector> x, const params& prm)
  {
    return copy_vector(*x, prm);
  }

  /// Create vector of the specified size.
  static std::shared_ptr<vector>
  create_vector(size_t size, const params&)
  {
    return std::make_shared<vector>(size);
  }

  /// Create direct solver for coarse level
  static std::shared_ptr<direct_solver>
  create_solver(std::shared_ptr<typename BuiltinBackend<real>::matrix> A, const params& prm)
  {
    return std::make_shared<direct_solver>(A, prm);
  }

  struct gather
  {
    thrust::device_vector<ptrdiff_t> I;
    mutable thrust::device_vector<value_type> T;

    gather(size_t src_size, const std::vector<ptrdiff_t>& I, const params&)
    : I(I)
    , T(I.size())
    {}

    void operator()(const vector& src, vector& dst) const
    {
      thrust::gather(I.begin(), I.end(), src.begin(), dst.begin());
    }

    void operator()(const vector& vec, std::vector<value_type>& vals) const
    {
      thrust::gather(I.begin(), I.end(), vec.begin(), T.begin());
      thrust::copy(T.begin(), T.end(), vals.begin());
    }
  };

  struct scatter
  {
    thrust::device_vector<ptrdiff_t> I;

    scatter(size_t size, const std::vector<ptrdiff_t>& I, const params&)
    : I(I)
    {}

    void operator()(const vector& src, vector& dst) const
    {
      thrust::scatter(src.begin(), src.end(), I.begin(), dst.begin());
    }
  };
};

//---------------------------------------------------------------------------
// Backend interface implementation
//---------------------------------------------------------------------------
template <typename V>
struct bytes_impl<thrust::device_vector<V>>
{
  static size_t get(const thrust::device_vector<V>& v)
  {
    return v.size() * sizeof(V);
  }
};

template <typename Alpha, typename Beta, typename V>
struct spmv_impl<Alpha, cuda_matrix<V>, thrust::device_vector<V>,
                 Beta, thrust::device_vector<V>>
{
  typedef cuda_matrix<V> matrix;
  typedef thrust::device_vector<V> vector;

  static void apply(Alpha alpha, const matrix& A, const vector& x,
                    Beta beta, vector& y)
  {
    A.spmv(alpha, x, beta, y);
  }
};

template <typename V>
struct residual_impl<cuda_matrix<V>,
                     thrust::device_vector<V>,
                     thrust::device_vector<V>,
                     thrust::device_vector<V>>
{
  typedef cuda_matrix<V> matrix;
  typedef thrust::device_vector<V> vector;

  static void apply(const vector& rhs, const matrix& A, const vector& x,
                    vector& r)
  {
    thrust::copy(rhs.begin(), rhs.end(), r.begin());
    A.spmv(-1, x, 1, r);
  }
};

template <typename V>
struct clear_impl<thrust::device_vector<V>>
{
  typedef thrust::device_vector<V> vector;

  static void apply(vector& x)
  {
    thrust::fill(x.begin(), x.end(), V());
  }
};

template <class V, class T>
struct copy_impl<V, thrust::device_vector<T>>
{
  static void apply(const V& x, thrust::device_vector<T>& y)
  {
    thrust::copy(x.begin(), x.end(), y.begin());
  }
};

template <class T, class V>
struct copy_impl<thrust::device_vector<T>, V>
{
  static void apply(const thrust::device_vector<T>& x, V& y)
  {
    thrust::copy(x.begin(), x.end(), y.begin());
  }
};

template <class T1, class T2>
struct copy_impl<thrust::device_vector<T1>, thrust::device_vector<T2>>
{
  static void apply(const thrust::device_vector<T1>& x, thrust::device_vector<T2>& y)
  {
    thrust::copy(x.begin(), x.end(), y.begin());
  }
};

template <typename V>
struct inner_product_impl<thrust::device_vector<V>,
                          thrust::device_vector<V>>
{
  typedef thrust::device_vector<V> vector;

  static V get(const vector& x, const vector& y)
  {
    return thrust::inner_product(x.begin(), x.end(), y.begin(), V());
  }
};

template <typename A, typename B, typename V>
struct axpby_impl<A, thrust::device_vector<V>,
                  B, thrust::device_vector<V>>
{
  typedef thrust::device_vector<V> vector;

  struct functor
  {
    A a;
    B b;
    functor(A a, B b)
    : a(a)
    , b(b)
    {}

    template <class Tuple>
    __host__ __device__ void operator()(Tuple t) const
    {
      using thrust::get;

      if (b)
        get<1>(t) = a * get<0>(t) + b * get<1>(t);
      else
        get<1>(t) = a * get<0>(t);
    }
  };

  static void apply(A a, const vector& x, B b, vector& y)
  {
    thrust::for_each(thrust::make_zip_iterator(
                     thrust::make_tuple(x.begin(), y.begin())),
                     thrust::make_zip_iterator(thrust::make_tuple(x.end(), y.end())),
                     functor(a, b));
  }
};

template <typename A, typename B, typename C, typename V>
struct axpbypcz_impl<A, thrust::device_vector<V>,
                     B, thrust::device_vector<V>,
                     C, thrust::device_vector<V>>
{
  typedef thrust::device_vector<V> vector;

  struct functor
  {
    A a;
    B b;
    C c;

    functor(A a, B b, C c)
    : a(a)
    , b(b)
    , c(c)
    {}

    template <class Tuple>
    __host__ __device__ void operator()(Tuple t) const
    {
      using thrust::get;

      if (c)
        get<2>(t) = a * get<0>(t) + b * get<1>(t) + c * get<2>(t);
      else
        get<2>(t) = a * get<0>(t) + b * get<1>(t);
    }
  };

  static void apply(A a, const vector& x,
                    B b, const vector& y,
                    C c, vector& z)
  {
    thrust::for_each(
    thrust::make_zip_iterator(
    thrust::make_tuple(
    x.begin(), y.begin(), z.begin())),
    thrust::make_zip_iterator(
    thrust::make_tuple(
    x.end(), y.end(), z.end())),
    functor(a, b, c));
  }
};

template <typename A, typename B, typename V>
struct vmul_impl<A, thrust::device_vector<V>, thrust::device_vector<V>,
                 B, thrust::device_vector<V>>
{
  typedef thrust::device_vector<V> vector;

  struct functor
  {
    A a;
    B b;
    functor(A a, B b)
    : a(a)
    , b(b)
    {}

    template <class Tuple>
    __host__ __device__ void operator()(Tuple t) const
    {
      using thrust::get;

      if (b)
        get<2>(t) = a * get<0>(t) * get<1>(t) + b * get<2>(t);
      else
        get<2>(t) = a * get<0>(t) * get<1>(t);
    }
  };

  static void apply(A a, const vector& x, const vector& y, B b, vector& z)
  {
    thrust::for_each(thrust::make_zip_iterator(
                     thrust::make_tuple(
                     x.begin(), y.begin(), z.begin())),
                     thrust::make_zip_iterator(
                     thrust::make_tuple(
                     x.end(), y.end(), z.end())),
                     functor(a, b));
  }
};

class cuda_event
{
 public:

  cuda_event()
  : e(create_event(), backend::detail::cuda_deleter())
  {}

  float operator-(cuda_event tic) const
  {
    float delta;
    cudaEventSynchronize(e.get());
    cudaEventElapsedTime(&delta, tic.e.get(), e.get());
    return delta / 1000.0f;
  }

 private:

  std::shared_ptr<std::remove_pointer<cudaEvent_t>::type> e;

  static cudaEvent_t create_event()
  {
    cudaEvent_t e;
    cudaEventCreate(&e);
    cudaEventRecord(e, 0);
    return e;
  }
};

struct cuda_clock
{
  typedef cuda_event value_type;

  static const char* units() { return "s"; }

  cuda_event current() const
  {
    return cuda_event();
  }
};

} // namespace Arcane::Alina::backend

#endif
