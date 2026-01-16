/*
 * Copyright 2020 IFPEN-CEA
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 */
/*
 * ILU0Preconditioner.h
 *
 *  Created on: Sep 20, 2010
 *      Author: gratienj
 */

#pragma once
#if defined (ALIEN_USE_EIGEN3) && !defined(ALIEN_USE_SYCL)
#include <Eigen/Dense>
#endif
#include <alien/handlers/scalar/CSRModifierViewT.h>

namespace Alien
{

template <typename MatrixT, typename VectorT>
class LUFactorisationAlgo
{
 public:
  // clang-format off
  typedef MatrixT                          MatrixType;
  typedef VectorT                          VectorType;
  typedef typename MatrixType::ProfileType ProfileType ;
  typedef typename MatrixType::ValueType   ValueType;
  typedef typename MatrixType::TagType     TagType ;
  // clang-format on

  LUFactorisationAlgo()
  {}

  virtual ~LUFactorisationAlgo()
  {}

  template <typename AlgebraT>
  void baseInit(AlgebraT& algebra, MatrixT const& matrix)
  {
    m_is_parallel = matrix.isParallel();
    m_alloc_size = matrix.getAllocSize();
    if constexpr (requires{matrix.blockSize();})
      m_block_size = matrix.blockSize();
    else
      m_block_size = 1 ;
    m_distribution = matrix.distribution();
    m_lu_matrix.reset(matrix.cloneTo(nullptr));
    m_profile = &m_lu_matrix->getProfile();
    m_work.resize(m_alloc_size);
    m_work.assign(m_work.size(), -1);
    algebra.allocate(AlgebraT::resource(matrix), m_x);
  }

  template <typename AlgebraT>
  void init(AlgebraT& algebra, MatrixT const& matrix)
  {
    baseInit(algebra, matrix);
    if(m_block_size==1)
      factorize(*m_lu_matrix);
    else
#if defined (ALIEN_USE_EIGEN3) && !defined (ALIEN_USE_SYCL)
      blockFactorize(*m_lu_matrix);
#else
    throw Arccore::FatalErrorException(
            A_FUNCINFO, "Eigen is required for BlockILU factorization");
#endif
    m_work.clear();
  }

  void factorize(MatrixT& matrix, bool bjacobi = true)
  {
    /*
       *
         For i = 1, . . . ,N Do:
            For k = 1, . . . , i - 1 and if (i, k) 2 NZ(A) Do:
                Compute aik := aik/akk
                For j = k + 1, . . . and if (i, j) 2 NZ(A), Do:
                   compute aij := aij - aik.ak,j.
                EndFor
            EndFor
         EndFor
       *
       */
    m_bjacobi = bjacobi;
    CSRModifierViewT<MatrixT> modifier(matrix);

    // clang-format off
    auto nrows  = modifier.nrows() ;
    auto kcol   = modifier.kcol() ;
    auto dcol   = modifier.dcol() ;
    auto cols   = modifier.cols() ;
    auto values = modifier.data() ;
    // clang-format on
    if (m_is_parallel) {
      auto& local_row_size = matrix.getDistStructInfo().m_local_row_size;
      if (m_bjacobi) {
        for (std::size_t irow = 1; irow < nrows; ++irow) // i=1->nrow
        {
          for (int k = kcol[irow]; k < dcol[irow]; ++k) // k=1 ->i-1
          {
            int krow = cols[k];
            ValueType aik = values[k] / values[dcol[krow]]; // aik = aik/akk
            values[k] = aik;
            for (int l = kcol[krow]; l < kcol[krow] + local_row_size[krow]; ++l)
              m_work[cols[l]] = l;
            for (int j = k + 1; j < kcol[krow] + local_row_size[irow]; ++j) // j=k+1->n
            {
              int jcol = cols[j];
              int kj = m_work[jcol];
              if (kj != -1) {
                values[j] -= aik * values[kj]; // aij = aij - aik*akj
              }
            }
            for (int l = kcol[krow]; l < kcol[krow] + local_row_size[krow]; ++l)
              m_work[cols[l]] = -1;
          }
        }
      }
      else {
        typename LUSendRecvTraits<TagType>::matrix_op_type op(matrix, m_distribution, m_work);
        op.recvLowerNeighbLUData(values);
        int first_upper_ghost_index = matrix.getDistStructInfo().m_first_upper_ghost_index;
        for (std::size_t irow = 1; irow < nrows; ++irow) // i=1->nrow
        {
          for (int k = kcol[irow]; k < dcol[irow]; ++k) // k=1 ->i-1
          {
            int krow = cols[k];
            ValueType aik = values[k] / values[dcol[krow]]; // aik = aik/akk
            values[k] = aik;
            for (int l = kcol[krow]; l < kcol[krow + 1]; ++l)
              m_work[cols[l]] = l;
            for (int j = k + 1; j < kcol[irow] + local_row_size[irow]; ++j) // j=k+1->n
            {
              int jcol = cols[j];
              int kj = m_work[jcol];
              if (kj != -1) {
                values[j] -= aik * values[kj]; // aij = aij - aik*akj
              }
            }
            for (int j = kcol[irow] + local_row_size[irow]; j < kcol[irow + 1]; ++j) // j=k+1->n
            {
              int jcol = cols[j];
              int kj = m_work[jcol];
              if ((kj != -1) && (jcol >= first_upper_ghost_index)) {
                values[j] -= aik * values[kj]; // aij = aij - aik*akj
              }
            }
            for (int l = kcol[krow]; l < kcol[krow + 1]; ++l)
              m_work[cols[l]] = -1;
          }
        }
        op.sendUpperNeighbLUData(values);
      }
    }
    else {
      for (std::size_t irow = 1; irow < nrows; ++irow) // i=1->nrow
      {
        for (int k = kcol[irow]; k < dcol[irow]; ++k) // k=1 ->i-1
        {
          int krow = cols[k];
          ValueType aik = values[k] / values[dcol[krow]]; // aik = aik/akk
          values[k] = aik;
          for (int l = kcol[krow]; l < kcol[krow + 1]; ++l)
            m_work[cols[l]] = l;
          for (int j = k + 1; j < kcol[irow + 1]; ++j) // j=k+1->n
          {
            int jcol = cols[j];
            int kj = m_work[jcol];
            if (kj != -1) {
              values[j] -= aik * values[kj]; // aij = aij - aik*akj
            }
          }
          for (int l = kcol[krow]; l < kcol[krow + 1]; ++l)
            m_work[cols[l]] = -1;
        }
      }
    }
  }

  void solveL(ValueType const* y, ValueType* x) const
  {
    CSRConstViewT<MatrixT> view(*m_lu_matrix);
    // clang-format off
    auto nrows  = view.nrows() ;
    auto kcol   = view.kcol() ;
    auto dcol   = view.dcol() ;
    auto cols   = view.cols() ;
    auto values = view.data() ;
    // clang-format on

    for (std::size_t irow = 0; irow < nrows; ++irow) {
      ValueType val = y[irow];
      for (int k = kcol[irow]; k < dcol[irow]; ++k)
        val -= values[k] * x[cols[k]];
      x[irow] = val;
    }
  }

  void solveU(ValueType const* y, ValueType* x) const
  {
    CSRConstViewT<MatrixT> view(*m_lu_matrix);
    // clang-format off
    auto nrows  = view.nrows() ;
    auto kcol   = view.kcol() ;
    auto dcol   = view.dcol() ;
    auto cols   = view.cols() ;
    auto values = view.data() ;
    // clang-format on
    if (m_is_parallel) {
      auto& local_row_size = m_lu_matrix->getDistStructInfo().m_local_row_size;
      for (int irow = (int)nrows - 1; irow > -1; --irow) {
        int dk = dcol[irow];
        ValueType val = y[irow];
        for (int k = dk + 1; k < kcol[irow] + local_row_size[irow]; ++k) {
          val -= values[k] * x[cols[k]];
        }
        x[irow] = val / values[dk];
      }
    }
    else {
      for (int irow = (int)nrows - 1; irow > -1; --irow) {
        int dk = dcol[irow];
        ValueType val = y[irow];
        for (int k = dk + 1; k < kcol[irow + 1]; ++k) {
          val -= values[k] * x[cols[k]];
        }
        x[irow] = val / values[dk];
      }
    }
  }

#if defined (ALIEN_USE_EIGEN3) && !defined (ALIEN_USE_SYCL)
  inline auto inv(Eigen::Map<Eigen::Matrix<ValueType,Eigen::Dynamic,Eigen::Dynamic>>const & block) const
  {
    assert(block.determinant()!=0) ;
    return block.inverse() ;
  }

  void blockFactorize(MatrixT& matrix, bool bjacobi = true)
  {
    /*
       *
         For i = 1, . . . ,N Do:
            For k = 1, . . . , i - 1 and if (i, k) 2 NZ(A) Do:
                Compute aik := aik/akk
                For j = k + 1, . . . and if (i, j) 2 NZ(A), Do:
                   compute aij := aij - aik.ak,j.
                EndFor
            EndFor
         EndFor
       *
       */
    using namespace Eigen;
    using Block2D     = Eigen::Matrix<ValueType,Eigen::Dynamic,Eigen::Dynamic> ;
    using Block2DView = Eigen::Map<Block2D> ;

    int N = m_block_size;
    int N2 = N*N;

    m_bjacobi = bjacobi;
    CSRModifierViewT<MatrixT> modifier(matrix);

    // clang-format off
    auto nrows  = modifier.nrows() ;
    auto kcol   = modifier.kcol() ;
    auto dcol   = modifier.dcol() ;
    auto cols   = modifier.cols() ;
    auto values = modifier.data() ;

    Block2D aik(N,N) ;
    // clang-format on
    if (m_is_parallel) {
      auto& local_row_size = matrix.getDistStructInfo().m_local_row_size;
      if (m_bjacobi) {
        for (std::size_t irow = 1; irow < nrows; ++irow) // i=1->nrow
        {
          for (int k = kcol[irow]; k < dcol[irow]; ++k) // k=1 ->i-1
          {
            int krow = cols[k];
            //ValueType aik = values[k] / values[dcol[krow]]; // aik = aik/akk
            //values[k] = aik;
            aik = Block2DView(values+k*N2,N,N) * inv(Block2DView(values+dcol[krow]*N2,N,N)) ;
            Block2DView(values+k*N2,N,N) = aik ;
            for (int l = kcol[krow]; l < kcol[krow] + local_row_size[krow]; ++l)
              m_work[cols[l]] = l;
            for (int j = k + 1; j < kcol[krow] + local_row_size[irow]; ++j) // j=k+1->n
            {
              int jcol = cols[j];
              int kj = m_work[jcol];
              if (kj != -1) {
                //values[j] -= aik * values[kj]; // aij = aij - aik*akj
                Block2DView(values+j*N2,N,N) -= aik * Block2DView(values+kj*N2,N,N) ;
              }
            }
            for (int l = kcol[krow]; l < kcol[krow] + local_row_size[krow]; ++l)
              m_work[cols[l]] = -1;
          }
        }
      }
      else {
        typename LUSendRecvTraits<TagType>::matrix_op_type op(matrix, m_distribution, m_work);
        op.recvLowerNeighbLUData(values);
        int first_upper_ghost_index = matrix.getDistStructInfo().m_first_upper_ghost_index;
        for (std::size_t irow = 1; irow < nrows; ++irow) // i=1->nrow
        {
          for (int k = kcol[irow]; k < dcol[irow]; ++k) // k=1 ->i-1
          {
            int krow = cols[k];
            //ValueType aik = values[k] / values[dcol[krow]]; // aik = aik/akk
            //values[k] = aik;
            aik = Block2DView(values+k*N2,N,N) * inv(Block2DView(values+dcol[krow]*N2,N,N)) ;
            for (int l = kcol[krow]; l < kcol[krow + 1]; ++l)
              m_work[cols[l]] = l;
            for (int j = k + 1; j < kcol[irow] + local_row_size[irow]; ++j) // j=k+1->n
            {
              int jcol = cols[j];
              int kj = m_work[jcol];
              if (kj != -1) {
                //values[j] -= aik * values[kj]; // aij = aij - aik*akj
                Block2DView(values+j*N2,N,N) -= aik * Block2DView(values+kj*N2,N,N) ;
              }
            }
            for (int j = kcol[irow] + local_row_size[irow]; j < kcol[irow + 1]; ++j) // j=k+1->n
            {
              int jcol = cols[j];
              int kj = m_work[jcol];
              if ((kj != -1) && (jcol >= first_upper_ghost_index)) {
                //values[j] -= aik * values[kj]; // aij = aij - aik*akj
                Block2DView(values+j*N2,N,N) -= aik * Block2DView(values+kj*N2,N,N) ;
              }
            }
            for (int l = kcol[krow]; l < kcol[krow + 1]; ++l)
              m_work[cols[l]] = -1;
          }
        }
        op.sendUpperNeighbLUData(values);
      }
    }
    else {
      for (std::size_t irow = 1; irow < nrows; ++irow) // i=1->nrow
      {
        for (int k = kcol[irow]; k < dcol[irow]; ++k) // k=1 ->i-1
        {
          int krow = cols[k];
          //ValueType aik = values[k] / values[dcol[krow]]; // aik = aik/akk
          //values[k] = aik;
          aik = Block2DView(values+k*N2,N,N) * inv(Block2DView(values+dcol[krow]*N2,N,N)) ;
          Block2DView(values+k*N2,N,N) = aik ;
          for (int l = kcol[krow]; l < kcol[krow + 1]; ++l)
            m_work[cols[l]] = l;
          for (int j = k + 1; j < kcol[irow + 1]; ++j) // j=k+1->n
          {
            int jcol = cols[j];
            int kj = m_work[jcol];
            if (kj != -1) {
              //values[j] -= aik * values[kj]; // aij = aij - aik*akj
              Block2DView(values+j*N2,N,N) -= aik * Block2DView(values+kj*N2,N,N) ;
            }
          }
          for (int l = kcol[krow]; l < kcol[krow + 1]; ++l)
            m_work[cols[l]] = -1;
        }
      }
    }
  }

  void blockSolveL(ValueType const* y, ValueType* x) const
  {
    using namespace Eigen;
    using Block2D     = Eigen::Matrix<ValueType,Eigen::Dynamic,Eigen::Dynamic> ;
    using Block2DView = Eigen::Map<Block2D> ;
    using Block1D     = Eigen::Matrix<ValueType,Dynamic,1> ;
    using Block1DView = Eigen::Map<Block1D> ;

    int N = m_block_size ;
    int N2 = N*N;


    Block1D val(N);

    CSRConstViewT<MatrixT> view(*m_lu_matrix);
    // clang-format off
    auto nrows  = view.nrows() ;
    auto kcol   = view.kcol() ;
    auto dcol   = view.dcol() ;
    auto cols   = view.cols() ;
    auto values = view.data() ;
    // clang-format on

    for (std::size_t irow = 0; irow < nrows; ++irow) {
      //ValueType val = y[irow];
      val = Block1DView(const_cast<ValueType*>(y+irow*N),N) ;
      for (int k = kcol[irow]; k < dcol[irow]; ++k)
      {
        //val -= values[k] * x[cols[k]];
        val -= Block2DView(const_cast<ValueType*>(values+k*N2),N,N) * Block1DView(x+cols[k]*N,N) ;
      }
      //x[irow] = val;
      Block1DView(x+irow*N,N) = val ;
    }
  }

  void blockSolveU(ValueType const* y, ValueType* x) const
  {
    using namespace Eigen;
    using Block2D     = Eigen::Matrix<ValueType,Eigen::Dynamic,Eigen::Dynamic> ;
    using Block2DView = Eigen::Map<Block2D> ;
    using Block1D     = Eigen::Matrix<ValueType,Dynamic,1> ;
    using Block1DView = Eigen::Map<Block1D> ;

    int N = m_block_size;
    int N2 = N*N;

    Block1D val(N);

    CSRConstViewT<MatrixT> view(*m_lu_matrix);
    // clang-format off
    auto nrows  = view.nrows() ;
    auto kcol   = view.kcol() ;
    auto dcol   = view.dcol() ;
    auto cols   = view.cols() ;
    auto values = view.data() ;
    // clang-format on
    if (m_is_parallel) {
      auto& local_row_size = m_lu_matrix->getDistStructInfo().m_local_row_size;
      for (int irow = (int)nrows - 1; irow > -1; --irow) {
        int dk = dcol[irow];
        //ValueType val = y[irow];
        val = Block1DView(const_cast<ValueType*>(y+irow*N),N) ;
        for (int k = dk + 1; k < kcol[irow] + local_row_size[irow]; ++k) {
          //val -= values[k] * x[cols[k]];
          val -= Block2DView(const_cast<ValueType*>(values+k*N2),N,N) * Block1DView(x+cols[k]*N,N) ;
        }
        //x[irow] = val / values[dk];
        Block1DView(x+irow*N,N) = inv(Block2DView(const_cast<ValueType*>(values+dk*N2),N,N)) * val;
      }
    }
    else {
      for (int irow = (int)nrows - 1; irow > -1; --irow) {
        int dk = dcol[irow];
        //ValueType val = y[irow];
        val = Block1DView(const_cast<ValueType*>(y+irow*N),N) ;
        for (int k = dk + 1; k < kcol[irow + 1]; ++k) {
          //val -= values[k] * x[cols[k]];
          val -= Block2DView(const_cast<ValueType*>(values+k*N2),N,N) * Block1DView(x+cols[k]*N,N) ;
        }
        //x[irow] = val / values[dk];
        Block1DView(x+irow*N,N) = inv(Block2DView(const_cast<ValueType*>(values+dk*N2),N,N)) * val;
      }
    }
  }

#endif

  template <typename AlgebraT>
  void solve([[maybe_unused]] AlgebraT& algebra, VectorType const& y, VectorType& x) const
  {
    if(m_block_size==1)
    {
      //////////////////////////////////////////////////////////////////////////
      //
      //     L.X1 = Y
      //
      solveL(y.data(), m_x.data());

      //////////////////////////////////////////////////////////////////////////
      //
      //     U.X = X1
      //
      solveU(m_x.data(), x.data());
    }
    else
    {
#if defined(ALIEN_USE_EIGEN3) && !defined(ALIEN_USE_SYCL)
      //////////////////////////////////////////////////////////////////////////
      //
      //     L.X1 = Y
      //
      blockSolveL(y.data(), m_x.data());

      //////////////////////////////////////////////////////////////////////////
      //
      //     U.X = X1
      //
      blockSolveU(m_x.data(), x.data());
#else
      throw Arccore::FatalErrorException(
              A_FUNCINFO, "Eigen is required for BlockILU resolution");
#endif
    }
  }

  const MatrixType& getLUMatrix() const
  {
    return *m_lu_matrix;
  }

 protected:
  // clang-format off
  std::unique_ptr<MatrixType>   m_lu_matrix ;
  int                           m_block_size                  = 1;
  ProfileType const*            m_profile                     = nullptr;
  mutable VectorType            m_x ;

  MatrixDistribution            m_distribution ;
  std::vector<int>              m_work ;
  std::size_t                   m_alloc_size                  = 0 ;
  bool                          m_is_parallel                 = false ;
  bool                          m_bjacobi                     = false ;

  std::vector<int>                    m_send_lu_ibuffer ;
  std::vector<std::vector<ValueType>> m_send_lu_buffer ;
  // clang-format on
};

template <typename AlgebraT>
class ILU0Preconditioner
{
 public:
  // clang-format off
  typedef AlgebraT                         AlgebraType ;
  typedef typename AlgebraType::Matrix     MatrixType;
  typedef typename AlgebraType::Vector     VectorType;
  typedef typename MatrixType::ProfileType ProfileType ;
  typedef typename MatrixType::ValueType   ValueType;
  // clang-format on

  typedef LUFactorisationAlgo<MatrixType, VectorType> AlgoType;

  ILU0Preconditioner(AlgebraType& algebra, MatrixType const& matrix, ITraceMng* trace_mng = nullptr)
  : m_algebra(algebra)
  , m_matrix(matrix)
  , m_trace_mng(trace_mng)
  {
  }

  virtual ~ILU0Preconditioner(){};

  void init()
  {
    m_algo.init(m_algebra, m_matrix);
  }

  void solve(VectorType const& y, VectorType& x) const
  {
    m_algo.solve(m_algebra, y, x);
  }

  void solve(AlgebraType& algebra, VectorType const& y, VectorType& x) const
  {
    m_algo.solve(algebra, y, x);
  }

  void update()
  {
    // update value from m_matrix
  }

 private:
  // clang-format off
  AlgebraType&                  m_algebra ;
  MatrixType const&             m_matrix;
  AlgoType                      m_algo ;

  ITraceMng*                    m_trace_mng = nullptr ;
  // clang-format on
};

} // namespace Alien
