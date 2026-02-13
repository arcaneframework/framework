// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#pragma once

namespace Alien
{
template <typename MatrixT, typename VectorT>
class FLUFactorisationAlgo
: public LUFactorisationAlgo<MatrixT, VectorT>
{
 public:
  // clang-format off
  typedef LUFactorisationAlgo<MatrixT,VectorT> BaseType ;
  typedef MatrixT                              MatrixType;
  typedef VectorT                              VectorType;
  typedef typename MatrixType::ProfileType     ProfileType ;
  typedef typename MatrixType::ValueType       ValueType;
  typedef typename MatrixType::TagType         TagType ;
  // clang-format on

  FLUFactorisationAlgo()
  {}

  virtual ~FLUFactorisationAlgo()
  {}

  void setParameter(std::string param, int value)
  {
    if (param.compare("nb-factor-iter") == 0)
      m_nb_factorization_iter = value;
    else if (param.compare("nb-solver-iter") == 0)
      m_nb_solver_iter = value;
  }

  void setParameter(std::string param, double value)
  {
    if (param.compare("tol") == 0)
      m_tol = value;
  }

  template <typename AlgebraT>
  void init(AlgebraT& algebra, MatrixT const& matrix)
  {
    BaseType::baseInit(algebra, matrix);
    algebra.allocate(AlgebraT::resource(matrix), m_xk);

    if (m_nb_factorization_iter == 0) {
      if(this->m_block_size==1)
        BaseType::factorize(*this->m_lu_matrix, false);
      else
      {
        BaseType::blockFactorize(*this->m_lu_matrix);
      }
    }
    else {
      factorizeMultiIter(*this->m_lu_matrix);
    }
    this->m_work.clear();

    if(this->m_block_size==1)
    {
      algebra.allocate(AlgebraT::resource(matrix), m_inv_diag);
      algebra.assign(m_inv_diag, 1.);
      algebra.computeInvDiag(*this->m_lu_matrix, m_inv_diag);
    }
    else
    {
      m_inv_diag.setBlockSize(this->m_block_size*this->m_block_size) ;
      algebra.allocate(AlgebraT::resource(matrix), m_inv_diag);
      algebra.computeInvDiag(*this->m_lu_matrix, m_inv_diag);

    }

    if (MatrixType::on_host_only) {
      if (this->m_is_parallel) {
        // Need to manage ghost and communications
        this->m_x.resize((Integer)this->m_alloc_size);
        m_xk.resize((Integer)this->m_alloc_size);
      }
    }
  }

  ///////////////////////////////////////////////////////////////////////
  //
  // FACTORIZATION
  //
  void factorizeIter(std::size_t nrows,
                     int const* kcol,
                     int const* dcol,
                     int const* cols,
                     ValueType* values,
                     ValueType* values0)
  {
    /*
       *
         For i = 1, . . . ,N Do:
            For k = 1, . . . , i - 1 and if (i, k) 2 NZ(A) Do:
                Compute aik := aik/akj
                For j = k + 1, . . . and if (i, j) 2 NZ(A), Do:
                   compute aij := aij - aik.ak,j.
                EndFor
            EndFor
         EndFor
       *
       */

    for (std::size_t irow = 1; irow < nrows; ++irow) // i=1->nrow
    {
      _factorizeRow(irow, kcol, dcol, cols, values, values0);
    }
  }

  void factorizeMultiIter(MatrixT const& matrix)
  {
    CSRModifierViewT<MatrixT> modifier(*this->m_lu_matrix);

    // clang-format off
    auto nrows  = modifier.nrows() ;
    auto nnz    = modifier.nnz() ;
    auto kcol   = modifier.kcol() ;
    auto dcol   = modifier.dcol() ;
    auto cols   = modifier.cols() ;
    auto values = modifier.data() ;
    // clang-format on

    if (this->m_lu_matrix->isParallel()) {
      typename LUSendRecvTraits<TagType>::matrix_op_type op(*this->m_lu_matrix, this->m_distribution, this->m_work);
      for (int iter = 0; iter < m_nb_factorization_iter; ++iter) {
        auto matrix_values = CSRConstViewT<MatrixT>(matrix).data();
        std::vector<ValueType> guest_values(nnz);
        std::copy(values, values + nnz, guest_values.data());
        std::copy(matrix_values, matrix_values + nnz, values);
        op.recvLowerNeighbLUData(values);
        op.sendUpperNeighbLUData(guest_values.data());
        factorizeIter(nrows, kcol, dcol, cols, values, guest_values.data());
      }
    }
    else {
      for (int iter = 0; iter < m_nb_factorization_iter; ++iter) {
        auto matrix_values = CSRConstViewT<MatrixT>(matrix).data();
        std::vector<ValueType> guest_values(nnz);
        std::copy(values, values + nnz, guest_values.data());
        std::copy(matrix_values, matrix_values + nnz, values);
        factorizeIter(nrows, kcol, dcol, cols, values, guest_values.data());
      }
    }
  }

  ///////////////////////////////////////////////////////////////////////
  //
  // TRIANGULAR RESOLUTION
  //
  template <typename AlgebraT>
  void solveL(AlgebraT& algebra,
              VectorType const& y,
              VectorType& x,
              VectorType& xk) const
  {
#ifdef ALIEN_USE_PERF_TIMER
    typename MatrixType::SentryType sentry(this->m_lu_matrix->timer(), "SolveL");
#endif
    if constexpr (MatrixType::on_host_only)
    {
      CSRConstViewT<MatrixT> view(*this->m_lu_matrix);
      // clang-format off
      auto nrows  = view.nrows();
      auto kcol   = view.kcol();
      auto dcol   = view.dcol();
      auto cols   = view.cols();
      auto values = view.data();

      auto y_ptr  = y.data();
      auto x_ptr  = x.data();
      auto xk_ptr = xk.data();
      // clang-format on

      std::copy(x_ptr, x_ptr + nrows, xk_ptr);
      if (this->m_is_parallel)
      {
        typedef typename LUSendRecvTraits<TagType>::vector_op_type SendRecvOpType;
        SendRecvOpType op(xk_ptr,
                          this->m_lu_matrix->getDistStructInfo().m_send_info,
                          this->m_lu_matrix->getSendPolicy(),
                          xk_ptr,
                          this->m_lu_matrix->getDistStructInfo().m_recv_info,
                          this->m_lu_matrix->getRecvPolicy(),
                          this->m_lu_matrix->getParallelMng(),
                          nullptr);

        auto& local_row_size = this->m_lu_matrix->getDistStructInfo().m_local_row_size;
        int first_upper_ghost_index = this->m_lu_matrix->getDistStructInfo().m_first_upper_ghost_index;
        op.lowerRecv();
        op.upperSend();
        for (std::size_t irow = 0; irow < nrows; ++irow) {
          ValueType val = y_ptr[irow];
          for (int k = kcol[irow]; k < dcol[irow]; ++k)
            val -= values[k] * xk_ptr[cols[k]];
          for (int k = kcol[irow] + local_row_size[irow]; k < kcol[irow + 1]; ++k) {
            if (cols[k] < first_upper_ghost_index)
              val -= values[k] * xk_ptr[cols[k]];
          }
          x_ptr[irow] = val;
        }
      }
      else
      {
        for (std::size_t irow = 0; irow < nrows; ++irow) {
          ValueType val = y_ptr[irow];
          for (int k = kcol[irow]; k < dcol[irow]; ++k)
            val -= values[k] * xk_ptr[cols[k]];
          x_ptr[irow] = val;
        }
      }
    }
    else
    {
      algebra.copy(x, xk);
      algebra.copy(y, x);
      algebra.addLMult(-1, *this->m_lu_matrix, xk, x);
    }
  }

  template <typename AlgebraT>
  void blockSolveL(AlgebraT& algebra,
              VectorType const& y,
              VectorType& x,
              VectorType& xk) const
  {
#ifdef ALIEN_USE_PERF_TIMER
    typename MatrixType::SentryType sentry(this->m_lu_matrix->timer(), "SolveL");
#endif
    if constexpr (MatrixType::on_host_only)
    {
#if defined (ALIEN_USE_EIGEN3)  && !defined(EIGEN3_DISABLED)
      using namespace Eigen;
      using Block2D     = Eigen::Matrix<ValueType,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> ;
      using Block2DView = Eigen::Map<Block2D> ;
      using Block1D     = Eigen::Matrix<ValueType,Dynamic,1> ;
      using Block1DView = Eigen::Map<Block1D> ;

      int N   = this->m_block_size ;
      int NxN = N*N;

      Block1D val(N);
      CSRConstViewT<MatrixT> view(*this->m_lu_matrix);
      // clang-format off
      auto nrows  = view.nrows();
      auto kcol   = view.kcol();
      auto dcol   = view.dcol();
      auto cols   = view.cols();
      auto values = view.data();

      auto y_ptr  = y.data();
      auto x_ptr  = x.data();
      auto xk_ptr = xk.data();
      // clang-format on

      std::copy(x_ptr, x_ptr + nrows*N, xk_ptr);
      if (this->m_is_parallel)
      {
        typedef typename LUSendRecvTraits<TagType>::vector_op_type SendRecvOpType;
        SendRecvOpType op(xk_ptr,
                          this->m_lu_matrix->getDistStructInfo().m_send_info,
                          this->m_lu_matrix->getSendPolicy(),
                          xk_ptr,
                          this->m_lu_matrix->getDistStructInfo().m_recv_info,
                          this->m_lu_matrix->getRecvPolicy(),
                          this->m_lu_matrix->getParallelMng(),
                          nullptr);

        auto& local_row_size = this->m_lu_matrix->getDistStructInfo().m_local_row_size;
        int first_upper_ghost_index = this->m_lu_matrix->getDistStructInfo().m_first_upper_ghost_index;
        op.lowerRecv();
        op.upperSend();
        for (std::size_t irow = 0; irow < nrows; ++irow)
        {
          //ValueType val = y_ptr[irow];
          val = Block1DView(const_cast<ValueType*>(y_ptr+irow*N),N) ;
          for (int k = kcol[irow]; k < dcol[irow]; ++k)
          {
            //val -= values[k] * xk_ptr[cols[k]];
            val -= Block2DView(const_cast<ValueType*>(values+k*NxN),N,N) * Block1DView(xk_ptr+cols[k]*N,N) ;
          }
          for (int k = kcol[irow] + local_row_size[irow]; k < kcol[irow + 1]; ++k)
          {
            if (cols[k] < first_upper_ghost_index)
            {
              //val -= values[k] * xk_ptr[cols[k]];
              val -= Block2DView(const_cast<ValueType*>(values+k*NxN),N,N) * Block1DView(xk_ptr+cols[k]*N,N) ;
            }
          }
          //x_ptr[irow] = val;
          Block1DView(x_ptr+irow*N,N) = val ;
        }
      }
      else
      {
        /*
        for (std::size_t irow = 0; irow < nrows; ++irow)
        {
          std::cout<<"XK["<<irow<<"]=\n";
          for(int i=0;i<N;++i)
            std::cout<<xk_ptr[irow*N+i]<<std::endl;
        }*/
        for (std::size_t irow = 0; irow < nrows; ++irow)
        {
          //ValueType val = y_ptr[irow];
          val = Block1DView(const_cast<ValueType*>(y_ptr+irow*N),N) ;
          //std::cout<<"LSOLVE MAT["<<irow<<"]:"<<std::endl;
          for (int k = kcol[irow]; k < dcol[irow]; ++k)
          {
            //val -= values[k] * xk_ptr[cols[k]];
            val -= Block2DView(const_cast<ValueType*>(values+k*NxN),N,N) * Block1DView(xk_ptr+cols[k]*N,N) ;
            //std::cout<<Block2DView(const_cast<ValueType*>(values+k*NxN),N,N)<<std::endl;
          }
          //x_ptr[irow] = val;
          Block1DView(x_ptr+irow*N,N) = val ;
          //std::cout<<"Y =\n"<<val<<std::endl ;
        }
      }
#endif
    }
    else
    {
      algebra.copy(x, xk);
      algebra.copy(y, x);
      algebra.addLMult(-1, *this->m_lu_matrix, xk, x);
    }
  }

  template <typename AlgebraT>
  void solveU(AlgebraT& algebra, VectorType const& y, VectorType& x, VectorType& xk) const
  {
#ifdef ALIEN_USE_PERF_TIMER
    typename MatrixType::SentryType sentry(this->m_lu_matrix->timer(), "SolveU");
#endif
    if constexpr (MatrixType::on_host_only)
    {
      CSRConstViewT<MatrixT> view(*this->m_lu_matrix);
      // clang-format off
      auto nrows  = view.nrows();
      auto kcol   = view.kcol();
      auto dcol   = view.dcol();
      auto cols   = view.cols();
      auto values = view.data();

      auto y_ptr  = y.data();
      auto x_ptr  = x.data();
      auto xk_ptr = xk.data();
      // clang-format on

      std::copy(x_ptr, x_ptr + nrows, xk_ptr);
      if (this->m_is_parallel) {
        auto& local_row_size = this->m_lu_matrix->getDistStructInfo().m_local_row_size;
        int first_upper_ghost_index = this->m_lu_matrix->getDistStructInfo().m_first_upper_ghost_index;

        typedef typename LUSendRecvTraits<TagType>::vector_op_type SendRecvOpType;
        SendRecvOpType op(xk_ptr,
                          this->m_lu_matrix->getDistStructInfo().m_send_info,
                          this->m_lu_matrix->getSendPolicy(),
                          xk_ptr,
                          this->m_lu_matrix->getDistStructInfo().m_recv_info,
                          this->m_lu_matrix->getRecvPolicy(),
                          this->m_lu_matrix->getParallelMng(),
                          nullptr);
        op.upperRecv();
        op.lowerSend();
        for (std::size_t irow = 0; irow < nrows; ++irow) {
          int dk = dcol[irow];
          ValueType val = y_ptr[irow];
          for (int k = dk + 1; k < kcol[irow] + local_row_size[irow]; ++k) {
            val -= values[k] * xk_ptr[cols[k]];
          }
          for (int k = kcol[irow] + local_row_size[irow]; k < kcol[irow + 1]; ++k) {
            if (cols[k] >= first_upper_ghost_index)
              val -= values[k] * xk_ptr[cols[k]];
          }
          val = val / values[dk];
          x_ptr[irow] = val;
        }
      }
      else {
        for (std::size_t irow = 0; irow < nrows; ++irow) {
          int dk = dcol[irow];
          ValueType val = y_ptr[irow];
          for (int k = dk + 1; k < kcol[irow + 1]; ++k) {
            val -= values[k] * xk_ptr[cols[k]];
          }
          val = val / values[dk];
          x_ptr[irow] = val;
        }
      }
    }
    else
    {
      algebra.copy(x, xk);
      algebra.copy(y, x);
      algebra.addUMult(-1., *this->m_lu_matrix, xk, x);
      algebra.pointwiseMult(m_inv_diag, x, x);
      //algebra.multInvDiag(*this->m_lu_matrix,x) ;
    }
  }

  template <typename AlgebraT>
  void blockSolveU(AlgebraT& algebra, VectorType const& y, VectorType& x, VectorType& xk) const
  {
#ifdef ALIEN_USE_PERF_TIMER
    typename MatrixType::SentryType sentry(this->m_lu_matrix->timer(), "SolveU");
#endif
    if constexpr (MatrixType::on_host_only)
    {
#if defined (ALIEN_USE_EIGEN3)  && !defined(EIGEN3_DISABLED)
      using namespace Eigen;
      using Block2D     = Eigen::Matrix<ValueType,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> ;
      using Block2DView = Eigen::Map<Block2D> ;
      using Block1D     = Eigen::Matrix<ValueType,Dynamic,1> ;
      using Block1DView = Eigen::Map<Block1D> ;

      int N   = this->m_block_size ;
      int NxN = N*N;

      Block1D val(N);

      CSRConstViewT<MatrixT> view(*this->m_lu_matrix);
      // clang-format off
      auto nrows  = view.nrows();
      auto kcol   = view.kcol();
      auto dcol   = view.dcol();
      auto cols   = view.cols();
      auto values = view.data();

      auto y_ptr  = y.data();
      auto x_ptr  = x.data();
      auto xk_ptr = xk.data();
      // clang-format on

      std::copy(x_ptr, x_ptr + nrows*N, xk_ptr);
      if (this->m_is_parallel) {
        auto& local_row_size = this->m_lu_matrix->getDistStructInfo().m_local_row_size;
        int first_upper_ghost_index = this->m_lu_matrix->getDistStructInfo().m_first_upper_ghost_index;

        typedef typename LUSendRecvTraits<TagType>::vector_op_type SendRecvOpType;
        SendRecvOpType op(xk_ptr,
                          this->m_lu_matrix->getDistStructInfo().m_send_info,
                          this->m_lu_matrix->getSendPolicy(),
                          xk_ptr,
                          this->m_lu_matrix->getDistStructInfo().m_recv_info,
                          this->m_lu_matrix->getRecvPolicy(),
                          this->m_lu_matrix->getParallelMng(),
                          nullptr);
        op.upperRecv();
        op.lowerSend();
        for (std::size_t irow = 0; irow < nrows; ++irow) {
          int dk = dcol[irow];
          //ValueType val = y_ptr[irow];
          val = Block1DView(const_cast<ValueType*>(y_ptr+irow*N),N) ;
          for (int k = dk + 1; k < kcol[irow] + local_row_size[irow]; ++k) {
            //val -= values[k] * xk_ptr[cols[k]];
            val -= Block2DView(const_cast<ValueType*>(values+k*NxN),N,N) * Block1DView(xk_ptr+cols[k]*N,N) ;
          }
          for (int k = kcol[irow] + local_row_size[irow]; k < kcol[irow + 1]; ++k) {
            if (cols[k] >= first_upper_ghost_index)
            {
              //val -= values[k] * xk_ptr[cols[k]];
              val -= Block2DView(const_cast<ValueType*>(values+k*NxN),N,N) * Block1DView(xk_ptr+cols[k]*N,N) ;
            }
          }
          //val = val / values[dk];
          //x_ptr[irow] = val;
          Block1DView(x_ptr+irow*N,N) = BaseType::inv(Block2DView(const_cast<ValueType*>(values+dk*NxN),N,N)) * val;
        }
      }
      else
      {
        /*
        for (std::size_t irow = 0; irow < nrows; ++irow)
        {
          std::cout<<"XK["<<irow<<"]=\n";
          for(int i=0;i<N;++i)
            std::cout<<xk_ptr[irow*N+i]<<std::endl;
        }*/
        for (std::size_t irow = 0; irow < nrows; ++irow)
        {
          //std::cout<<"U SOLVE MAT["<<irow<<"]:"<<std::endl ;
          int dk = dcol[irow];
          //ValueType val = y_ptr[irow];
          val = Block1DView(const_cast<ValueType*>(y_ptr+irow*N),N) ;
          for (int k = dk + 1; k < kcol[irow + 1]; ++k) {
            //val -= values[k] * xk_ptr[cols[k]];
            val -= Block2DView(const_cast<ValueType*>(values+k*NxN),N,N) * Block1DView(xk_ptr+cols[k]*N,N) ;
            //std::cout<<Block2DView(const_cast<ValueType*>(values+k*NxN),N,N)<<std::endl;
          }
          //val = val / values[dk];
          //x_ptr[irow] = val;
          Block1DView(x_ptr+irow*N,N) = BaseType::inv(Block2DView(const_cast<ValueType*>(values+dk*NxN),N,N)) * val;
          //std::cout<<"Y["<<irow<<"]=\n"<<val<<std::endl;
        }
        /*
        for (std::size_t irow = 0; irow < nrows; ++irow)
        {
          int dk = dcol[irow];
          std::cout<<"INV DIAG["<<irow<<"]\n"<<BaseType::inv(Block2DView(const_cast<ValueType*>(values+dk*NxN),N,N))<<std::endl ;
          std::cout<<"Y["<<irow<<"]=\n";
          for(int i=0;i<N;++i)
            std::cout<<x_ptr[irow*N+i]<<std::endl;
        }*/
      }
#endif
    }
    else
    {
      algebra.copy(x, xk);
      algebra.copy(y, x);
      algebra.addUMult(-1., *this->m_lu_matrix, xk, x);
      algebra.copy(x, xk);
      algebra.multDiag(m_inv_diag, xk, x);
      //algebra.multInvDiag(*this->m_lu_matrix,x) ;
    }
  }

  template <typename AlgebraT>
  void solve(AlgebraT& algebra, VectorType const& x, VectorType& y) const
  {

    //////////////////////////////////////////////////////////////////////////
    //
    //     L.X1 = Y
    //
    algebra.copy(x, this->m_x);
    if(this->m_block_size==1)
    {
      for (int iter = 0; iter < m_nb_solver_iter; ++iter) {
        solveL(algebra, x, this->m_x, m_xk);
      }
    }
    else
    {
      for (int iter = 0; iter < m_nb_solver_iter; ++iter) {
        blockSolveL(algebra, x, this->m_x, m_xk);
      }
    }

    //////////////////////////////////////////////////////////////////////
    //
    // Solve U.X = L-1(Y)
    //
    algebra.copy(this->m_x, y);
    if(this->m_block_size==1)
    {
      for (int iter = 0; iter < m_nb_solver_iter; ++iter) {
        solveU(algebra, this->m_x, y, m_xk);
      }
    }
    else
    {
      for (int iter = 0; iter < m_nb_solver_iter; ++iter) {
        blockSolveU(algebra, this->m_x, y, m_xk);
      }
    }
  }

 private:
  void _factorizeRow(std::size_t irow,
                     int const* kcol,
                     int const* dcol,
                     int const* cols,
                     ValueType* values,
                     ValueType* values0)
  {
    /*
       *
            For k = 1, . . . , i - 1 and if (i, k) 2 NZ(A) Do:
                Compute aik := aik/akj
                For j = k + 1, . . . and if (i, j) 2 NZ(A), Do:
                   compute aij := aij - aik.ak,j.
                EndFor
            EndFor
       *
       */

    for (int k = kcol[irow]; k < dcol[irow]; ++k) // k=1 ->i-1
    {
      int krow = cols[k];
      typename BaseType::ValueType aik = values[k] / values0[dcol[krow]];
      values[k] = aik; // aik = aik/akk
      for (int l = kcol[krow]; l < kcol[krow + 1]; ++l)
        this->m_work[cols[l]] = l;
      for (int j = k + 1; j < kcol[irow + 1]; ++j) // j=k+1->n
      {
        int jcol = cols[j];
        int kj = this->m_work[jcol];
        if (kj != -1) {
          values[j] -= aik * values0[kj]; // aij = aij - aik*akj
        }
      }
      for (int l = kcol[krow]; l < kcol[krow + 1]; ++l)
        this->m_work[cols[l]] = -1;
    }
  }

  // clang-format off
  mutable VectorType        m_xk ;
  mutable VectorType        m_inv_diag ;
  // clang-format on

 public:
  //!PARAMETERS
  // clang-format off
  int       m_nb_factorization_iter = 0 ;
  int       m_nb_solver_iter        = 0 ;
  ValueType m_tol                   = 0 ;
  // clang-format on
};

template <typename AlgebraT>
class FILU0Preconditioner
{
 public:
  // clang-format off
  typedef AlgebraT                        AlgebraType ;
  typedef typename AlgebraType::Matrix    MatrixType;
  typedef typename AlgebraType::Vector    VectorType;
  typedef typename MatrixType::ValueType  ValueType;
  // clang-format on

  typedef FLUFactorisationAlgo<MatrixType, VectorType> AlgoType;

  FILU0Preconditioner(AlgebraType& algebra,
                      MatrixType const& matrix,
                      ITraceMng* trace_mng = nullptr)
  : m_algebra(algebra)
  , m_matrix(matrix)
  , m_trace_mng(trace_mng)
  {}

  virtual ~FILU0Preconditioner()
  {}

  void setParameter(std::string param, int value)
  {
    m_algo.setParameter(param, value);
  }

  void setParameter(std::string param, double value)
  {
    m_algo.setParameter(param, value);
  }

  void init()
  {
    m_algo.init(m_algebra, m_matrix);
    if (m_trace_mng) {
      m_trace_mng->info() << "FILU Preconditioner :";
      m_trace_mng->info() << "     Nb Factor iter :" << m_algo.m_nb_factorization_iter;
      m_trace_mng->info() << "     Nb Solver iter :" << m_algo.m_nb_solver_iter;
      m_trace_mng->info() << "     Tolerance      :" << m_algo.m_tol;
    }
  }

  void solve(VectorType const& y, VectorType& x) const
  {
    m_algo.solve(m_algebra, y, x);
  }

  void solve(AlgebraType& algebra, VectorType const& y, VectorType& x) const
  {
    m_algo.solve(algebra, y, x);
  }

 private:
  // clang-format off
  AlgebraType&                  m_algebra ;
  MatrixType const&             m_matrix;
  AlgoType                      m_algo ;

  ITraceMng*                    m_trace_mng = nullptr ;
  bool                          m_verbose   = false ;
  // clang-format on
};

} // namespace Alien
