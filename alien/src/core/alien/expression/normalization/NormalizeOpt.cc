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

/*!
 * \file NormalizeOpt.cc
 * \brief NormalizeOpt.cc
 */

#include "NormalizeOpt.h"

#include <alien/core/impl/MultiVectorImpl.h>
#include <alien/kernels/simple_csr/SimpleCSRMatrix.h>
#include <alien/kernels/simple_csr/SimpleCSRVector.h>

#ifdef ALIEN_USE_EIGEN2
#include <Eigen/Core>
#include <Eigen/LU>
#include <iostream>

#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

using namespace Arccore;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

NormalizeOpt::NormalizeOpt()
: m_algo(StdLU)
, m_sum_first_eq(false)
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void NormalizeOpt::setAlgo(eAlgoType algo)
{
  m_algo = algo;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void NormalizeOpt::setOpt(eOptType opt, bool flag)
{
  switch (opt) {
  case SumFirstEq:
    m_sum_first_eq = flag;
    break;
  default:
    FatalErrorException(A_FUNCINFO, String::format("Unhandle option type: ", opt));
    break;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

NormalizeOpt::eErrorType
NormalizeOpt::normalize(IMatrix& m, IVector& x) const
{
  MatrixImpl& A = m.impl()->get<Alien::BackEnd::tag::simplecsr>(true);
  VectorImpl& b = x.impl()->get<Alien::BackEnd::tag::simplecsr>(true);
  return _normalize(A, b);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

NormalizeOpt::eErrorType
NormalizeOpt::normalize(
CompositeMatrix& m, CompositeVector& x, ConstArrayView<Integer> eq_ids) const
{
  // need to update timestamp of all submatrices
  MatrixImpl& A00 = m(0, 0).impl()->get<Alien::BackEnd::tag::simplecsr>(true);
  MatrixImpl& A01 = m(0, 1).impl()->get<Alien::BackEnd::tag::simplecsr>(true);
  MatrixImpl& A10 = m(1, 0).impl()->get<Alien::BackEnd::tag::simplecsr>(true);
  MatrixImpl& A11 = m(1, 1).impl()->get<Alien::BackEnd::tag::simplecsr>(true);
  VectorImpl& b0 = x[0].impl()->get<Alien::BackEnd::tag::simplecsr>(true);
  VectorImpl& b1 = x[1].impl()->get<Alien::BackEnd::tag::simplecsr>(true);
  {
    eErrorType error =
    _normalize(A00, A01, m(0, 1).impl()->hasFeature("transposed"), eq_ids, b0);
    if (error != NoError)
      return error;
  }
  {
    eErrorType error =
    _normalize(A11, eq_ids, A10, m(1, 0).impl()->hasFeature("transposed"), b1);
    return error;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <>
void NormalizeOpt::Op::multInvDiag<0>()
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <>
void NormalizeOpt::Op::multInvDiag<1>()
{
  for (Integer irow = 0; irow < m_local_size; ++irow) {
    Integer off = m_row_offset[irow];
    Real diag = m_matrix[off];
    for (Integer col = off; col < m_row_offset[irow + 1]; ++col) {
      m_matrix[col] /= diag;
    }
    m_rhs[irow] /= diag;

    if (!m_keep_diag)
      m_matrix[off] = 0.;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <Integer N>
void NormalizeOpt::Op::multInvDiag()
{
#ifdef ALIEN_USE_EIGEN2
  if (m_algo == EigenLU) {
    using namespace Eigen;
    using namespace std;
    typedef Eigen::Matrix<Real, N, N, RowMajor> MatrixType;
    typedef Eigen::Matrix<Real, N, 1> VectorType;
    const Integer NxN = N * N;
    // TOCHECK : to be removed ?
    // cout<<"MULT INV DIAG B"<<m_equations_num<<endl;
    UniqueArray<Real> block(NxN);
    UniqueArray<Real> vect(N);
    Map<MatrixType> m(block.begin(), N, N);
    Map<VectorType> v(vect.begin(), N);
    if (m_diag_first) {
      for (Integer irow = 0; irow < m_local_size; ++irow) {
        Integer off = m_row_offset[irow];

        Map<MatrixType> diag(&m_matrix[off * NxN], N, N);
        if (diag.determinant() == 0) {
          cout << "Non inversible diagonal : row=" << irow << endl;
          cout << " DIAG :" << endl;
          cout << diag << endl;
        }
        MatrixType inv_diag = diag.inverse();
        // TOCHECK : to be removed ?
        // PartialPivLU< MatrixType > lu(diag);
        // MatrixType inv_diag = lu.inverse() ;

        // OFF DIAGONAL treatment
        for (Integer col = off + 1; col < m_row_offset[irow + 1]; ++col) {
          block.copy(ArrayView<Real>(NxN, &m_matrix[col * NxN]));
          Map<MatrixType> matrix(&m_matrix[col * NxN], N, N);
          matrix = inv_diag * m;
          // TOCHECK : to be removed ?
          // matrix = lu.solve(m) ;
        }

        // RHS treatment
        vect.copy(ArrayView<Real>(N, &m_rhs[irow * N]));
        Map<VectorType> rhs(&m_rhs[irow * N], N);
        rhs = inv_diag * v;
        // TOCHECK : to be removed
        // rhs = lu.solve(v) ;

        if (m_keep_diag)
          diag.setIdentity();
        else
          diag.setZero();
      }
    }
    else {
      for (Integer irow = 0; irow < m_local_size; ++irow) {
        Integer off = m_upper_diag_offset[irow];

        Map<MatrixType> diag(&m_matrix[off * NxN], N, N);
        if (diag.determinant() == 0) {
          cout << "Non inversible diagonal : row=" << irow << endl;
          cout << " DIAG :" << endl;
          cout << diag << endl;
        }
        MatrixType inv_diag = diag.inverse();
        // TOCHECK : to be removed ?
        // PartialPivLU< MatrixType > lu(diag);
        // MatrixType inv_diag = lu.inverse() ;

        // OFF DIAGONAL treatment
        for (Integer col = m_row_offset[irow]; col < off; ++col) {
          block.copy(ArrayView<Real>(NxN, &m_matrix[col * NxN]));
          Map<MatrixType> matrix(&m_matrix[col * NxN], N, N);
          matrix = inv_diag * m;
          // TOCHECK : to be removed ?
          // matrix = lu.solve(m) ;
        }
        // skip diagonal block
        for (Integer col = off + 1; col < m_row_offset[irow + 1]; ++col) {
          block.copy(ArrayView<Real>(NxN, &m_matrix[col * NxN]));
          Map<MatrixType> matrix(&m_matrix[col * NxN], N, N);
          matrix = inv_diag * m;
          // TOCHECK : to be removed ?
          // matrix = lu.solve(m) ;
        }

        // RHS treatment
        vect.copy(ArrayView<Real>(N, &m_rhs[irow * N]));
        Map<VectorType> rhs(&m_rhs[irow * N], N);
        rhs = inv_diag * v;
        // TOCHECK : to be removed ?
        // rhs = lu.solve(v) ;

        if (m_keep_diag)
          diag.setIdentity();
        else
          diag.setZero();
      }
    }
  }
  else
#endif
  {
    if (m_diag_first) {
      for (Integer irow = 0; irow < m_local_size; ++irow) {
        Integer off = m_row_offset[irow];

        Real* diag = &m_matrix[off * N * N];
        LU<N> lu(diag);

        // OFF DIAGONAL treatment
        for (Integer col = off + 1; col < m_row_offset[irow + 1]; ++col) {
          lu.template solve<N, false>(&m_matrix[col * N * N]);
        }

        // RHS treatment
        lu.template solve<1, true>(&m_rhs[irow * N]);

        if (m_keep_diag)
          lu.setIdentity();
        else
          lu.setZero();
      }
    }
    else {
      for (Integer irow = 0; irow < m_local_size; ++irow) {
        Integer off = m_upper_diag_offset[irow];

        Real* diag = &m_matrix[off * N * N];
        LU<N> lu(diag);

        // OFF DIAGONAL treatment
        for (Integer col = m_row_offset[irow]; col < off; ++col) {
          lu.template solve<N, false>(&m_matrix[col * N * N]);
        }
        // skip diagonal block
        for (Integer col = off + 1; col < m_row_offset[irow + 1]; ++col) {
          lu.template solve<N, false>(&m_matrix[col * N * N]);
        }
        // RHS treatment
        lu.template solve<1, true>(&m_rhs[irow * N]);

        if (m_keep_diag)
          lu.setIdentity();
        else
          lu.setZero();
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

NormalizeOpt::eErrorType
NormalizeOpt::_normalize(MatrixImpl& m, VectorImpl& x) const
{
  Op op(m, x, m_algo, m_sum_first_eq);
  if (m.block()) {
    switch (m.block()->size()) {
    case 1:
      op.multInvDiag<1>();
      break;
    case 2:
      op.multInvDiag<2>();
      break;
    case 3:
      op.multInvDiag<3>();
      break;
    case 4:
      op.multInvDiag<4>();
      break;
    default:
      op.multInvDiag<0>();
      break;
    }
  }
  return NoError;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <>
void NormalizeOpt::Op2::multInvDiag<0>()
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <>
void NormalizeOpt::Op2::multInvDiag<1>()
{
  if (m_submatrix2) {
    if (m_diag_first) {
      ////////////////////////////////////////////
      //
      // NORMALIZE SubMatrix00 and RHS
      for (Integer irow = 0; irow < m_local_size; ++irow) {
        Integer off = m_row_offset[irow];
        Real diag = m_matrix[off];
        for (Integer col = off + 1; col < m_row_offset[irow + 1]; ++col) {
          m_matrix[col] /= diag;
        }
        m_rhs[irow] /= diag;
      }

      ////////////////////////////////////////////
      //
      // NORMALIZE SubMatrix01
      if (m_trans)
        for (Integer i = 0; i < m_eq_ids.size(); ++i) {
          Integer ieq = m_eq_ids[i];
          for (Integer k = m_extra_eq_row_offset[ieq]; k < m_extra_eq_row_offset[ieq + 1];
               ++k) {
            Integer irow = m_extra_eq_cols[k];
            Integer off = m_row_offset[irow];
            Real diag = m_matrix[off];
            m_extra_eq_matrix[k] /= diag;
          }
        }
      else
        for (Integer i = 0; i < m_eq_ids.size(); ++i) {
          Integer irow = m_eq_ids[i];
          Integer off = m_row_offset[irow];
          Real diag = m_matrix[off];
          for (Integer k = m_extra_eq_row_offset[irow];
               k < m_extra_eq_row_offset[irow + 1]; ++k) {
            m_extra_eq_matrix[k] /= diag;
          }
        }
      ////////////////////////////////////////////
      //
      // SET DIAG TO Id
      for (Integer irow = 0; irow < m_local_size; ++irow) {
        Integer off_diag = m_row_offset[irow];
        m_matrix[off_diag] = m_keep_diag ? 1. : 0.;
      }
    }
    else {
      ////////////////////////////////////////////
      //
      // NORMALIZE SubMatrix00 and RHS
      for (Integer irow = 0; irow < m_local_size; ++irow) {
        Integer off_diag = m_upper_diag_offset[irow];
        Real diag = m_matrix[off_diag];
        for (Integer col = m_row_offset[irow]; col < off_diag; ++col) {
          m_matrix[col] /= diag;
        }
        for (Integer col = off_diag + 1; col < m_row_offset[irow + 1]; ++col) {
          m_matrix[col] /= diag;
        }
        m_rhs[irow] /= diag;
      }

      ////////////////////////////////////////////
      //
      // NORMALIZE SubMatrix01
      if (m_trans)
        for (Integer i = 0; i < m_eq_ids.size(); ++i) {
          Integer ieq = m_eq_ids[i];
          for (Integer k = m_extra_eq_row_offset[ieq]; k < m_extra_eq_row_offset[ieq + 1];
               ++k) {
            Integer irow = m_extra_eq_cols[k];
            Integer off = m_upper_diag_offset[irow];
            Real diag = m_matrix[off];
            m_extra_eq_matrix[k] /= diag;
          }
        }
      else
        for (Integer i = 0; i < m_eq_ids.size(); ++i) {
          Integer irow = m_eq_ids[i];
          Integer off = m_upper_diag_offset[irow];
          Real diag = m_matrix[off];
          for (Integer k = m_extra_eq_row_offset[irow];
               k < m_extra_eq_row_offset[irow + 1]; ++k) {
            m_extra_eq_matrix[k] /= diag;
          }
        }
      ////////////////////////////////////////////
      //
      // SET DIAG TO Id
      for (Integer irow = 0; irow < m_local_size; ++irow) {
        Integer off_diag = m_upper_diag_offset[irow];
        m_matrix[off_diag] = m_keep_diag ? 1. : 0.;
      }
    }
  }

  if (m_submatrix1) {
    ////////////////////////////////////////////
    //
    // NORMALIZE SubMatrix10 SubMatrix11 RHS
    for (Integer i = 0; i < m_eq_ids.size(); ++i) {
      Integer ieq = m_eq_ids[i];
      Real diag = m_matrix[ieq];
      for (Integer j = m_extra_eq_row_offset[ieq]; j < m_extra_eq_row_offset[ieq + 1];
           ++j) {
        for (Integer k = 0; k < m_nuk2; ++k)
          m_extra_eq_matrix[j * m_nuk2 + k] /= diag;
      }
      m_rhs[ieq] /= diag;
      m_matrix[ieq] = 1.;
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <Integer N>
void NormalizeOpt::Op2::multInvDiag()
{
  if (m_submatrix2) {
    if (m_diag_first) {
      ////////////////////////////////////////////
      //
      // NORMALIZE SubMatrix00
      for (Integer irow = 0; irow < m_local_size; ++irow) {
        Integer off = m_row_offset[irow];

        Real* diag = &m_matrix[off * N * N];
        LU<N> lu(diag);

        // OFF DIAGONAL treatment
        for (Integer col = off + 1; col < m_row_offset[irow + 1]; ++col) {
          lu.template solve<N, false>(&m_matrix[col * N * N]);
        }

        // RHS treatment
        lu.template solve<1, true>(&m_rhs[irow * N]);
      }

      ////////////////////////////////////////////
      //
      // NORMALIZE SubMatrix01
      if (m_trans)
        for (Integer i = 0; i < m_eq_ids.size(); ++i) {
          Integer ieq = m_eq_ids[i];
          for (Integer k = m_extra_eq_row_offset[ieq]; k < m_extra_eq_row_offset[ieq + 1];
               ++k) {
            Integer irow = m_extra_eq_cols[k];
            Integer off = m_row_offset[irow];
            Real* diag = &m_matrix[off * N * N];
            LU<N> lu(diag, false);

            lu.template solve<1, true>(&m_extra_eq_matrix[k * N]);
          }
        }
      else
        for (Integer i = 0; i < m_eq_ids.size(); ++i) {
          Integer irow = m_eq_ids[i];
          Integer off = m_row_offset[irow];
          Real* diag = &m_matrix[off * N * N];
          LU<N> lu(diag, false);
          for (Integer k = m_extra_eq_row_offset[irow];
               k < m_extra_eq_row_offset[irow + 1]; ++k) {
            lu.template solve<N, false>(&m_extra_eq_matrix[k * N]);
          }
        }

      ////////////////////////////////////////////
      //
      // SET DIAG TO Id
      if (m_keep_diag)
        for (Integer irow = 0; irow < m_local_size; ++irow) {
          Integer off = m_row_offset[irow];
          Real* diag = &m_matrix[off * N * N];
          LU<N> lu(diag, false);

          lu.setIdentity();
        }
      else
        for (Integer irow = 0; irow < m_local_size; ++irow) {
          Integer off = m_row_offset[irow];
          Real* diag = &m_matrix[off * N * N];
          LU<N> lu(diag, false);
          lu.setZero();
        }
    }
    else {
      ////////////////////////////////////////////
      //
      // NORMALIZE SubMatrix01
      for (Integer irow = 0; irow < m_local_size; ++irow) {
        Integer off = m_upper_diag_offset[irow];

        Real* diag = &m_matrix[off * N * N];
        LU<N> lu(diag);

        // OFF DIAGONAL treatment
        for (Integer col = m_row_offset[irow]; col < off; ++col) {
          lu.template solve<N, false>(&m_matrix[col * N * N]);
        }
        // skip diagonal block
        for (Integer col = off + 1; col < m_row_offset[irow + 1]; ++col) {
          lu.template solve<N, false>(&m_matrix[col * N * N]);
        }
        // RHS treatment
        lu.template solve<1, true>(&m_rhs[irow * N]);
      }
      ////////////////////////////////////////////
      //
      // NORMALIZE SubMatrix01
      if (m_trans)
        for (Integer i = 0; i < m_eq_ids.size(); ++i) {
          Integer ieq = m_eq_ids[i];
          for (Integer k = m_extra_eq_row_offset[ieq]; k < m_extra_eq_row_offset[ieq + 1];
               ++k) {
            Integer irow = m_extra_eq_cols[k] - m_local_offset;

            Integer off = m_upper_diag_offset[irow];
            Real* diag = &m_matrix[off * N * N];
            LU<N> lu(diag, false);

            lu.template solve<1, true>(&m_extra_eq_matrix[k * N]);
          }
        }
      else {
        for (Integer i = 0; i < m_eq_ids.size(); ++i) {
          Integer irow = m_eq_ids[i];
          Integer off = m_upper_diag_offset[irow];
          Real* diag = &m_matrix[off * N * N];
          LU<N> lu(diag, false);
          for (Integer k = m_extra_eq_row_offset[irow];
               k < m_extra_eq_row_offset[irow + 1]; ++k) {
            lu.template solve<N, false>(&m_extra_eq_matrix[k * N]);
          }
        }
      }

      ////////////////////////////////////////////
      //
      // SET DIAG TO Id
      if (m_keep_diag)
        for (Integer irow = 0; irow < m_local_size; ++irow) {
          Integer off = m_upper_diag_offset[irow];
          Real* diag = &m_matrix[off * N * N];
          LU<N> lu(diag, false);

          lu.setIdentity();
        }
      else
        for (Integer irow = 0; irow < m_local_size; ++irow) {
          Integer off = m_upper_diag_offset[irow];
          Real* diag = &m_matrix[off * N * N];
          LU<N> lu(diag, false);
          lu.setZero();
        }
    }
  }

  if (m_submatrix1) {
    ////////////////////////////////////////////
    //
    // NORMALIZE SubMatrix10 SubMatrix11 RHS
    for (Integer i = 0; i < m_eq_ids.size(); ++i) {
      Integer ieq = m_eq_ids[i];
      Real diag = m_matrix[ieq];
      for (Integer j = m_extra_eq_row_offset[ieq]; j < m_extra_eq_row_offset[ieq + 1];
           ++j) {
        for (Integer k = 0; k < N; ++k)
          m_extra_eq_matrix[j * N + k] /= diag;
      }
      m_rhs[ieq] /= diag;
      m_matrix[ieq] = 1.;
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

NormalizeOpt::eErrorType
NormalizeOpt::_normalize(MatrixImpl& m, MatrixImpl& m2, bool trans,
                         ConstArrayView<Integer> eq_ids, VectorImpl& x) const
{
  Op2 op(m, m2, trans, eq_ids, x, m_algo, m_sum_first_eq);

  switch (x.block()->size()) {
  case 1:
    op.multInvDiag<1>();
    break;
  case 2:
    op.multInvDiag<2>();
    break;
  case 3:
    op.multInvDiag<3>();
    break;
  case 4:
    op.multInvDiag<4>();
    break;
  default:
    op.multInvDiag<0>();
    break;
  }
  return NoError;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

NormalizeOpt::eErrorType
NormalizeOpt::_normalize(MatrixImpl& m, ConstArrayView<Integer> eq_ids, MatrixImpl& m2,
                         bool trans, VectorImpl& x) const
{
  Op2 op(m, eq_ids, m2, trans, x, m_algo, m_sum_first_eq);

  switch (x.block()->size()) {
  case 1:
    op.multInvDiag<1>();
    break;
  case 2:
    op.multInvDiag<2>();
    break;
  case 3:
    op.multInvDiag<3>();
    break;
  case 4:
    op.multInvDiag<4>();
    break;
  default:
    op.multInvDiag<0>();
    break;
  }
  return NoError;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
