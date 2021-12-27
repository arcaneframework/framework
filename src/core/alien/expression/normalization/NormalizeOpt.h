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
 * \file NormalizeOpt.h
 * \brief NormalizeOpt.h
 */

#pragma once

#include <alien/data/CompositeMatrix.h>
#include <alien/data/CompositeVector.h>
#include <alien/data/IMatrix.h>
#include <alien/data/IVector.h>
#include <alien/kernels/composite/CompositeMatrix.h>
#include <alien/kernels/composite/CompositeVector.h>
#include <alien/kernels/simple_csr/SimpleCSRMatrix.h>
#include <alien/kernels/simple_csr/SimpleCSRVector.h>
#include <alien/utils/Precomp.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup expression
 * \brief Normalize a linear system
 */
class ALIEN_EXPORT NormalizeOpt
{
 public:
  //! Type of algorithm
  enum eAlgoType
  {
    StdLU,
    EigenLU
  };

  //! Type of the options
  enum eOptType
  {
    SumFirstEq
  };

  //! Type of the error
  enum eErrorType
  {
    NoError,
    PivotError,
    WithErrors
  };

  //! Type of the matrix implementation
  typedef SimpleCSRMatrix<Arccore::Real> MatrixImpl;
  //! Type of the vector implementation
  typedef SimpleCSRVector<Arccore::Real> VectorImpl;

  //! Constructor
  NormalizeOpt();

  //! Free resources
  virtual ~NormalizeOpt() {}

  /*!
   * \brief Set the type of algorithm
   * \param[in] algo The type of algorithm
   */
  void setAlgo(eAlgoType algo);

  /*!
   * \brief Set the options
   * \param[in] opt The option
   * \param[in] flag Whether or not the option is activated
   */
  void setOpt(eOptType opt, bool flag);

  /*!
   * \brief Normalize the linear system
   * \param[in] matrix The matrix
   * \param[in] vector The vector
   * \returns The eventual error
   */
  eErrorType normalize(IMatrix& matrix, IVector& vector) const;

  /*!
   * \brief Normalize a composite linear system
   * \param[in] matrix The composite matrix
   * \param[in] vector The composite vector
   * \param[in] eq_ids The ids of the equations
   * \returns The eventual error
   */
  eErrorType normalize(CompositeMatrix& matrix, CompositeVector& vector,
                       ConstArrayView<Integer> eq_ids) const;

  template <int N, bool check_null_pivot = true>
  class LU;
  class Op;
  class Op2;

 private:
  /*!
   * \brief Normalize a linear system
   * \param[in] matrix The matrix
   * \param[in] vector The vector
   * \returns The eventual error
   */
  eErrorType _normalize(MatrixImpl& matrix, VectorImpl& vector) const;

  /*!
   * \brief Normalize a linear system
   * \param[in] matrix The first submatrix
   * \param[in] matrix2 The second submatrix
   * \param[in] trans Whether or not the matrix is transposed
   * \param[in] eq_ids The ids of the equations
   * \param[in] vector The vector
   * \returns The eventual error
   */
  eErrorType _normalize(MatrixImpl& matrix, MatrixImpl& matrix2, bool trans,
                        ConstArrayView<Integer> eq_ids, VectorImpl& vector) const;

  /*!
   * \brief Normalize a linear system
   * \param[in] matrix The first submatrix
   * \param[in] eq_ids The ids of the equations
   * \param[in] matrix2 The second submatrix
   * \param[in] trans Whether or not the matrix is transposed
   * \param[in] vector The vector
   * \returns The eventual error
   */
  eErrorType _normalize(MatrixImpl& matrix,
                        Arccore::ConstArrayView<Arccore::Integer> eq_ids, MatrixImpl& matrix2, bool trans,
                        VectorImpl& vector) const;

  //! The algorithm
  eAlgoType m_algo;
  //! Flag to sum the fist equation
  bool m_sum_first_eq;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief LU normalization
 * \tparam N The size of the matrix
 * \tparam check_null_pivot Whether or not a null pivot should be checked
 */
template <int N, bool check_null_pivot>
class NormalizeOpt::LU
{
 public:
  //! Type of the blocks
  typedef Arccore::Real Block2DType[N][N];
  /*
   * \brief Constructor
   * \param[in] Ap The matrix
   * \param[in] factorize Whether or not the matrix should be factorized
   */
  LU(Arccore::Real* Ap, bool factorize = true)
  : m_A(*(Block2DType*)Ap)
  {
    if (factorize) {
      Block2DType& A = *(Block2DType*)Ap;

      for (int k = 0; k < N; ++k) {
        if (check_null_pivot) {
          assert(A[k][k] != 0);
          A[k][k] = 1 / A[k][k];
        }
        else {
          if (A[k][k] == 0)
            A[k][k] = 0;
          else
            A[k][k] = 1 / A[k][k];
        }
        for (int i = k + 1; i < N; ++i) {
          A[i][k] *= A[k][k];
          // TOCHECK : to be removed ?
          // A[i][k] /= A[k][k];
        }
        for (int i = k + 1; i < N; ++i) {
          for (int j = k + 1; j < N; ++j) {
            A[i][j] -= A[i][k] * A[k][j];
          }
        }
      }
    }
  }

  //! Set identity
  void setIdentity()
  {
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < i; ++j) {
        m_A[i][j] = 0;
      }
      m_A[i][i] = 1;
      for (int j = i + 1; j < N; ++j) {
        m_A[i][j] = 0;
      }
    }
  }

  //! Set zero
  void setZero()
  {
    Real* ptr = (Real*)m_A;
    for (int i = 0; i < N * N; ++i) {
      ptr[i] = 0;
    }
  }

  /*!
   * \brief LXSolve
   * \param[in] Xp The rhs
   * \tparam NRhs The number of rhs
   * \tparam TransRhs Wheteher the rhs is transposed
   */
  template <int NRhs, bool TransRhs>
  void LXSolve(Arccore::Real* Xp) const
  {
    if (TransRhs) {
      typedef Real Rhs2DType[NRhs][N];
      Rhs2DType& X = *(Rhs2DType*)Xp;

      for (int k = 0; k < NRhs; ++k) {
        for (int i = 1; i < N; ++i) {
          for (int j = 0; j < i; ++j) {
            X[k][i] -= m_A[i][j] * X[k][j];
          }
        }
      }
    }
    else {
      typedef Real Rhs2DType[N][NRhs];
      Rhs2DType& X = *(Rhs2DType*)Xp;

      for (int i = 1; i < N; ++i) {
        for (int j = 0; j < i; ++j) {
          for (int k = 0; k < NRhs; ++k) {
            X[i][k] -= m_A[i][j] * X[j][k];
          }
        }
      }
    }
  }

  /*!
   * \brief UXSolve
   * \param[in] Xp The rhs
   * \tparam NRhs The number of rhs
   * \tparam TransRhs Wheteher the rhs is transposed
   */
  template <int NRhs, bool TransRhs>
  void UXSolve(Arccore::Real* Xp) const
  {
    if (TransRhs) {
      typedef Real Rhs2DType[NRhs][N];
      Rhs2DType& X = *(Rhs2DType*)Xp;
      for (int i = N - 1; i >= 0; --i) {
        for (int k = NRhs - 1; k >= 0; --k) {
          for (int j = N - 1; j > i; --j) {
            X[k][i] -= m_A[i][j] * X[k][j];
          }
        }
        for (int k = NRhs - 1; k >= 0; --k) {
          // TOCHECK : to be removed ?
          // X[k][i] /= m_A[i][i];
          X[k][i] *= m_A[i][i];
        }
      }
    }
    else {
      typedef Real Rhs2DType[N][NRhs];
      Rhs2DType& X = *(Rhs2DType*)Xp;

      for (int i = N - 1; i >= 0; --i) {
        for (int j = N - 1; j > i; --j) {
          for (int k = NRhs - 1; k >= 0; --k) {
            X[i][k] -= m_A[i][j] * X[j][k];
          }
        }
        for (int k = NRhs - 1; k >= 0; --k) {
          X[i][k] *= m_A[i][i];
          // TOCHECK: to be removed ?
          // X[i][k] /= m_A[i][i];
        }
      }
    }
  }

  /*!
   * \brief Solve
   * \param[in] Xp The rhs
   * \tparam NRhs The number of rhs
   * \tparam TransRhs Whether or not the rhs is transposed
   */
  template <int NRhs, bool TransRhs>
  void solve(Arccore::Real* Xp) const
  {
    LXSolve<NRhs, TransRhs>(Xp);
    UXSolve<NRhs, TransRhs>(Xp);
  }

 private:
  //! The matrix
  Block2DType& m_A;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Normalize operator
 */
class NormalizeOpt::Op
{
 public:
  /*!
   * \brief Constructor
   * \param[in] m Matrix implementation
   * \param[in] x Vector implementation
   * \param[in] algo The normalize algorithm
   * \param[in] sum_first_eq Whether or not the first equation must be summed
   */
  Op(MatrixImpl& m, VectorImpl& x, eAlgoType algo, bool sum_first_eq)
  : m_local_size(m.getCSRProfile().getNRow())
  , m_local_offset(m.getLocalOffset())
  , m_cols(m.getCSRProfile().getCols())
  , m_row_offset(m.getCSRProfile().getRowOffset())
  , m_matrix(m.internal().getDataPtr())
  , m_rhs(x.getDataPtr())
  , m_sum_first_eq(sum_first_eq)
  , m_algo(algo)
  {
    // TOCHECK : to be removed ?
    // m_equations_num = m.space().structInfo().size() ;
    m_equations_num = 1;
    if (m.block() && m.block()->size())
      m_equations_num = m.block()->size();
    m_unknowns_num = m_equations_num;
    m_block_size = m_equations_num * m_unknowns_num;
    m_diag_first = m.getCSRProfile().getDiagFirstOpt();
    if (!m_diag_first)
      m_upper_diag_offset = m.getCSRProfile().getUpperDiagOffset();
    m_keep_diag = true;
    if (m_sum_first_eq) {
      if (m_diag_first)
        sumBlockEq<true>();
      else
        sumBlockEq<false>();
    }
  }

  /*!
   * \brief Check if an equation is null
   * \param[in] irow The id of the equation
   * \param[in] diag_offset The diagonal offset
   */
  void checkNullEq(Arccore::Integer irow, Arccore::Integer diag_offset)
  {
    // search line with only zero
    for (Integer ieq = 0; ieq < m_equations_num; ++ieq) {
      bool ok = false;
      Integer col = m_row_offset[irow] + diag_offset;
      for (Integer ui = 0; ui < m_unknowns_num; ++ui)
        if (m_matrix[col * m_block_size + ij(ieq, ui)] != 0) {
          ok = true;
          break;
        }
      if (!ok) {
        // search null column
        for (Arccore::Integer ui = 0; ui < m_unknowns_num; ++ui) {
          bool ok2 = false;
          for (Integer jeq = 0; jeq < m_equations_num; ++jeq)
            if (m_matrix[col * m_block_size + ij(jeq, ui)] != 0) {
              ok2 = true;
              break;
            }
          if (!ok2) {
            // put 1 on ui column
            m_matrix[col * m_block_size + ij(ieq, ui)] = 1.;
            break;
          }
        }
      }
    }
  }

  /*!
   * \brief Row sum of blocks equation
   * \tparam diag_first Whether or not the first entry is the diagonal entry
   */
  template <bool diag_first>
  void sumBlockEq()
  {
    for (Integer irow = 0; irow < m_local_size; ++irow) {
      checkNullEq(irow, (diag_first ? 0 : m_upper_diag_offset[irow] - m_row_offset[irow]));
      for (Integer col = m_row_offset[irow]; col < m_row_offset[irow + 1]; ++col) {
        for (Integer ieq = 1; ieq < m_equations_num; ++ieq)
          for (Integer ui = 0; ui < m_unknowns_num; ++ui)
            m_matrix[col * m_block_size + ij(0, ui)] +=
            m_matrix[col * m_block_size + ij(ieq, ui)];
      }
      for (Integer ieq = 1; ieq < m_equations_num; ++ieq)
        m_rhs[irow * m_equations_num] += m_rhs[irow * m_equations_num + ieq];
    }
  }

  //! Invert diagonal
  template <int N>
  void multInvDiag();

 protected:
  Integer ijk(Integer i, Integer j, Integer k) const
  {
    return k * m_block_size + i * m_unknowns_num + j;
  }

  Arccore::Integer ij(Arccore::Integer i, Arccore::Integer j) const
  {
    return i * m_unknowns_num + j;
  }

  Arccore::Integer ik(Arccore::Integer i, Arccore::Integer k) const
  {
    return k * m_equations_num + i;
  }

  Arccore::Integer jk(Arccore::Integer j, Arccore::Integer k) const
  {
    return k * m_unknowns_num + j;
  }

  //! The local size
  Arccore::Integer m_local_size;
  //! The local offset
  Arccore::Integer m_local_offset;
  //! Column pointer
  Arccore::ConstArrayView<Arccore::Integer> m_cols;
  //! The rows offset
  Arccore::ConstArrayView<Arccore::Integer> m_row_offset;
  //! The indices of the upper diagonal
  Arccore::ConstArrayView<Arccore::Integer> m_upper_diag_offset;
  //! The number of equations
  Arccore::Integer m_equations_num;
  //! The number of unknowns
  Arccore::Integer m_unknowns_num;
  //! The block size
  Arccore::Integer m_block_size;
  //! The matrix
  Arccore::Real* m_matrix;
  //! The rhs
  Arccore::Real* m_rhs;
  //! Whether or not diagonal should be kept
  bool m_keep_diag;
  //! Whether or not the diagonal entry is stored first
  bool m_diag_first;
  //! Whether or not to sum the first equation
  bool m_sum_first_eq;
  //! The algorithm type
  eAlgoType m_algo;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// TOCHECK : to be removed ?
/*
  class NormalizeOpt::Op
  {
  public :
  Op(MatrixImpl& m, VectorImpl& x,eAlgoType algo,bool sum_first_eq)
  : m_local_size(m.getCSRProfile().getNRow())
  , m_local_offset(m.getLocalOffset())
  , m_cols(m.getCSRProfile().getCols())
  , m_row_offset(m.getCSRProfile().getRowOffset())
  , m_matrix(m.internal().getDataPtr())
  , m_rhs(x.getDataPtr())
  , m_sum_first_eq(sum_first_eq)
  , m_algo(algo)
  {
  m_equations_num = m.space().structInfo().size() ;
  m_unknowns_num = m_equations_num ;
  m_block_size = m_equations_num*m_unknowns_num ;
  m_diag_first = m.getCSRProfile().getDiagFirstOpt() ;
  if(!m_diag_first)
  m_upper_diag_offset = m.getCSRProfile().getUpperDiagOffset() ;
  m_keep_diag = true ;
  if(m_sum_first_eq)
  {
  if(m_diag_first)
  sumBlockEq<true>() ;
  else
  sumBlockEq<false>() ;
  }
  }

  void checkNullEq(Integer irow,Integer diag_offset)
  {
  //search line with only zero
  for(Integer ieq=0;ieq<m_equations_num;++ieq)
  {
  bool ok = false ;
  Integer col = m_row_offset[irow]+diag_offset ;
  for(Integer ui=0;ui<m_unknowns_num;++ui)
  if(m_matrix[col*m_block_size+ij(ieq,ui)]!=0)
  {
  ok = true ;
  break ;
  }
  if(!ok)
  {
  //search null colonne
  for(Integer ui=0;ui<m_unknowns_num;++ui)
  {
  bool ok2 = false ;
  for(Integer jeq=0;jeq<m_equations_num;++jeq)
  if(m_matrix[col*m_block_size+ij(jeq,ui)]!=0)
  {
  ok2 = true ;
  break ;
  }
  if(!ok2)
  {
  //put 1 on ui colonne
  m_matrix[col*m_block_size+ij(ieq,ui)] = 1. ;
  break ;
  }
  }
  }
  }
  }

  template<bool diag_first>
  void
  sumBlockEq()
  {
  for(Integer irow=0;irow<m_local_size;++irow)
  {
  checkNullEq(irow,(diag_first?0:m_upper_diag_offset[irow]-m_row_offset[irow])) ;
  for(Integer col= m_row_offset[irow];col<m_row_offset[irow+1];++col)
  {
  for(Integer ieq=1;ieq<m_equations_num;++ieq)
  for(Integer ui=0;ui<m_unknowns_num;++ui)
  m_matrix[col*m_block_size+ij(0,ui)] += m_matrix[col*m_block_size+ij(ieq,ui)] ;
  }
  for(Integer ieq=1;ieq<m_equations_num;++ieq)
  m_rhs[irow*m_equations_num] += m_rhs[irow*m_equations_num+ieq] ;
  }
  }

  template<int N>
  void multInvDiag() ;
  protected :
  Integer ijk(Integer i, Integer j, Integer k) const {
  return k*m_block_size+i*m_unknowns_num+j ;
  }
  Integer ij(Integer i, Integer j) const {
  return i*m_unknowns_num+j ;
  }
  Integer ik(Integer i, Integer k) const {
  return k*m_equations_num+i ;
  }
  Integer jk(Integer j, Integer k) const {
  return k*m_unknowns_num+j ;
  }

  Integer                 m_local_size ;
  Integer                 m_local_offset ;
  ConstArrayView<Integer> m_cols  ;
  ConstArrayView<Integer> m_row_offset ;
  ConstArrayView<Integer> m_upper_diag_offset ;
  Integer                 m_equations_num ;
  Integer                 m_unknowns_num ;
  Integer                 m_block_size ;
  Real*                   m_matrix ;
  Real*                   m_rhs ;
  bool                    m_keep_diag ;
  bool                    m_diag_first ;
  bool                    m_sum_first_eq;
  eAlgoType               m_algo ;
  } ;
*/

/*!
 * \brief Normalize operator for composite matrices
 */
class NormalizeOpt::Op2 : public NormalizeOpt::Op
{
 public:
  /*!
   * \brief Constructor
   * \param[in] m First submatrix implementation
   * \param[in] m2 Second submatrix implementation
   * \param[in] trans Transposed flag
   * \param[in] eq_ids The ids of equation
   * \param[in] x Vector implementation
   * \param[in] algo The normalize algorithm
   * \param[in] sum_first_eq Whether or not the first equation must be summed
   */
  Op2(MatrixImpl& m, MatrixImpl& m2, bool trans,
      Arccore::ConstArrayView<Arccore::Integer> eq_ids, VectorImpl& x, eAlgoType algo,
      bool sum_first_eq)
  : Op(m, x, algo, sum_first_eq)
  , m_nb_extra_eq(m2.getCSRProfile().getNRow())
  , m_eq_ids(eq_ids)
  , m_submatrix1(false)
  , m_submatrix2(true)
  , m_trans(trans)
  , m_nuk2(0)
  , m_extra_eq_cols(m2.getCSRProfile().getCols())
  , m_extra_eq_row_offset(m2.getCSRProfile().getRowOffset())
  , m_extra_eq_matrix(m2.internal().getDataPtr())
  {
    if (m_sum_first_eq) {
      if (m_trans) {
        // TOCHECK : to be removed ?
        // Integer neq = m2.space().structInfo().size() ;
        Integer neq = 1;
        if (m2.block() && m2.block()->size())
          neq = m2.block()->size();
        for (Integer i = 0; i < m_eq_ids.size(); ++i) {
          Integer ieq = m_eq_ids[i];
          for (Integer k = m_extra_eq_row_offset[ieq]; k < m_extra_eq_row_offset[ieq + 1];
               ++k) {
            Real val = m_extra_eq_matrix[k * neq];
            for (Integer ieq = 1; ieq < neq; ++ieq)
              val += m_extra_eq_matrix[k * neq + ieq];
            m_extra_eq_matrix[k * neq] = val;
          }
        }
      }
      else {
        // TOCHECK : to be removed ?
        // Integer neq = m2.eqSpace().structInfo().size() ;
        Integer neq = 1;
        if (m2.rowBlock() && m2.rowBlock()->maxBlockSize())
          neq = m2.rowBlock()->maxBlockSize();
        for (Integer i = 0; i < m_eq_ids.size(); ++i) {
          Integer irow = m_eq_ids[i];
          for (Integer k = m_extra_eq_row_offset[irow];
               k < m_extra_eq_row_offset[irow + 1]; ++k) {
            Real val = m_extra_eq_matrix[k * neq];
            for (Integer ieq = 1; ieq < neq; ++ieq)
              val += m_extra_eq_matrix[k * neq + ieq];
            m_extra_eq_matrix[k * neq] = val;
          }
        }
      }
    }
  }

  /*!
   * \brief Constructor
   * \param[in] m First submatrix implementation
   * \param[in] eq_ids The ids of equation
   * \param[in] m2 Second submatrix implementation
   * \param[in] trans Transposed flag
   * \param[in] x Vector implementation
   * \param[in] algo The normalize algorithm
   * \param[in] sum_first_eq Whether or not the first equation must be summed
   */
  Op2(MatrixImpl& m, Arccore::ConstArrayView<Arccore::Integer> eq_ids, MatrixImpl& m2,
      bool trans, VectorImpl& x, eAlgoType algo, bool sum_first_eq)
  : Op(m, x, algo, sum_first_eq)
  , m_nb_extra_eq(m2.getCSRProfile().getNRow())
  , m_eq_ids(eq_ids)
  , m_submatrix1(true)
  , m_submatrix2(false)
  , m_trans(trans)
  , m_nuk2(0)
  , m_extra_eq_cols(m2.getCSRProfile().getCols())
  , m_extra_eq_row_offset(m2.getCSRProfile().getRowOffset())
  , m_extra_eq_matrix(m2.internal().getDataPtr())
  {
    if (m_trans)
      // TOCHECK : to be removed
      // m_nuk2 = m2.eqSpace().structInfo().size() ;
      m_nuk2 = m2.rowBlock()->maxBlockSize();
    else
      // TOCHECK : to be removed
      // m_nuk2 = m2.uSpace().structInfo().size() ;
      m_nuk2 = m2.colBlock()->maxBlockSize();
  }

  //! Invert diagonal
  template <int N>
  void multInvDiag();

 private:
  //! Number of extra equations
  Arccore::Integer m_nb_extra_eq;
  //! Ids of the equations
  Arccore::ConstArrayView<Arccore::Integer> m_eq_ids;
  //! Whether or not the first submatrix should be normalized
  bool m_submatrix1;
  //! Whether or not the second submatrix should be normalized
  bool m_submatrix2;
  //! Transposed flag
  bool m_trans;
  //! Number of unknowns in extra equations
  Arccore::Integer m_nuk2;
  //! Pointer on extra equations unknowns columns
  Arccore::ConstArrayView<Arccore::Integer> m_extra_eq_cols;
  //! Pointer on extra equations unknowns rows
  Arccore::ConstArrayView<Arccore::Integer> m_extra_eq_row_offset;
  //! Pointer of extra equations datas
  Arccore::Real* m_extra_eq_matrix;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
