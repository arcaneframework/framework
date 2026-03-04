// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#pragma once

#include <string>

#include "boost/numeric/ublas/matrix_expression.hpp"
#include "boost/numeric/ublas/vector_expression.hpp"
#include "boost/numeric/ublas/matrix.hpp"
#include "boost/numeric/ublas/vector.hpp"
#include "boost/type_traits/is_same.hpp"

#include "alien/local_direct_solvers/algorithms/lapackUtils.h"

namespace ublas = boost::numeric::ublas;

/*!
  \class LUSolver
  \author Daniele A. Di Pietro
  \date 2006-07-29
  \brief Solve the linear system \f$A x = b\f$ using the \f$LU\f$ factorization


  The dense \f$LU\f$ solver can be used in one of the following ways:

  - if the matrix \f$A\f$ needn't be preserved after computation and the
  associated system has to be solved only once, simply call solve(A, b).
  At exit, the argument A will contain the \f$LU\f$ factorization of the
  original matrix.
  In order to avoid vector/matrix copies, the overwrite_solve() members may be
  used instead, the sole difference being that the solution is overwritten over
  the rhs vector/matrix.
  When the matrix \f$A\f$ passed as an argument is stored column-wise, the
  \f$L\f$ and \f$U\f$ factors are overwritten in the lower and upper triangular
  part of \f$A\f$ respectively.
  When \f$A\f$ is stored row-wise, the \f$U^t\f$ and \f$L^t\f$ factors are
  stored in the lower and upper triangular part of \f$A\f$ respectively.

  - if the matrix \f$A\f$ needs to be preserved after the computation or the
  associated system has to be solved several times, consider using a two-step
  strategy by first invoking factor(A), which makes a local copy of
  the matrix and factor it, and the calling solve(b) member, which actually
  solves the associated system.
  At exit, the matrix A will be preserved and its factorization will be stored
  in the LUSolver object.
  When solve() is replaced by overwrite_solve(), the solution is overwritten
  on the provided argument.
*/

template<typename T>
class LUSolver {
public:
  enum ErrorCode {
    E_NoError           = 0,
    E_SingularMatrix    = 1,
    E_NonSquareMatrix   = 2,
    E_InconsistentSizes = 3,
    E_SubstitutionError = 4,
    E_NoFactoredMatrix  = 5
  };

  enum WarningCode {
    W_EmptyMatrix       = 0
  };
 public:
  struct Warning {
    std::string msg;
    WarningCode code;
    Warning(const std::string& _msg, WarningCode _code)
      : msg(_msg), code(_code) {}
  };

  struct Error {
    std::string msg;
    ErrorCode code;
    Error(const std::string& _msg, ErrorCode _code)
      : msg(_msg), code(_code) {}
  };

 public:
  //! Internal matrix type.
  typedef ublas::matrix<T, ublas::column_major> matrix_type;
  typedef ublas::matrix<T, ublas::column_major> MatrixType;
  //! Internal vector type
  typedef ublas::vector<T>                      vector_type;
  typedef ublas::vector<T>                      VectorType;
  //! Size type
  typedef typename matrix_type::size_type       size_type;
  //! Pivot vector type
  typedef ublas::vector<int>                    pv_type;

 public:
  //! Constructor
  LUSolver(bool no_exception=false)
  : m_factored(false)
  , m_no_exception(no_exception){}
  virtual ~LUSolver() {}

  public:
  //! Copy and factor the matrix
  template<class E>
  int factor(const ublas::matrix_expression<E>& A);

 public:
  //! Solve the linear system for a single rhs
  template<class E>
  vector_type solve(const ublas::vector_expression<E>& b);

  //! Solve the linear system for multiple rhs'
  template<class E>
  matrix_type solve(const ublas::matrix_expression<E>& B);

  //! Solve the linear system for a single rhs and overwrite the solution
  template<class E>
  void overwrite_solve(ublas::vector_expression<E>& b);

  //! Solve the linear system for multiple rhs' and overwrite the solution
  void overwrite_solve(matrix_type& B);

  //! Overwrite the \f$LU\f$ factorization of the matrix \f$A\f$ and
  //! solve the system \f$A x = b\f$ for a single rhs
  template<class L, class E>
  vector_type solve(ublas::matrix<T, L>& A, const ublas::vector_expression<E>& b);

  //! Overwrite the \f$LU\f$ factorization of the matrix \f$A\f$ and
  //! solve the system \f$A x = b\f$ for a multiple rhs'
  template<class L, class E>
  matrix_type solve(ublas::matrix<T, L>& A, const ublas::matrix_expression<E>& b);


  //! Overwrite the \f$LU\f$ factorization of the matrix \f$A\f$ and
  //! solve the system \f$A x = b\f$ for a single rhs.
  //! The solution is overwritten over \f$b\f$
  template<class L>
  void overwrite_solve(ublas::matrix<T, L>& A, vector_type& b);

  //! Overwrite the \f$LU\f$ factorization of the matrix \f$A\f$ and
  //! solve the system \f$A x = b\f$ for a single rhs.
  //! The solution is overwritten over \f$b\f$
  template<class L>
  void overwrite_solve(ublas::matrix<T, L>& A, matrix_type& b);

  T det(); //Compute the determinant of factored matrix

 private:
  //! Factored matrix available
  bool        m_factored;

  //! Number of rows of A
  int m_M;
  //! Number of columns of A
  int m_N;
  //! The matrix to invert
  matrix_type m_A;
  //! Leading dimension of A
  int         m_LDA;
  //! Pivot indices
  pv_type     m_IPIV;

  bool m_no_exception ;
};

////////////////////////////////////////////////////////////
// Implementation

template<typename T>
template<class E>
int LUSolver<T>::factor(const ublas::matrix_expression<E>& A) {
  m_A = A;

  // Check that A is non-empty and square
  if( m_A.size1() == 0 ) throw( Warning("Empty matrix", W_EmptyMatrix) );
  if( m_A.size1() != m_A.size2() ) throw( Error("Non-square matrix", E_NonSquareMatrix) );

  // Prepare arguments
  m_M = m_N = m_LDA = m_A.size1();
  m_IPIV.resize(m_N);

  // Factor the system
  int INFO;
  XGETRF<T>::apply(m_M, m_N, &m_A(0, 0), m_LDA, &m_IPIV(0), INFO);

  // Handle the singular case
  if( INFO > 0 ) {
    if(m_no_exception)
      return E_SingularMatrix ;
    else
      throw( Error("Singular matrix", E_SingularMatrix) );
  }
  m_factored = true;
  return E_NoError ;
}

////////////////////////////////////////////////////////////

template<typename T>
template<class E>
typename LUSolver<T>::vector_type LUSolver<T>::solve(const ublas::vector_expression<E>& b) {
  // Check whether the matrix has already been factored
  if( !m_factored ) throw( Error("No factored matrix available", E_NoFactoredMatrix) );

  // Check that b has the correct size
  vector_type x = b();
  if( x.size() != (size_type)m_N ) throw( Error("Inconsistent matrix/rhs sizes", E_InconsistentSizes) );

  // Fortran stores matrices are stored column-wise, and so is matrix_type,
  // so transpose is unnecessary
  char TRANS = 'N';
  int NRHS = 1;
  int INFO;
  XGETRS<T>::apply(TRANS, m_N, NRHS, &m_A(0, 0), m_LDA, &m_IPIV(0),
                   &x(0), m_N, INFO);

  if( INFO > 0 ) throw( Error("Error in substitution", E_SubstitutionError) );
  return x;
}

template<typename T>
template<class E>
typename LUSolver<T>::matrix_type LUSolver<T>::solve(const ublas::matrix_expression<E>& B) {
  // Check whether the matrix has already been factored
  if( !m_factored ) throw( Error("No factored matrix available", E_NoFactoredMatrix) );

  // Check that b has the correct size
  matrix_type X = B();
  if( X.size1() != (size_type)m_N ) throw( Error("Inconsistent matrix/rhs sizes", E_InconsistentSizes) );

  // Fortran stores matrices are stored column-wise, and so is matrix_type,
  // so transpose is unnecessary
  char TRANS = 'N';
  int NRHS = X.size2();
  int INFO;
  XGETRS<T>::apply(TRANS, m_N, NRHS, &m_A(0, 0), m_LDA, &m_IPIV(0),
                   &X(0, 0), m_N, INFO);

  if( INFO > 0 ) throw( Error("Error in substitution", E_SubstitutionError) );
  return X;
}

////////////////////////////////////////////////////////////

template<typename T>
template<class E>
void LUSolver<T>::overwrite_solve(ublas::vector_expression<E>& b) {
  // Check whether the matrix has already been factored
  if( !m_factored ) throw( Error("No factored matrix available", E_NoFactoredMatrix) );

  // Check that b has the correct size
  if( b().size() != (size_type)m_N ) throw( Error("Inconsistent matrix/rhs sizes", E_InconsistentSizes) );

  // Fortran stores matrices are stored column-wise, and so is matrix_type,
  // so transpose is unnecessary
  char TRANS = 'N';
  int NRHS = 1;
  int INFO;
  XGETRS<T>::apply(TRANS, m_N, NRHS, &m_A(0, 0), m_LDA, &m_IPIV(0),
                   &b()(0), m_N, INFO);

  if( INFO > 0 ) throw( Error("Error in substitution", E_SubstitutionError) );
}

template<typename T>
void LUSolver<T>::overwrite_solve(matrix_type& B) {
  // Check whether the matrix has already been factored
  if( !m_factored ) throw( Error("No factored matrix available", E_NoFactoredMatrix) );

  // Check that b has the correct size
  if( B.size1() != (size_type)m_N ) throw( Error("Inconsistent matrix/rhs sizes", E_InconsistentSizes) );

  // Fortran stores matrices are stored column-wise, and so is matrix_type,
  // so transpose is unnecessary
  char TRANS = 'N';
  int NRHS = B.size2();
  int INFO;
  XGETRS<T>::apply(TRANS, m_N, NRHS, &m_A(0, 0), m_LDA, &m_IPIV(0),
                   &B(0, 0), m_N, INFO);

  if( INFO > 0 ) throw( Error("Error in substitution", E_SubstitutionError) );
}

////////////////////////////////////////////////////////////

template<typename T>
template<class L, class E>
typename LUSolver<T>::vector_type LUSolver<T>::solve(ublas::matrix<T, L>& A, const ublas::vector_expression<E>& b) {
  // Check that A is non-empty and square
  if( A.size1() == 0 ) throw( Warning("Empty matrix", W_EmptyMatrix) );
  if( A.size1() != A.size2() ) throw( Error("Non-square matrix", E_NonSquareMatrix) );

  // Prepare arguments
  m_M = m_N = m_LDA = A.size1();
  m_IPIV.resize(m_N);

  // Check that b has the correct size
  vector_type x = b();
  if( x.size() != (size_type)m_N ) throw( Error("Inconsistent matrix/rhs sizes", E_InconsistentSizes) );

  // Factor the system
  int INFO;
  XGETRF<T>::apply(m_M, m_N, &A(0, 0), m_LDA, &m_IPIV(0), INFO);

  // Handle the singular case
  if( INFO > 0 ) throw( Error("Singular matrix", E_SingularMatrix) );

  // Fortran matrices are stored column-wise, so a check is necessary to
  // determine the correct value for the TRANS parameter
  char TRANS = boost::is_same<typename L::orientation_category, ublas::column_major_tag>::value ? 'N' : 'T';
  int NRHS = 1;
  XGETRS<T>::apply(TRANS, m_N, NRHS, &A(0, 0), m_LDA, &m_IPIV(0),
                   &x(0), m_N, INFO);

  if( INFO > 0 ) throw( Error("Error in substitution", E_SubstitutionError) );
  return x;
}

template<typename T>
template<class L, class E>
typename LUSolver<T>::matrix_type LUSolver<T>::solve(ublas::matrix<T, L>& A, const ublas::matrix_expression<E>& b) {
  // Check that A is non-empty and square
  if( A.size1() == 0 ) throw( Warning("Empty matrix", W_EmptyMatrix) );
  if( A.size1() != A.size2() ) throw( Error("Non-square matrix", E_NonSquareMatrix) );

  // Prepare arguments
  m_M = m_N = m_LDA = A.size1();
  m_IPIV.resize(m_N);

  // Check that b has the correct size
  matrix_type X = b();
  if( X.size1() != (size_type)m_N ) throw( Error("Inconsistent matrix/rhs sizes", E_InconsistentSizes) );

  // Factor the system
  int INFO;
  XGETRF<T>::apply(m_M, m_N, &A(0, 0), m_LDA, &m_IPIV(0), INFO);

  // Handle the singular case
  if( INFO > 0 ) throw( Error("Singular matrix", E_SingularMatrix) );

  // Fortran matrices are stored column-wise, so a check is necessary to
  // determine the correct value for the TRANS parameter
  char TRANS = boost::is_same<typename L::orientation_category, ublas::column_major_tag>::value ? 'N' : 'T';
  int NRHS = X.size2();
  XGETRS<T>::apply(TRANS, m_N, NRHS, &A(0, 0), m_LDA, &m_IPIV(0),
                   &X(0, 0), m_N, INFO);

  if( INFO > 0 ) throw( Error("Error in substitution", E_SubstitutionError) );
  return X;
}

////////////////////////////////////////////////////////////

template<typename T>
template<class L>
void LUSolver<T>::overwrite_solve(ublas::matrix<T, L>& A, vector_type& b) {
  // Check that A is non-empty and square
  if( A.size1() == 0 ) throw( Warning("Empty matrix", W_EmptyMatrix) );
  if( A.size1() != A.size2() ) throw( Error("Non-square matrix", E_NonSquareMatrix) );

  // Prepare arguments
  m_M = m_N = m_LDA = A.size1();
  m_IPIV.resize(m_N);

  // Check that b has the correct size
  if( b.size() != (size_type)m_N ) throw( Error("Inconsistent matrix/rhs sizes", E_InconsistentSizes) );

  // Factor the system
  int INFO;
  XGETRF<T>::apply(m_M, m_N, &A(0, 0), m_LDA, &m_IPIV(0), INFO);

  // Handle the singular case
  if( INFO > 0 ) throw( Error("Singular matrix", E_SingularMatrix) );

  // Fortran matrices are stored column-wise, so a check is necessary to
  // determine the correct value for the TRANS parameter
  char TRANS = boost::is_same<typename L::orientation_category, ublas::column_major_tag>::value ? 'N' : 'T';
  int NRHS = 1;
  XGETRS<T>::apply(TRANS, m_N, NRHS, &A(0, 0), m_LDA, &m_IPIV(0),
                   &b(0), m_N, INFO);

  if( INFO > 0 ) throw( Error("Error in substitution", E_SubstitutionError) );
}

template<typename T>
template<class L>
void LUSolver<T>::overwrite_solve(ublas::matrix<T, L>& A, matrix_type& b) {
  // Check that A is non-empty and square
  if( A.size1() == 0 ) throw( Warning("Empty matrix", W_EmptyMatrix) );
  if( A.size1() != A.size2() ) throw( Error("Non-square matrix", E_NonSquareMatrix) );

  // Prepare arguments
  m_M = m_N = m_LDA = A.size1();
  m_IPIV.resize(m_N);

  // Check that b has the correct size
  if( b.size1() != (size_type)m_N ) throw( Error("Inconsistent matrix/rhs sizes", E_InconsistentSizes) );

  // Factor the system
  int INFO;
  XGETRF<T>::apply(m_M, m_N, &A(0, 0), m_LDA, &m_IPIV(0), INFO);

  // Handle the singular case
  if( INFO > 0 ) throw( Error("Singular matrix", E_SingularMatrix) );

  // Fortran matrices are stored column-wise, so a check is necessary to
  // determine the correct value for the TRANS parameter
  char TRANS = boost::is_same<typename L::orientation_category, ublas::column_major_tag>::value ? 'N' : 'T';
  int NRHS = b.size2();
  XGETRS<T>::apply(TRANS, m_N, NRHS, &A(0, 0), m_LDA, &m_IPIV(0),
                   &b(0, 0), m_N, INFO);

  if( INFO > 0 ) throw( Error("Error in substitution", E_SubstitutionError) );
}


////////////////////////////////////////////////////////////

template<typename T>
T LUSolver<T>::det() {
  // Check whether the matrix has already been factored
  if( !m_factored ) throw( Error("No factored matrix availabl", E_NoFactoredMatrix) );

  auto mat_size = m_A.size1();
  T det = m_A(0,0);
  for (std::size_t i = 1 ; i < mat_size ; i++){
    det *= m_A(i,i);
  }
  return det;
}


