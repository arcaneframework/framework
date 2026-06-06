// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Matrix.h                                                    (C) 2000-2026 */
/*                                                                           */
/* Linear algebra matrix.                                                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_MATVEC_MATRIX_H
#define ARCANE_CORE_MATVEC_MATRIX_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Array.h"
#include "arcane/utils/Numeric.h"

#include "arcane/core/matvec/Vector.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MatVec
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class Vector;
class MatrixImpl;
class IPreconditioner;
class AMG;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Matrix with CSR storage.
 *
 * Matrices operate by reference
 */
class ARCANE_CORE_EXPORT Matrix
{
 public:

  Matrix() = default;
  Matrix(Integer nb_row, Integer nb_column);
  Matrix(const Matrix& rhs);
  ~Matrix();
  void operator=(const Matrix& rhs);

 private:

  explicit Matrix(MatrixImpl* impl);

 public:

  //! Clone the matrix
  Matrix clone() const;

 public:

  // Number of rows of the matrix
  Integer nbRow() const;
  // Number of columns of the matrix
  Integer nbColumn() const;
  //! Sets the number of non-zero elements for each row
  void setRowsSize(IntegerConstArrayView rows_size);
  //! Sets the values of the matrix elements
  void setValues(IntegerConstArrayView columns, RealConstArrayView values);
  //! Prints the matrix
  void dump(std::ostream& o) const;
  //! Matrix values
  RealConstArrayView values() const;
  //! Matrix values
  RealArrayView values();
  //! Indices of the first elements of each row
  IntegerConstArrayView rowsIndex() const;
  //! Column indices of the values
  IntegerConstArrayView columns() const;
  //! Indices of the first elements of each row
  IntegerArrayView rowsIndex();
  //! Column indices of the values
  IntegerArrayView columns();
  //! Sets the value of a matrix element
  void setValue(Integer row, Integer column, Real value);
  //! Returns the value of a matrix element
  Real value(Integer row, Integer column) const;
  //! Arranges the storage so that the diagonal is the first element
  void sortDiagonale();
  //! Reads the matrix in X Y format
  static Matrix read(const String& filename);
  //! Reads the matrix in Hypre format
  static Matrix readHypre(const String& filename);

 private:

  //! Implementation
  MatrixImpl* m_impl = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Interface for a preconditioner.
 *
 * The preconditioner takes a vector as input (generally the residual)
 * and multiplies it by an approximation of the inverse matrix of the
 * linear system.
 * 
 */
class ARCANE_CORE_EXPORT IPreconditioner
{
 public:

  virtual ~IPreconditioner() {}

 public:

  virtual void apply(Vector& out_vec, const Vector& vec) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_CORE_EXPORT ConjugateGradientSolver
{
 public:

  ConjugateGradientSolver() = default;
  bool solve(const Matrix& a, const Vector& b, Vector& x, Real epsilon, IPreconditioner* p = 0);
  Integer nbIteration() const { return m_nb_iteration; }
  Real residualNorm() const { return m_residual_norm; }
  void setMaxIteration(Integer max_iteration)
  {
    m_max_iteration = max_iteration;
  }

 private:

  Integer m_nb_iteration = 0;
  Real m_residual_norm = 0.0;
  Integer m_max_iteration = 5000;

 private:

  void _applySolver(const Matrix& a, const Vector& b, Vector& x, Real epsilon, IPreconditioner* p);
  void _applySolver2(const Matrix& a, const Vector& b, Vector& x, Real epsilon, IPreconditioner* precond);
  void _applySolverAsHypre(const Matrix& a, const Vector& b, Vector& x, Real tol, IPreconditioner* precond);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Diagonal preconditioner.
 *
 * This preconditioner calculates an approximation of the inverse
 * of a matrix A M by only taking its diagonal and inverting it.
 */
class ARCANE_CORE_EXPORT DiagonalPreconditioner
: public IPreconditioner
{
 public:

  explicit DiagonalPreconditioner(const Matrix& matrix);

 public:

  virtual void apply(Vector& out_vec, const Vector& vec);

 private:

  Vector m_inverse_diagonal;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_CORE_EXPORT MatrixOperation
{
 public:

  void matrixVectorProduct(const Matrix& mat, const Vector& vec, Vector& out_vec);
  void matrixVectorProduct2(const Matrix& mat, const Vector& vec, Vector& out_vec);
  Real dot(const Vector& vec);
  Real dot(const Vector& vec1, const Vector& vec2);
  void negateVector(Vector& vec);
  void scaleVector(Vector& vec, Real mul);
  void addVector(Vector& out_vec, const Vector& vec);
};

class ARCANE_CORE_EXPORT MatrixOperation2
{
 public:

  Matrix matrixMatrixProduct(const Matrix& left_matrix, const Matrix& right_matrix);
  Matrix matrixMatrixProductFast(const Matrix& left_matrix, const Matrix& right_matrix);
  Matrix transpose(const Matrix& matrix);
  Matrix transposeFast(const Matrix& matrix);
  // Applies the matrix product L * M * R with L = transpose(R)
  Matrix applyGalerkinOperator(const Matrix& left_matrix, const Matrix& matrix, const Matrix& right_matrix);
  Matrix applyGalerkinOperator2(const Matrix& left_matrix, const Matrix& matrix,
                                const Matrix& right_matrix);

 private:

  void _dumpColumnMatrix(std::ostream& o, IntegerConstArrayView columns_index,
                         IntegerConstArrayView rows,
                         RealConstArrayView values);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_CORE_EXPORT AMGPreconditioner
: public IPreconditioner
{
 public:

  explicit AMGPreconditioner(ITraceMng* tm)
  : m_trace_mng(tm)
  {}
  ~AMGPreconditioner() override;

 public:

  virtual void build(const Matrix& matrix);

 public:

  void apply(Vector& out_vec, const Vector& vec) override;

 private:

  ITraceMng* m_trace_mng = nullptr;
  AMG* m_amg = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_CORE_EXPORT AMGSolver
{
 public:

  AMGSolver(ITraceMng* tm)
  : m_trace_mng(tm)
  , m_amg(0)
  {}
  virtual ~AMGSolver();

 public:

  virtual void build(const Matrix& matrix);

 public:

  virtual void solve(const Vector& vector_b, Vector& vector_x);

 private:

  ITraceMng* m_trace_mng = nullptr;
  AMG* m_amg = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Direct solver using Gaussian pivot.
 *
 * To be used only for small matrices (<1000 rows), otherwise the
 * computation time becomes very important.
 */
class ARCANE_CORE_EXPORT DirectSolver
{
 public:

  void solve(const Matrix& matrix, const Vector& vector_b, Vector& vector_x);

 private:

  void _solve(RealArrayView mat_values, RealArrayView vec_values, Integer size);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::MatVec

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
