// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Matrix.h                                                    (C) 2000-2025 */
/*                                                                           */
/* Matrix d'algèbre linéraire.                                               */
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
 * \brief Matrice avec stockage CSR.
 *
 * Les matrices fonctionnent par référence
 */
class ARCANE_CORE_EXPORT Matrix
{
 public:

  Matrix()
  : m_impl(0)
  {}
  Matrix(Integer nb_row, Integer nb_column);
  Matrix(const Matrix& rhs);
  ~Matrix();
  void operator=(const Matrix& rhs);

 private:

  Matrix(MatrixImpl* impl);

 public:

  //! Clone la matrice
  Matrix clone() const;

 public:

  // Nombre de lignes de la matrice
  Integer nbRow() const;
  // Nombre de colonnes de la matrice
  Integer nbColumn() const;
  //! Positionne le nombre d'éléments non nuls de chaque ligne
  void setRowsSize(IntegerConstArrayView rows_size);
  //! Positionne les valeurs des éléments de la matrice
  void setValues(IntegerConstArrayView columns, RealConstArrayView values);
  //! Imprime la matrice
  void dump(std::ostream& o) const;
  //! Valeurs de la matrice
  RealConstArrayView values() const;
  //! Valeurs de la matrice
  RealArrayView values();
  //! Indices des premiers éléments de chaque ligne
  IntegerConstArrayView rowsIndex() const;
  //! Indices des colonnes des valeurs
  IntegerConstArrayView columns() const;
  //! Indices des premiers éléments de chaque ligne
  IntegerArrayView rowsIndex();
  //! Indices des colonnes des valeurs
  IntegerArrayView columns();
  //! Positionne la valeur d'un élément de la matrice
  void setValue(Integer row, Integer column, Real value);
  //! Retourne la valeur d'un élément de la matrice
  Real value(Integer row, Integer column) const;
  //! Arrange le stockage pour que la diagonale soit le premier élément
  void sortDiagonale();
  //! Lit la matrice au format X Y
  static Matrix read(const String& filename);
  //! Lit la matrice au format Hypre
  static Matrix readHypre(const String& filename);

 private:

  //! Implémentation
  MatrixImpl* m_impl;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface d'un préconditionneur.
 *
 * Le préconditionneur prend un vecteur en entrée (en général le résidu)
 * et le multiplie à une approximation de la matrice inverse du
 * système linéaire.
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

  ConjugateGradientSolver()
  : m_nb_iteration(0)
  , m_residual_norm(0.0)
  , m_max_iteration(5000)
  {}
  bool solve(const Matrix& a, const Vector& b, Vector& x, Real epsilon, IPreconditioner* p = 0);
  Integer nbIteration() const { return m_nb_iteration; }
  Real residualNorm() const { return m_residual_norm; }
  void setMaxIteration(Integer max_iteration)
  {
    m_max_iteration = max_iteration;
  }

 private:

  Integer m_nb_iteration;
  Real m_residual_norm;
  Integer m_max_iteration;

 private:

  void _applySolver(const Matrix& a, const Vector& b, Vector& x, Real epsilon, IPreconditioner* p);
  void _applySolver2(const Matrix& a, const Vector& b, Vector& x, Real epsilon, IPreconditioner* precond);
  void _applySolverAsHypre(const Matrix& a, const Vector& b, Vector& x, Real tol, IPreconditioner* precond);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Préconditionneur diagonal.
 *
 * Ce préconditionneur calcule une approximation de l'inverse
 * d'une matrice \a M en ne prenant que sa diagonale et en l'inversant.
 */
class ARCANE_CORE_EXPORT DiagonalPreconditioner
: public IPreconditioner
{
 public:

  DiagonalPreconditioner(const Matrix& matrix);

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
  // Applique le produit de matrice L * M * R avec L = transpose(R)
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

  AMGPreconditioner(ITraceMng* tm)
  : m_trace_mng(tm)
  , m_amg(0)
  {}
  virtual ~AMGPreconditioner();

 public:

  virtual void build(const Matrix& matrix);

 public:

  virtual void apply(Vector& out_vec, const Vector& vec);

 private:

  ITraceMng* m_trace_mng;
  AMG* m_amg;
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

  ITraceMng* m_trace_mng;
  AMG* m_amg;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Solveur direct utilisant le pivot de gauss.
 *
 * A utiliser uniquement pour les petites matrices (<1000 lignes) sinon le
 * temps de calcul devient très important.
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
