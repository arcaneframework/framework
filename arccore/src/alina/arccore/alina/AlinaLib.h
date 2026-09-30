// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AlinaLib.h                                                  (C) 2000-2026 */
/*                                                                           */
/* Public API for Alina.                                      .              */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_ALINA_ALINALIB_H
#define ARCCORE_ALINA_ALINALIB_H
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

#include "arccore/alina/AlinaGlobal.h"

#include "arccore/base/Span.h"
#include "arccore/message_passing/MessagePassingGlobal.h"

#include <memory>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::AlinaLib
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Convergence info
struct ARCCORE_ALINA_EXPORT AlinaConvergenceInfo
{
  int iterations = 0;
  double residual = 0.0;
};

class AlinaPreconditioner;
class AlinaParametersImpl;
class AlinaPreconditionerImpl;
class AlinaSequentialSolver;
class AlinaSequentialSolverImpl;
class AlinaDistributedSolver;
class AlinaDistributedSolverImpl;
class AlinaSequentialSolverImpl;
class AlinaDistributedSolverImpl;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Type of supported solver
enum class eAlinaSolverType
{
  ConjugateGradient,
  BiCGStab,
  GMRES
};

//! Type of supported preconditioner
enum class eAlinaPreconditionerType
{
  AMG,
  Diagonal
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Handle parameters for Alina.
 *
 * This class uses a reference semantic for copy.
 */
class ARCCORE_ALINA_EXPORT AlinaParameters
{
  friend AlinaPreconditioner;
  friend AlinaSequentialSolver;
  friend AlinaDistributedSolver;
  friend AlinaSequentialSolverImpl;
  friend AlinaDistributedSolverImpl;

 public:

  AlinaParameters();

 public:

  //! Set Int32 parameter in the parameter list
  void setInt32(const char* name, Int32 value);

  //! Set Int64 parameter in the parameter list
  void setInt64(const char* name, Int64 value);

  //! Set floating point parameter in the parameter list
  void setReal(const char* name, Real value);

  //! Set floating point parameter in the parameter list
  void setString(const char* name, const char* value);

  //! Read parameters from a JSON file
  void readFromJSON(const char* fname);

 public:

  // Options specific to solvers
  void setSolverAbsoluteTolerance(Real value);
  void setSolverRelativeTolerance(Real value);
  void setSolverMaxIteration(Int32 value);
  void setSolverVerbosity(Int32 value);

  void setSolverPreconditioner(eAlinaPreconditionerType v);
  void setSolverType(eAlinaSolverType v);

 private:

  std::shared_ptr<AlinaParametersImpl> m_p;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief CSR Matrix view used in AlinaLib.
 */
class ARCCORE_ALINA_EXPORT AlinaCSRMatrixView
{
 public:

  AlinaCSRMatrixView(Int32 nb_row, const Int32* row_indexes, const Int32* columns, const double* values)
  : m_nb_row(nb_row)
  , m_row_indexes(row_indexes, nb_row + 1)
  , m_columns(columns, row_indexes[nb_row])
  , m_values(values, row_indexes[nb_row])
  {}
  AlinaCSRMatrixView(SmallSpan<const Int32> row_indexes,
                     SmallSpan<const Int32> columns, SmallSpan<const double> values)
  : m_nb_row(row_indexes.size() - 1)
  , m_row_indexes(row_indexes)
  , m_columns(columns)
  , m_values(values)
  {}

 public:

  constexpr Int32 nbRow() const { return m_nb_row; }
  constexpr SmallSpan<const Int32> rowIndexes() const { return m_row_indexes; }
  constexpr SmallSpan<const Int32> columns() const { return m_columns; }
  constexpr SmallSpan<const double> values() const { return m_values; }

 public:

  /*!
   * \brief Check that sizes are valid:
   * - rowIndexes().size() = nbRow() + 1;
   * - columns().size() = rowIndexes[nbRow()];
   * - values().size() = rowIndexes[nbRow()];
   */
  void checkSizes() const;

 private:

  Int32 m_nb_row = 0;
  SmallSpan<const Int32> m_row_indexes;
  SmallSpan<const Int32> m_columns;
  SmallSpan<const double> m_values;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Handle preconditioner for a solver
 *
 * This class uses a reference semantic for copy.
 */
class ARCCORE_ALINA_EXPORT AlinaPreconditioner
{
 public:

  AlinaPreconditioner(int n,
                      const int* ptr,
                      const int* col,
                      const double* val,
                      const AlinaParameters* prm);

 public:

  //! Apply AMG preconditioner (x = M^(-1) * rhs).
  void apply(const double* rhs, double* x);

  //! Printout preconditioner structure
  void report();

 private:

  std::shared_ptr<AlinaPreconditionerImpl> m_p;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Sequential solver.
 */
class ARCCORE_ALINA_EXPORT AlinaSequentialSolver
{
 public:

  /*!
   * \brief Build a solver for matrix \a matrix.
   *
   * The matrix view \a matrix_view passed as arguments must remain valid
   * for as long as this instance is alive.
   * \a parameters may be null. In this case we use the default parameters.
   */
  AlinaSequentialSolver(const AlinaCSRMatrixView& matrix_view,
                        const AlinaParameters* parameters);

 public:

  //! Solve the problem for the given right-hand side.
  AlinaConvergenceInfo solve(SmallSpan<const double> rhs,
                             SmallSpan<double> x);

  //! Solve the problem for the given matrix and the right-hand side.
  AlinaConvergenceInfo solveMatrix(const AlinaCSRMatrixView& matrix_view,
                                   SmallSpan<const double> rhs,
                                   SmallSpan<double> x);

  //! Printout solver structure
  void report();

 private:

  std::shared_ptr<AlinaSequentialSolverImpl> m_p;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Distributed solver.
 */
class ARCCORE_ALINA_EXPORT AlinaDistributedSolver
{
 public:

  /*!
   * \brief Create distributed solver.
   *
   * The matrix view \a matrix_view passed as arguments must remain valid
   * for as long as this instance is alive.
   */
  AlinaDistributedSolver(MessagePassing::IMessagePassingMng* comm,
                         const AlinaCSRMatrixView& matrix_view,
                         const AlinaParameters& params);

  //! Find solution for the given RHS.
  AlinaConvergenceInfo solve(SmallSpan<const double> rhs,
                             SmallSpan<double> x);

 public:

  std::shared_ptr<AlinaDistributedSolverImpl> m_p;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::AlinaLib

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
