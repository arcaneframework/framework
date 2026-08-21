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

#include <memory>

#include <mpi.h>

// Convergence info
struct ARCCORE_ALINA_EXPORT AlinaConvergenceInfo
{
  int iterations = 0;
  double residual = 0.0;
};

typedef double (*AlinaDefVecFunction)(int vec, ptrdiff_t coo, void* data);

class AlinaLib;
class AlinaPreconditioner;
class AlinaParametersImpl;
class AlinaPreconditionerImpl;
class AlinaSequentialSolver;
class AlinaSequentialSolverImpl;
class AlinaDistributedSolver;
class AlinaDistributedSolverImpl;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Handle parameters for Alina.
 *
 * This class uses a reference semantic for copy.
 */
class ARCCORE_ALINA_EXPORT AlinaParameters
{
  friend AlinaLib;
  friend AlinaPreconditioner;
  friend AlinaSequentialSolver;
  friend AlinaDistributedSolver;

 public:

  AlinaParameters();

 public:

  //! Set Int32 parameter in the parameter list
  void setInt32(const char* name, Arcane::Int32 value);

  //! Set Int64 parameter in the parameter list
  void setInt64(const char* name, Arcane::Int64 value);

  //! Set floating point parameter in the parameter list
  void setReal(const char* name, Arcane::Real value);

  //! Set floating point parameter in the parameter list
  void setString(const char* name, const char* value);

  //! Read parameters from a JSON file
  void readFromJSON(const char* fname);

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

  using Int32 = Arcane::Int32;

 public:

  AlinaCSRMatrixView(Int32 nb_row, const Int32* row_indexes, const Int32* columns, const double* values)
  : m_nb_row(nb_row)
  , m_row_indexes(row_indexes, nb_row + 1)
  , m_columns(columns, row_indexes[nb_row])
  , m_values(values, row_indexes[nb_row])
  {}
  AlinaCSRMatrixView(Arcane::SmallSpan<const Int32> row_indexes,
                     Arcane::SmallSpan<const Int32> columns, Arcane::SmallSpan<const double> values)
  : m_nb_row(row_indexes.size() - 1)
  , m_row_indexes(row_indexes)
  , m_columns(columns)
  , m_values(values)
  {}

 public:

  constexpr Int32 nbRow() const { return m_nb_row; }
  constexpr Arcane::SmallSpan<const Int32> rowIndexes() const { return m_row_indexes; }
  constexpr Arcane::SmallSpan<const Int32> columns() const { return m_columns; }
  constexpr Arcane::SmallSpan<const double> values() const { return m_values; }

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
  Arcane::SmallSpan<const Int32> m_row_indexes;
  Arcane::SmallSpan<const Int32> m_columns;
  Arcane::SmallSpan<const double> m_values;
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
  friend AlinaLib;

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
  AlinaConvergenceInfo solve(double const* rhs,
                             double* x);

  //! Solve the problem for the given matrix and the right-hand side.
  AlinaConvergenceInfo solveMatrix(const AlinaCSRMatrixView& matrix_view,
                                   double const* rhs,
                                   double* x);

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
  AlinaDistributedSolver(MPI_Comm comm,
                         const AlinaCSRMatrixView& matrix_view,
                         int n_def_vec,
                         AlinaDefVecFunction def_vec_func,
                         void* def_vec_data,
                         const AlinaParameters& params);

  //! Find solution for the given RHS.
  AlinaConvergenceInfo solve(double const* rhs, double* x);

 public:

  std::shared_ptr<AlinaDistributedSolverImpl> m_p;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
