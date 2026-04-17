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

#include <mpi.h>

// Convergence info
struct ARCCORE_ALINA_EXPORT AlinaConvergenceInfo
{
  int iterations;
  double residual;
};

typedef double(* AlinaDefVecFunction)(int vec, ptrdiff_t coo, void* data);

//! Handle parameters.
struct AlinaParameters;

//! Handle preconditioner
struct AlinaPreconditioner;

//! Sequential solver;
struct AlinaSequentialSolver;

//! Distributed solver;
struct AlinaDistributedSolver;

class ARCCORE_ALINA_EXPORT AlinaLib
{
 public:

  // Set integer parameter in a parameter list.
  static void params_set_int(AlinaParameters* prm, const char* name, int value);

  // Set floating point parameter in a parameter list.
  static void params_set_float(AlinaParameters* prm, const char* name, float value);

  // Set floating point parameter in a parameter list.
  static void params_set_string(AlinaParameters* prm, const char* name, const char* value);

  // Read parameters from a JSON file
  static void params_read_json(AlinaParameters* prm, const char* fname);

  // Destroy parameter list.
  static void params_destroy(AlinaParameters* prm);

  // Create parameter list.
  static AlinaParameters* params_create();

  // Create AMG preconditioner.
  static AlinaPreconditioner* preconditioner_create(int n,
                                                          const int* ptr,
                                                          const int* col,
                                                          const double* val,
                                                          AlinaParameters* parameters);

  // Apply AMG preconditioner (x = M^(-1) * rhs).
  static void preconditioner_apply(AlinaPreconditioner* amg, const double* rhs, double* x);

  // Printout preconditioner structure
  static void preconditioner_report(AlinaPreconditioner* amg);

  // Destroy AMG preconditioner
  static void preconditioner_destroy(AlinaPreconditioner* amg);

  // Create iterative solver preconditioned by AMG.
  static AlinaSequentialSolver* solver_create(int n,
                                              const int* ptr,
                                              const int* col,
                                              const double* val,
                                              AlinaParameters* parameters);

  // Solve the problem for the given right-hand side.
  static AlinaConvergenceInfo solver_solve(AlinaSequentialSolver* solver,
                                double const* rhs,
                                double* x);

  // Solve the problem for the given matrix and the right-hand side.
  static AlinaConvergenceInfo solver_solve_matrix(AlinaSequentialSolver* solver,
                                       int const* A_ptr,
                                       int const* A_col,
                                       double const* A_val,
                                       double const* rhs,
                                       double* x);

  // Printout solver structure
  static void solver_report(AlinaSequentialSolver* solver);

  // Destroy iterative solver.
  static void solver_destroy(AlinaSequentialSolver* solver);

  // Create distributed solver.
  static AlinaDistributedSolver* solver_mpi_create(MPI_Comm comm,
                                                   ptrdiff_t n,
                                                   const ptrdiff_t* ptr,
                                                   const ptrdiff_t* col,
                                                   const double* val,
                                                   int n_def_vec,
                                                   AlinaDefVecFunction def_vec_func,
                                                   void* def_vec_data,
                                                   AlinaParameters* params);

  // Find solution for the given RHS.
  static AlinaConvergenceInfo solver_mpi_solve(AlinaDistributedSolver* solver,
                                    double const* rhs,
                                    double* x);

  // Destroy the distributed solver.
  static void solver_mpi_destroy(AlinaDistributedSolver* solver);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
