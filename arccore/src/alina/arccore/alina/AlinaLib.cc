// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AlinaLib.cc                                                 (C) 2000-2026 */
/*                                                                           */
/* Public API for Alina.                                      .              */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/alina/RelaxationRuntime.h"
#include "arccore/alina/CoarseningRuntime.h"
#include "arccore/alina/SolverRuntime.h"
#include "arccore/alina/PreconditionedSolver.h"
#include "arccore/alina/DistributedSolverRuntime.h"
#include "arccore/alina/DistributedDirectSolverRuntime.h"
#include "arccore/alina/DistributedSubDomainDeflation.h"
#include "arccore/alina/AMG.h"
#include "arccore/alina/BuiltinBackend.h"
#include "arccore/alina/AlinaLib.h"

#include <iostream>

using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using Backend = Alina::BuiltinBackend<double>;
//using Backend = Alina::BuiltinBackend<double,Int32,Int32>;
using PreconditionerType = Alina::AMG<Backend, Alina::CoarseningRuntime, Alina::RelaxationRuntime>;
using SequentialSolverType = Alina::PreconditionedSolver<PreconditionerType, Alina::SolverRuntime<Backend>>;
typedef Alina::PropertyTree Params;

//---------------------------------------------------------------------------

using DistributedSolverType = Alina::DistributedSubDomainDeflation<PreconditionerType,
                                                                   Alina::DistributedSolverRuntime<Backend>,
                                                                   Alina::DistributedDirectSolverRuntime<Backend>>;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

struct AlinaParameters
{
  Params m_properties;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

struct AlinaPreconditioner
{
  explicit AlinaPreconditioner(PreconditionerType* preconditioner)
  : m_preconditioner(preconditioner)
  {}
  ~AlinaPreconditioner()

  {
    delete m_preconditioner;
  }
  PreconditionerType* m_preconditioner = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

struct AlinaSequentialSolver
{
  explicit AlinaSequentialSolver(SequentialSolverType* solver)
  : m_solver(solver)
  {}
  ~AlinaSequentialSolver()
  {
    delete m_solver;
  }
  SequentialSolverType* m_solver = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

struct AlinaDistributedSolver
{
  explicit AlinaDistributedSolver(DistributedSolverType* solver)
  : m_solver(solver)
  {}
  ~AlinaDistributedSolver()
  {
    delete m_solver;
  }
  DistributedSolverType* m_solver = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AlinaConvergenceInfo
_toConvInfo(const Alina::SolverResult& r)
{
  AlinaConvergenceInfo x;
  x.iterations = r.nbIteration();
  x.residual = r.residual();
  return x;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AlinaParameters* AlinaLib::
params_create()
{
  return new AlinaParameters();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AlinaLib::
params_set_int(AlinaParameters* prm, const char* name, int value)
{
  prm->m_properties.put(name, value);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AlinaLib::
params_set_float(AlinaParameters* prm, const char* name, float value)
{
  prm->m_properties.put(name, value);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AlinaLib::
params_set_string(AlinaParameters* prm, const char* name, const char* value)
{
  prm->m_properties.put(name, value);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AlinaLib::
params_read_json(AlinaParameters* prm, const char* fname)
{
  Params& p = prm->m_properties;
  p.read_json(fname);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AlinaLib::
params_destroy(AlinaParameters* prm)
{
  delete prm;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AlinaPreconditioner* AlinaLib::
preconditioner_create(int n,
                      const int* ptr,
                      const int* col,
                      const double* val,
                      AlinaParameters* prm)
{
  SmallSpan<const int> ptr_range(ptr, n + 1);
  SmallSpan<const int> col_range(col, ptr[n]);
  SmallSpan<const double> val_range(val, ptr[n]);

  auto A = std::make_tuple(n, ptr_range, col_range, val_range);

  PreconditionerType* amg = nullptr;
  if (prm)
    amg = new PreconditionerType(A, prm->m_properties);
  else
    amg = new PreconditionerType(A);
  return new AlinaPreconditioner(amg);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AlinaLib::
preconditioner_apply(AlinaPreconditioner* handle, const double* rhs, double* x)
{
  PreconditionerType* amg = handle->m_preconditioner;

  size_t n = Alina::backend::nbRow(amg->system_matrix());

  SmallSpan<double> x_range(x, n);
  SmallSpan<const double> rhs_range(rhs, n);

  amg->apply(rhs_range, x_range);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AlinaLib::
preconditioner_report(AlinaPreconditioner* handle)
{
  std::cout << *(handle->m_preconditioner) << std::endl;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AlinaLib::
preconditioner_destroy(AlinaPreconditioner* handle)
{
  delete handle;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AlinaSequentialSolver* AlinaLib::
solver_create(int n, const int* ptr,
              const int* col,
              const double* val,
              AlinaParameters* prm)
{
  SmallSpan<const int> ptr_range(ptr, n + 1);
  SmallSpan<const int> col_range(col, ptr[n]);
  SmallSpan<const double> val_range(val, ptr[n]);

  auto A = std::make_tuple(n, ptr_range, col_range, val_range);

  SequentialSolverType* solver = new SequentialSolverType(A);
  if (prm)
    solver = new SequentialSolverType(A, prm->m_properties);
  else
    solver = new SequentialSolverType(A);
  std::cout << "Printing solver infos\n";
  std::cout << (*solver) << std::endl;
  Alina::PropertyTree ptree;
  solver->prm.get(ptree);
  std::cout << "SOLVER_PARAMS: " << ptree << "\n";
  return new AlinaSequentialSolver(solver);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AlinaLib::
solver_report(AlinaSequentialSolver* handle)
{
  SequentialSolverType* slv = handle->m_solver;

  std::cout << slv->precond() << std::endl;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AlinaLib::
solver_destroy(AlinaSequentialSolver* handle)
{
  delete handle;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AlinaConvergenceInfo AlinaLib::
solver_solve(AlinaSequentialSolver* handle,
             const double* rhs,
             double* x)
{
  SequentialSolverType* slv = handle->m_solver;

  size_t n = slv->size();

  SmallSpan<double> x_range(x, n);
  SmallSpan<const double> rhs_range(rhs, n);

  Alina::SolverResult r = (*slv)(rhs_range, x_range);

  return _toConvInfo(r);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AlinaConvergenceInfo AlinaLib::
solver_solve_matrix(AlinaSequentialSolver* handle,
                    int const* A_ptr,
                    int const* A_col,
                    double const* A_val,
                    const double* rhs,
                    double* x)
{
  SequentialSolverType* slv = handle->m_solver;

  size_t n = slv->size();

  SmallSpan<double> x_range(x, n);
  SmallSpan<const double> rhs_range(rhs, n);

  SmallSpan<const int> ptr_range(A_ptr, n + 1);
  SmallSpan<const int> col_range(A_col, A_ptr[n]);
  SmallSpan<const double> val_range(A_val, A_ptr[n]);

  auto A = std::make_tuple(n, ptr_range, col_range, val_range);

  Alina::SolverResult r = (*slv)(A, rhs_range, x_range);

  return _toConvInfo(r);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

struct deflation_vectors
{
  int n;
  AlinaDefVecFunction user_func;
  void* user_data;

  deflation_vectors(int n, AlinaDefVecFunction user_func, void* user_data)
  : n(n)
  , user_func(user_func)
  , user_data(user_data)
  {}

  int dim() const { return n; }

  double operator()(int i, ptrdiff_t j) const
  {
    return user_func(i, j, user_data);
  }
};

//---------------------------------------------------------------------------
AlinaDistributedSolver* AlinaLib::
solver_mpi_create(MPI_Comm comm,
                  ptrdiff_t n,
                  const ptrdiff_t* ptr,
                  const ptrdiff_t* col,
                  const double* val,
                  int n_def_vec,
                  AlinaDefVecFunction def_vec_func,
                  void* def_vec_data,
                  AlinaParameters* params)
{
  std::function<double(ptrdiff_t, unsigned)> dv = deflation_vectors(n_def_vec, def_vec_func, def_vec_data);
  Alina::PropertyTree prm = params->m_properties;
  prm.put("num_def_vec", n_def_vec);
  prm.put("def_vec", &dv);

  SmallSpan<const ptrdiff_t> ptr_range(ptr, n + 1);
  SmallSpan<const ptrdiff_t> col_range(col, ptr[n]);
  SmallSpan<const double> val_range(val, ptr[n]);

  auto A = std::make_tuple(n, ptr_range, col_range, val_range);
  Alina::mpi_communicator mpi_comm(comm);
  auto* p = new DistributedSolverType(mpi_comm, A, prm);

  return new AlinaDistributedSolver(p);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AlinaConvergenceInfo AlinaLib::
solver_mpi_solve(AlinaDistributedSolver* handle,
                 double const* rhs,
                 double* x)
{
  DistributedSolverType* solver = handle->m_solver;

  size_t n = solver->size();

  SmallSpan<double> x_range(x, n);
  SmallSpan<const double> rhs_range(rhs, n);

  AlinaConvergenceInfo cnv;

  std::tie(cnv.iterations, cnv.residual) = (*solver)(rhs_range, x_range);

  return cnv;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AlinaLib::
solver_mpi_destroy(AlinaDistributedSolver* handle)
{
  delete handle;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
