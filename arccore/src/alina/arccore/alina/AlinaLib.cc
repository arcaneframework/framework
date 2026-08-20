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

#include "arccore/concurrency/Mutex.h"

#include <iostream>

using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//using Backend = Alina::BuiltinBackend<double>;
using Backend = Alina::BuiltinBackend<double,Int32,Int32>;
using PreconditionerType = Alina::AMG<Backend, Alina::CoarseningRuntime, Alina::RelaxationRuntime>;
using SequentialSolverType = Alina::PreconditionedSolver<PreconditionerType, Alina::SolverRuntime<Backend>>;
typedef Alina::PropertyTree Params;

//---------------------------------------------------------------------------

using DistributedSolverType = Alina::DistributedSubDomainDeflation<PreconditionerType,
                                                                   Alina::DistributedSolverRuntime<Backend>,
                                                                   Alina::DistributedDirectSolverRuntime<Backend>>;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
AlinaConvergenceInfo
_toConvInfo(const Alina::SolverResult& r)
{
  AlinaConvergenceInfo x;
  x.iterations = r.nbIteration();
  x.residual = r.residual();
  return x;
}
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class AlinaParametersImpl
{
 public:

  Params m_properties;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AlinaParameters::
AlinaParameters()
: m_p(std::make_shared<AlinaParametersImpl>())
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AlinaParameters::
setInt32(const char* name, Arcane::Int32 value)
{
  m_p->m_properties.put(name, value);
}

void AlinaParameters::
setInt64(const char* name, Arcane::Int64 value)
{
  m_p->m_properties.put(name, value);
}

void AlinaParameters::
setReal(const char* name, Arcane::Real value)
{
  m_p->m_properties.put(name, value);
}

void AlinaParameters::
setString(const char* name, const char* value)
{
  m_p->m_properties.put(name, value);
}

void AlinaParameters::
readFromJSON(const char* fname)
{
  Params& p = m_p->m_properties;
  p.read_json(fname);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class AlinaPreconditionerImpl
{
 public:

  explicit AlinaPreconditionerImpl(PreconditionerType* preconditioner)
  : m_preconditioner(preconditioner)
  {}
  ~AlinaPreconditionerImpl()
  {
    delete m_preconditioner;
  }

  PreconditionerType* m_preconditioner = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AlinaPreconditioner::
AlinaPreconditioner(int n,
                    const int* ptr,
                    const int* col,
                    const double* val,
                    const AlinaParameters* prm)
{
  SmallSpan<const int> ptr_range(ptr, n + 1);
  SmallSpan<const int> col_range(col, ptr[n]);
  SmallSpan<const double> val_range(val, ptr[n]);

  auto A = std::make_tuple(n, ptr_range, col_range, val_range);

  PreconditionerType* amg = nullptr;
  if (prm)
    amg = new PreconditionerType(A, prm->m_p->m_properties);
  else
    amg = new PreconditionerType(A);
  m_p = std::make_shared<AlinaPreconditionerImpl>(amg);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AlinaPreconditioner::
apply(const double* rhs, double* x)
{
  PreconditionerType* amg = m_p->m_preconditioner;

  size_t n = Alina::backend::nbRow(amg->system_matrix());

  SmallSpan<double> x_range(x, n);
  SmallSpan<const double> rhs_range(rhs, n);

  amg->apply(rhs_range, x_range);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AlinaPreconditioner::
report()
{
  std::cout << *(m_p->m_preconditioner) << std::endl;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

struct AlinaSequentialSolverImpl
{
  explicit AlinaSequentialSolverImpl(SequentialSolverType* solver)
  : m_solver(solver)
  {}
  ~AlinaSequentialSolverImpl()
  {
    delete m_solver;
  }
  SequentialSolverType* m_solver = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AlinaSequentialSolver::
AlinaSequentialSolver(int n, const int* ptr,
                      const int* col,
                      const double* val,
                      const AlinaParameters* prm)
{
  SmallSpan<const int> ptr_range(ptr, n + 1);
  SmallSpan<const int> col_range(col, ptr[n]);
  SmallSpan<const double> val_range(val, ptr[n]);

  auto A = std::make_tuple(n, ptr_range, col_range, val_range);

  auto* solver = new SequentialSolverType(A);
  if (prm)
    solver = new SequentialSolverType(A, prm->m_p->m_properties);
  else
    solver = new SequentialSolverType(A);
  std::cout << "Printing solver infos\n";
  std::cout << (*solver) << std::endl;
  Alina::PropertyTree ptree;
  solver->prm.get(ptree);
  std::cout << "SOLVER_PARAMS: " << ptree << "\n";
  m_p = std::make_shared<AlinaSequentialSolverImpl>(solver);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AlinaSequentialSolver::
report()
{
  SequentialSolverType* slv = m_p->m_solver;

  std::cout << slv->precond() << std::endl;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AlinaConvergenceInfo AlinaSequentialSolver::
solve(const double* rhs, double* x)
{
  SequentialSolverType* slv = m_p->m_solver;

  size_t n = slv->size();

  SmallSpan<double> x_range(x, n);
  SmallSpan<const double> rhs_range(rhs, n);

  Alina::SolverResult r = (*slv)(rhs_range, x_range);

  return _toConvInfo(r);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AlinaConvergenceInfo AlinaSequentialSolver::
solveMatrix(int const* A_ptr,
                    int const* A_col,
                    double const* A_val,
                    const double* rhs,
                    double* x)
{
  SequentialSolverType* slv = m_p->m_solver;

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


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class AlinaDistributedSolverImpl
{
 public:

  explicit AlinaDistributedSolverImpl(DistributedSolverType* solver)
  : m_solver(solver)
  {}
  ~AlinaDistributedSolverImpl()
  {
    delete m_solver;
  }
  DistributedSolverType* m_solver = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AlinaDistributedSolver::
AlinaDistributedSolver(MPI_Comm comm,
                       ptrdiff_t n,
                       const int* ptr,
                       const int* col,
                       const double* val,
                       int n_def_vec,
                       AlinaDefVecFunction def_vec_func,
                       void* def_vec_data,
                       const AlinaParameters& params)
{
  std::function<double(ptrdiff_t, unsigned)> dv = deflation_vectors(n_def_vec, def_vec_func, def_vec_data);
  Alina::PropertyTree prm = params.m_p->m_properties;
  prm.put("num_def_vec", n_def_vec);
  prm.put("def_vec", &dv);

  SmallSpan<const int> ptr_range(ptr, n + 1);
  SmallSpan<const int> col_range(col, ptr[n]);
  SmallSpan<const double> val_range(val, ptr[n]);

  auto A = std::make_tuple(n, ptr_range, col_range, val_range);
  Alina::mpi_communicator mpi_comm(comm);
  auto* p = new DistributedSolverType(mpi_comm, A, prm);

  m_p = std::make_shared<AlinaDistributedSolverImpl>(p);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AlinaConvergenceInfo AlinaDistributedSolver::
solve(double const* rhs, double* x)
{
  DistributedSolverType* solver = m_p->m_solver;

  size_t n = solver->size();

  SmallSpan<double> x_range(x, n);
  SmallSpan<const double> rhs_range(rhs, n);

  AlinaConvergenceInfo cnv;

  Alina::SolverResult r = (*solver)(rhs_range, x_range);

  return _toConvInfo(r);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
