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

void AlinaCSRMatrixView::
checkSizes() const
{
  Int32 nb_row = nbRow();
  Int32 n1 = m_row_indexes.size();
  if (n1 != (nb_row + 1))
    ARCCORE_FATAL("Bad size '{0}' for rowIndexes() (expected value = {1})",n1, nb_row+1);
  Int32 nb_value = m_row_indexes[nb_row];
  Int32 n2 = m_columns.size();
  Int32 n3 = m_values.size();
  if (n2 != nb_value)
    ARCCORE_FATAL("Bad size '{0}' for columns() (expected value = {1})",n2, nb_value);
  if (n3 != nb_value)
    ARCCORE_FATAL("Bad size '{0}' for values() (expected value = {1})",n3, nb_value);
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
AlinaSequentialSolver(const AlinaCSRMatrixView& matrix_view,
                      const AlinaParameters* prm)
{
  matrix_view.checkSizes();
  auto A = std::make_tuple(matrix_view.nbRow(), matrix_view.rowIndexes(),
                           matrix_view.columns(), matrix_view.values());

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
solve(SmallSpan<const double> rhs, SmallSpan<double> x)
{
  SequentialSolverType* slv = m_p->m_solver;

  Alina::SolverResult r = (*slv)(rhs, x);

  return _toConvInfo(r);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AlinaConvergenceInfo AlinaSequentialSolver::
solveMatrix(const AlinaCSRMatrixView& matrix_view,
            SmallSpan<const double> rhs,
            SmallSpan<double> x)
{
  SequentialSolverType* slv = m_p->m_solver;

  Int32 n = slv->size();
  matrix_view.checkSizes();

  if (n != matrix_view.nbRow())
    ARCCORE_FATAL("Bad number of rows v={0} expected={1}", matrix_view.nbRow(), n);

  auto A = std::make_tuple(matrix_view.nbRow(), matrix_view.rowIndexes(),
                           matrix_view.columns(), matrix_view.values());

  Alina::SolverResult r = (*slv)(A, rhs, x);

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
                       const AlinaCSRMatrixView& matrix_view,
                       //ptrdiff_t n,
                       //const int* ptr,
                       //const int* col,
                       //const double* val,
                       int n_def_vec,
                       AlinaDefVecFunction def_vec_func,
                       void* def_vec_data,
                       const AlinaParameters& params)
{
  std::function<double(ptrdiff_t, unsigned)> dv = deflation_vectors(n_def_vec, def_vec_func, def_vec_data);
  Alina::PropertyTree prm = params.m_p->m_properties;
  prm.put("num_def_vec", n_def_vec);
  prm.put("def_vec", &dv);
  matrix_view.checkSizes();

  auto A = std::make_tuple(matrix_view.nbRow(), matrix_view.rowIndexes(),
                           matrix_view.columns(), matrix_view.values());

  Alina::mpi_communicator mpi_comm(comm);
  auto* p = new DistributedSolverType(mpi_comm, A, prm);

  m_p = std::make_shared<AlinaDistributedSolverImpl>(p);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AlinaConvergenceInfo AlinaDistributedSolver::
solve(SmallSpan<const double> rhs, SmallSpan<double> x)
{
  DistributedSolverType* solver = m_p->m_solver;

  size_t n = solver->size();

  AlinaConvergenceInfo cnv;

  Alina::SolverResult r = (*solver)(rhs, x);

  return _toConvInfo(r);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
