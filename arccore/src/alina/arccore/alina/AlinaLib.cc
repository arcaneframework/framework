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

#include "AlinaUtils.h"
#include "arccore/alina/RelaxationRuntime.h"
#include "arccore/alina/CoarseningRuntime.h"
#include "arccore/alina/SolverRuntime.h"
#include "arccore/alina/PreconditionedSolver.h"
#include "arccore/alina/PreconditionerRuntime.h"
#include "arccore/alina/DistributedSolverRuntime.h"
#include "arccore/alina/DistributedPreconditioner.h"
#include "arccore/alina/BuiltinBackend.h"
#include "arccore/alina/AlinaLib.h"
#include "arccore/alina/DistributedPreconditionedSolver.h"

#include "arccore/message_passing/IMessagePassingMng.h"

#include "arccore/trace/ITraceMng.h"

#include "arccore/base/NotSupportedException.h"
#include "arccore/base/NotImplementedException.h"

#include <iostream>
#include <memory>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::AlinaLib
{
using Arcane::IMessagePassingMng;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//using Backend = Alina::BuiltinBackend<double>;
using Backend = Alina::BuiltinBackend<double, Int32, Int32>;
using PreconditionerType = Alina::PreconditionerRuntime<Backend>;
using SequentialSolverType = Alina::PreconditionedSolver<PreconditionerType, Alina::SolverRuntime<Backend>>;

//---------------------------------------------------------------------------

using DistributedSolverType = Alina::DistributedPreconditionedSolver<Alina::DistributedPreconditioner<Backend>,
                                                                     Alina::DistributedSolverRuntime<Backend>>;

typedef double (*AlinaDefVecFunction)(int vec, ptrdiff_t coo, void* data);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

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
} // namespace

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AlinaCSRMatrixView::
dump(std::ostream& o) const
{
  const AlinaCSRMatrixView& matrix_view = *this;
  Int32 nb_row = matrix_view.nbRow();
  const auto default_precision{ o.precision() };
  constexpr auto max_precision{ std::numeric_limits<long double>::digits10 + 1 };
  o << std::setprecision(max_precision);
  o << "MATRIX nb_row=" << nb_row << "\n";
  auto rows_index = matrix_view.rowIndexes();
  auto columns = matrix_view.columns();
  auto values = matrix_view.values();
  for (Int32 i = 0; i < nb_row; ++i) {
    Int32 begin = rows_index[i];
    Int32 end = rows_index[i + 1];
    o << "ROW I=" << i << " index0=" << begin << " index1=" << end << "\n";
    for (Int32 z = begin; z < end; ++z) {
      o << "  C=" << std::setw(8) << columns[z] << "  V=" << values[z] << "\n";
    }
  }
  o << std::setprecision(default_precision);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class AlinaParametersImpl
{
 public:

  Alina::PropertyTree m_properties;
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
  Alina::PropertyTree& p = m_p->m_properties;
  p.read_json(fname);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class AlinaSolverParametersImpl
{
 public:

  AlinaSolverParametersImpl(MessagePassing::IMessagePassingMng* mpm, ITraceMng* tm)
  {
    if (mpm)
      m_message_passing_mng = makeRef(mpm);
    if (!tm)
      tm = arccoreCreateDefaultTraceMng();
    m_trace_mng = makeRef(tm);
  }

 public:

  Alina::PropertyTree m_properties;
  Ref<ITraceMng> m_trace_mng;
  Ref<IMessagePassingMng> m_message_passing_mng;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AlinaSolverParameters::
AlinaSolverParameters()
{
  m_p = std::make_shared<AlinaSolverParametersImpl>(nullptr,nullptr);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AlinaSolverParameters::
AlinaSolverParameters(MessagePassing::IMessagePassingMng* mpm, ITraceMng* tm)
{
  m_p = std::make_shared<AlinaSolverParametersImpl>(mpm,tm);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AlinaSolverParameters::
~AlinaSolverParameters()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ITraceMng* AlinaSolverParameters::
traceMng() const
{
  return m_p->m_trace_mng.get();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AlinaSolverParameters::
setInt32(const char* name, Arcane::Int32 value)
{
  m_p->m_properties.put(name, value);
}

void AlinaSolverParameters::
setInt64(const char* name, Arcane::Int64 value)
{
  m_p->m_properties.put(name, value);
}

void AlinaSolverParameters::
setReal(const char* name, Arcane::Real value)
{
  m_p->m_properties.put(name, value);
}

void AlinaSolverParameters::
setString(const char* name, const char* value)
{
  m_p->m_properties.put(name, value);
}

void AlinaSolverParameters::
readFromJSON(const char* fname)
{
  Alina::PropertyTree& p = m_p->m_properties;
  p.read_json(fname);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AlinaSolverParameters::
setSolverAbsoluteTolerance(Arcane::Real value)
{
  m_p->m_properties.put("solver.abstol", value);
}
void AlinaSolverParameters::
setSolverRelativeTolerance(Arcane::Real value)
{
  m_p->m_properties.put("solver.tol", value);
}
void AlinaSolverParameters::
setSolverMaxIteration(Arcane::Int32 value)
{
  m_p->m_properties.put("solver.maxiter", value);
}
void AlinaSolverParameters::
setSolverVerbosity(Arcane::Int32 value)
{
  m_p->m_properties.put("solver.verbose", value);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AlinaSolverParameters::
setSolverPreconditioner(eAlinaPreconditionerType v)
{
  auto& x = m_p->m_properties;
  switch (v) {
  case eAlinaPreconditionerType::AMG:
    x.put("precond.class", "amg");
    break;
  case eAlinaPreconditionerType::Diagonal:
    x.put("precond.class", "relaxation");
    x.put("precond.relax.type", "spai0");
    break;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AlinaSolverParameters::
setSolverType(eAlinaSolverType v)
{
  const char* name = nullptr;
  switch (v) {
  case eAlinaSolverType::ConjugateGradient:
    name = "cg";
    break;
  case eAlinaSolverType::BiCGStab:
    name = "bicgstab";
    break;
  case eAlinaSolverType::GMRES:
    name = "gmres";
    break;
  }
  if (!name)
    ARCCORE_THROW(NotSupportedException, "Invalid value '{0}' for solver type", static_cast<int>(v));
  m_p->m_properties.put("solver.type", name);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AlinaCSRMatrixView::
checkSizes() const
{
  Int32 nb_row = nbRow();
  Int32 n1 = m_row_indexes.size();
  if (n1 != (nb_row + 1))
    ARCCORE_FATAL("Bad size '{0}' for rowIndexes() (expected value = {1})", n1, nb_row + 1);
  Int32 nb_value = m_row_indexes[nb_row];
  Int32 n2 = m_columns.size();
  Int32 n3 = m_values.size();
  if (n2 != nb_value)
    ARCCORE_FATAL("Bad size '{0}' for columns() (expected value = {1})", n2, nb_value);
  if (n3 != nb_value)
    ARCCORE_FATAL("Bad size '{0}' for values() (expected value = {1})", n3, nb_value);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

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

class IAlinaSolverImpl
{
 public:

  virtual ~IAlinaSolverImpl() = default;

 public:

  virtual AlinaConvergenceInfo
  solve(SmallSpan<const double> rhs, SmallSpan<double> x) = 0;

  virtual AlinaConvergenceInfo
  solveMatrix(const AlinaCSRMatrixView& matrix_view,
              SmallSpan<const double> rhs,
              SmallSpan<double> x) = 0;

  virtual void report(std::ostream& o) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Implementation for a sequential solver.
 */
class AlinaSequentialSolverImpl
: public IAlinaSolverImpl
{
 public:

  explicit AlinaSequentialSolverImpl(const AlinaCSRMatrixView& matrix_view,
                                     const AlinaSolverParameters& prm)
  {
    matrix_view.checkSizes();

    auto A = std::make_tuple(matrix_view.nbRow(), matrix_view.rowIndexes(),
                             matrix_view.columns(), matrix_view.values());

    m_solver = new SequentialSolverType(A, prm.m_p->m_properties);
    ITraceMng* tm = prm.traceMng();
    tm->info() << "Printing solver infos\n"
               << (*m_solver) << "\n"
               << "PARAMS: " << prm.m_p->m_properties;
    Alina::PropertyTree ptree;
    m_solver->prm.get(ptree);
    tm->info() << "SOLVER_PARAMS: " << ptree;
  }
  ~AlinaSequentialSolverImpl() override
  {
    delete m_solver;
  }

  AlinaConvergenceInfo solve(SmallSpan<const double> rhs, SmallSpan<double> x) override
  {
    Alina::SolverResult r = (*m_solver)(rhs, x);
    return _toConvInfo(r);
  }
  AlinaConvergenceInfo solveMatrix(const AlinaCSRMatrixView& matrix_view,
                                   SmallSpan<const double> rhs,
                                   SmallSpan<double> x) override
  {
    SequentialSolverType* slv = m_solver;

    Int32 n = slv->size();
    matrix_view.checkSizes();

    if (n != matrix_view.nbRow())
      ARCCORE_FATAL("Bad number of rows v={0} expected={1}", matrix_view.nbRow(), n);

    auto A = std::make_tuple(matrix_view.nbRow(), matrix_view.rowIndexes(),
                             matrix_view.columns(), matrix_view.values());

    Alina::SolverResult r = (*slv)(A, rhs, x);

    return _toConvInfo(r);
  }
  void report(std::ostream& o) override
  {
    o << m_solver->precond() << "\n";
  }

 public:

  SequentialSolverType* m_solver = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Implementation for a distributed solver.
 */
class AlinaDistributedSolverImpl
: public IAlinaSolverImpl
{
 public:

  explicit AlinaDistributedSolverImpl(const AlinaCSRMatrixView& matrix_view,
                                      const AlinaSolverParameters& params)
  {
    IMessagePassingMng* mpm = params.m_p->m_message_passing_mng.get();
    matrix_view.checkSizes();

    Alina::PropertyTree& prm = params.m_p->m_properties;

    auto A = std::make_tuple(matrix_view.nbRow(), matrix_view.rowIndexes(),
                             matrix_view.columns(), matrix_view.values());

    Alina::AlinaCommunicator mpi_comm(mpm);
    m_solver = new DistributedSolverType(mpi_comm, A, prm);

    ITraceMng* tm = params.traceMng();
    tm->info() << "Printing solver infos\n"
               << (*m_solver) << "\n"
               << "PARAMS: " << params.m_p->m_properties;
    Alina::PropertyTree ptree;
    m_solver->prm.get(ptree);
    tm->info() << "SOLVER_PARAMS: " << ptree;
  }
  ~AlinaDistributedSolverImpl()
  {
    delete m_solver;
  }

 public:

  AlinaConvergenceInfo solve(SmallSpan<const double> rhs, SmallSpan<double> x) override
  {
    AlinaConvergenceInfo cnv;

    Alina::SolverResult r = (*m_solver)(rhs, x);

    return _toConvInfo(r);
  }

  AlinaConvergenceInfo solveMatrix([[maybe_unused]] const AlinaCSRMatrixView& matrix_view,
                                   [[maybe_unused]] SmallSpan<const double> rhs,
                                   [[maybe_unused]] SmallSpan<double> x) override
  {
    ARCCORE_THROW(NotImplementedException, "Solve with different matrix");
  }
  void report([[maybe_unused]] std::ostream& o) override
  {
  }

 public:

  DistributedSolverType* m_solver = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AlinaSolver::
AlinaSolver(const AlinaSolverParameters& parameters, const AlinaCSRMatrixView& matrix_view)
{
  IMessagePassingMng* mpm = parameters.m_p->m_message_passing_mng.get();
  if (mpm) {
    m_p = std::make_shared<AlinaDistributedSolverImpl>(matrix_view, parameters);
  }
  else
    m_p = std::make_shared<AlinaSequentialSolverImpl>(matrix_view, parameters);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AlinaSolver::
report()
{
  m_p->report(std::cout);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AlinaConvergenceInfo AlinaSolver::
solve(SmallSpan<const double> rhs, SmallSpan<double> x)
{
  return m_p->solve(rhs, x);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AlinaConvergenceInfo AlinaSolver::
solveMatrix(const AlinaCSRMatrixView& matrix_view,
            SmallSpan<const double> rhs,
            SmallSpan<double> x)
{
  return m_p->solveMatrix(matrix_view, rhs, x);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::AlinaLib

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
