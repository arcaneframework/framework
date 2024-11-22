// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#pragma once

#include <alien/utils/Precomp.h>
#include <alien/core/backend/IInternalLinearSolverT.h>
#include <alien/kernels/composyx/linear_solver/ComposyxOptionTypes.h>
#include <alien/expression/solver/SolverStat.h>
#include <alien/core/backend/IInternalLinearSolverT.h>
#include <alien/utils/ObjectWithTrace.h>
#include <alien/kernels/simple_csr/SimpleCSRVector.h>
#include <alien/kernels/simple_csr/SimpleCSRMatrix.h>
#include <alien/AlienComposyxPrecomp.h>

#include <alien/kernels/composyx/data_structure/ComposyxVector.h>
#include <alien/kernels/composyx/data_structure/ComposyxMatrix.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
class IOptionsComposyxSolver;


BEGIN_COMPOSYXINTERNAL_NAMESPACE

template <typename ValueT> class SolverInternal;

END_COMPOSYXINTERNAL_NAMESPACE

namespace Alien {


class ALIEN_COMPOSYX_EXPORT ComposyxInternalSolver
    //: public IInternalLinearSolver<SimpleCSRMatrix<Real>, SimpleCSRVector<Real> >
    : public ILinearSolver,
      public ObjectWithTrace
{
 public:
  typedef SolverStatus Status;

  typedef ComposyxMatrix<Real> ComposyxMatrixType;
  typedef ComposyxVector<Real> ComposyxVectorType;
  typedef ComposyxInternal::SolverInternal<Real> InternalType ;

  /** Constructeur de la classe */
  ComposyxInternalSolver(IMessagePassingMng* parallel_mng, IOptionsComposyxSolver* options);

  /** Destructeur de la classe */
  virtual ~ComposyxInternalSolver() ;

 public:
  //! return package back end name
  String getBackEndName() const { return "composyx"; }
  String getName() const { return "composyx"; }

  void init();
  void init(int argc, char const** argv);

  //! Finalize
  void end();

  void updateParallelMng(IMessagePassingMng* pm) { m_parallel_mng = pm; }

  //! Indicateur de support de r�solution parall�le
  bool hasParallelSupport() const { return true; }

  //! Compatible linear algebra
  std::shared_ptr<Alien::ILinearAlgebra> algebra() const
  {
    return std::shared_ptr<Alien::ILinearAlgebra>();
  }

  //! Etat du solveur
  const Alien::ILinearSolver::Status& getStatus() const { return m_status; }

  const SolverStat& getSolverStat() const { return m_stat; }
  SolverStat& getSolverStat() { return m_stat; }

  bool solve(const Alien::IMatrix& A, const Alien::IVector& b, Alien::IVector& x);

#ifdef ALIEN_USE_COMPOSYX
  bool solve(const ComposyxMatrixType& A,
             const ComposyxVectorType& b,
             ComposyxVectorType& x);
#endif

  void setNullSpaceConstantOption(bool flag) {}

  void internalPrintInfo() const;

 private:

  IMessagePassingMng* m_parallel_mng = nullptr;
  IOptionsComposyxSolver* m_options = nullptr;
#ifdef ALIEN_USE_COMPOSYX
  std::unique_ptr<InternalType> m_internal ;
#endif


  Alien::SolverStat m_stat; //<! Statistiques d'ex�cution du solveur
  Alien::SolverStater<ComposyxInternalSolver> m_stater;
  Alien::ILinearSolver::Status m_status;
  Real m_init_solver_time = 0.;
  Real m_iter_solver_time = 0.;
  Integer m_output_level = 0;
};
}

