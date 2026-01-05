// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#pragma once

#include <alien/utils/Precomp.h>
#include <alien/core/backend/IInternalLinearSolverT.h>
#include <alien/expression/solver/SolverStater.h>
#include <alien/core/backend/IInternalLinearSolverT.h>
#include <alien/utils/ObjectWithTrace.h>
#include <alien/kernels/simple_csr/SimpleCSRVector.h>
#include <alien/kernels/simple_csr/SimpleCSRMatrix.h>
#include <alien/kernels/simple_csr/algebra/SimpleCSRLinearAlgebra.h>
#include <alien/kernels/simple_csr/algebra/SimpleCSRInternalLinearAlgebra.h>
#include <alien/kernels/common/AlienCoreSolverBaseT.h>

class IOptionsAlienCoreSolver;

namespace Alien {

class SolverStat;

class ALIEN_CORE_SOLVERS_EXPORT AlienCoreLinearSolver
: public AlienCoreSolverBaseT<Alien::SimpleCSRInternalLinearAlgebra>
{
 private:
  typedef SolverStatus Status;

  typedef SimpleCSRMatrix<Real> CSRMatrixType;
  typedef SimpleCSRVector<Real> CSRVectorType;
  typedef SimpleCSRInternal::MatrixInternal<Real> CSRInternalMatrixType;

 public:
  typedef AlienCoreSolverBaseT<Alien::SimpleCSRInternalLinearAlgebra> BaseType ;

  /** Constructeur de la classe */
  AlienCoreLinearSolver(
      Arccore::MessagePassing::IMessagePassingMng* parallel_mng = nullptr,
      IOptionsAlienCoreSolver* options = nullptr);

  /** Destructeur de la classe */
  virtual ~AlienCoreLinearSolver() {}

 public:
  String getBackEndName() const { return "simple_csr"; }
  String getName() const { return "AlienCoreSolver"; }


  void init(int argv, char const** argc) ;

  void init() ;

  bool solve(IMatrix const& A, IVector const& b, IVector& x);
  bool solve(const CSRMatrixType& A, const CSRVectorType& b, CSRVectorType& x);

  bool solve();

  //! Indicateur de support de résolution parallèle
  bool hasParallelSupport() const { return true; }

  std::shared_ptr<ILinearAlgebra> algebra() const;

  //! Etat du solveur
  const Alien::SolverStatus& getStatus() const;

  const SolverStat& getSolverStat() const { return m_stat; }
  SolverStat& getSolverStat() { return m_stat; }

 protected:
  SolverStat m_stat;
  SolverStater<AlienCoreLinearSolver> m_stater;

};

} // namespace Alien

