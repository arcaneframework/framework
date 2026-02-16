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

#ifdef ALIEN_USE_SYCL
#include <alien/kernels/sycl/SYCLPrecomp.h>
#include "alien/kernels/sycl/data/SYCLEnv.h"
#include <alien/kernels/sycl/data/SYCLBEllPackMatrix.h>
#include <alien/kernels/sycl/data/SYCLVector.h>
#include <alien/kernels/sycl/algebra/SYCLLinearAlgebra.h>
#include <alien/kernels/sycl/algebra/SYCLInternalLinearAlgebra.h>
#endif

class IOptionsAlienCoreSolver;

namespace Alien {

class SolverStat;

class ALIEN_CORE_SYCL_SOLVERS_EXPORT AlienCoreSYCLLinearSolver
: public AlienCoreSolverBaseT<Alien::SYCLInternalLinearAlgebra>
{
 private:
  typedef SolverStatus Status;

  typedef Alien::SYCLInternalLinearAlgebra::Matrix MatrixType;
  typedef Alien::SYCLInternalLinearAlgebra::Vector VectorType;

 public:
  typedef AlienCoreSolverBaseT<Alien::SYCLInternalLinearAlgebra> BaseType ;

  /** Constructeur de la classe */
  AlienCoreSYCLLinearSolver(
      Arccore::MessagePassing::IMessagePassingMng* parallel_mng = nullptr,
      IOptionsAlienCoreSolver* options = nullptr);

  /** Destructeur de la classe */
  virtual ~AlienCoreSYCLLinearSolver() {}

 public:
  String getBackEndName() const { return "sycl"; }
  String getName() const { return "AlienCoreSYCLSolver"; }


  void init(int argv, char const** argc) ;

  void init() ;

  void setDiagScaling(IMatrix const& A) ;
  void setDiagScaling(MatrixType const& A) ;

  bool solve(IMatrix const& A, IVector const& b, IVector& x);
  bool solve(const MatrixType& A, const VectorType& b, VectorType& x);

  bool solve();

  //! Indicateur de support de résolution parallèle
  bool hasParallelSupport() const { return true; }

  std::shared_ptr<ILinearAlgebra> algebra() const;

  //! Etat du solveur
  const Alien::SolverStatus& getStatus() const;

  const SolverStat& getSolverStat() const { return m_stat; }
  SolverStat& getSolverStat() { return m_stat; }

  void printCurrentTimeInfo() {}


 protected:
  SolverStat m_stat;
  SolverStater<AlienCoreSYCLLinearSolver> m_stater;

};

} // namespace Alien

