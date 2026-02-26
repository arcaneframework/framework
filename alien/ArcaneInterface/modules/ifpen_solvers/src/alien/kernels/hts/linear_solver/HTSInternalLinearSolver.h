// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <alien/utils/Precomp.h>
#include <alien/core/backend/IInternalLinearSolverT.h>
#include <alien/kernels/hts/linear_solver/HTSOptionTypes.h>
#include <alien/expression/solver/SolverStater.h>
#include <alien/core/backend/IInternalLinearSolverT.h>
#include <alien/utils/ObjectWithTrace.h>
#include <alien/kernels/simple_csr/SimpleCSRVector.h>
#include <alien/kernels/simple_csr/SimpleCSRMatrix.h>
#include <alien/AlienIFPENSolversPrecomp.h>

class IOptionsHTSSolver;

namespace Alien {

class SolverStat;

class ALIEN_IFPEN_SOLVERS_EXPORT HTSInternalLinearSolver
    //: public IInternalLinearSolver<SimpleCSRMatrix<Real>, SimpleCSRVector<Real> >
: public ILinearSolver
, public ILinearSolverWithDiagScaling
, public ObjectWithTrace
{
 private:
  typedef SolverStatus Status;

  typedef SimpleCSRMatrix<Real> CSRMatrixType;
  typedef SimpleCSRVector<Real> CSRVectorType;
  typedef SimpleCSRInternal::MatrixInternal<Real> CSRInternalMatrixType;

 public:
  /** Constructeur de la classe */
  HTSInternalLinearSolver(
      Arccore::MessagePassing::IMessagePassingMng* parallel_mng = nullptr,
      IOptionsHTSSolver* options = nullptr);

  /** Destructeur de la classe */
  virtual ~HTSInternalLinearSolver() ;

 public:
  //! Initialisation
  void init(int argv, char const** argc);
  void init();
  void updateParallelMng(Arccore::MessagePassing::IMessagePassingMng* pm);
  void updateParameters();

  // void setDiagScal(double* Diag, int size);
  //! Finalize
  void end();

  String getBackEndName() const { return "htssolver"; }

  void setDiagScaling(IMatrix const& A) ;
  void setDiagScaling(CSRMatrixType const& A) ;

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

  String getName() const { return "htssolver"; }

  //! Etat du solveur
  void setNullSpaceConstantOption(bool flag)
  {
    alien_warning([&] { cout() << "Null Space Constant Option not yet implemented"; });
  }

  void printInfo() const;
  void printInfo();
  void printCurrentTimeInfo() {}



 private:
  void updateLinearSystem();
  inline void _startPerfCount();
  inline void _endPerfCount();

 protected:

 private:
  //! Structure interne du solveur

  struct Impl ;
  std::unique_ptr<Impl> m_impl ;

  Integer m_output_level = 0;

  bool m_use_mpi = false;
  bool m_diag_scaling_is_set = false ;
  Arccore::MessagePassing::IMessagePassingMng* m_parallel_mng = nullptr;
  Alien::SolverStatus m_status;

  Integer m_solve_num = 0;
  Integer m_total_iter_num = 0;
  Real m_current_solve_time = 0.;
  Real m_total_solve_time = 0.;
  Real m_current_system_time = 0.;
  Real m_total_system_time = 0.;
  Real m_int_total_solve_time = 0.;
  Real m_int_total_setup_time = 0.;
  Real m_int_total_finish_time = 0.;

  SolverStat m_stat;
  SolverStater<HTSInternalLinearSolver> m_stater;

  IOptionsHTSSolver* m_options = nullptr;
  std::vector<double> m_pressure_diag;
};

} // namespace Alien

