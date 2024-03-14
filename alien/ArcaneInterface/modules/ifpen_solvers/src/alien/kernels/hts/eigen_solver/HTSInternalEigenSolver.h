// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*
 * HTSInternalEIGENSolver.h
 *
 *  Created on: 22 déc. 2014
 *      Author: gratienj
 */

#ifndef ALIEN_KERNELS_HTS_EIGENSOLVER_HTSINTERNALEIGENSOLVER_H
#define ALIEN_KERNELS_HTS_EIGENSOLVER_HTSINTERNALEIGENSOLVER_H

#include <alien/utils/Precomp.h>
#include <alien/core/backend/IInternalEigenSolverT.h>
#include <alien/kernels/hts/eigen_solver/HTSEigenOptionTypes.h>
#include <alien/core/backend/EigenSolver.h>
#include <alien/core/backend/EigenSolverT.h>
#include <alien/utils/ObjectWithTrace.h>
#include <alien/kernels/simple_csr/SimpleCSRVector.h>
#include <alien/kernels/simple_csr/SimpleCSRMatrix.h>
#include <alien/AlienIFPENSolversPrecomp.h>

class IOptionsHTSEigenSolver;

namespace Alien {

class ALIEN_IFPEN_SOLVERS_EXPORT HTSInternalEigenSolver
    //: public IInternalEigenSolver<SimpleCSRMatrix<Real>, SimpleCSRVector<Real> >
    : public IGeneralizedEigenSolver,
      public ObjectWithTrace
{
 private:
  typedef IEigenSolver::Status Status;

  typedef SimpleCSRMatrix<Real> CSRMatrixType;
  typedef SimpleCSRVector<Real> CSRVectorType;
  typedef SimpleCSRInternal::MatrixInternal<Real> CSRInternalMatrixType;

 public:
  /** Constructeur de la classe */
  HTSInternalEigenSolver(
      Arccore::MessagePassing::IMessagePassingMng* parallel_mng = nullptr,
      IOptionsHTSEigenSolver* options = nullptr);

  /** Destructeur de la classe */
  virtual ~HTSInternalEigenSolver() {}

 public:
  //! Initialisation
  void init(int argv, char const** argc);
  void init();

  String getBackEndName() const { return "htssolver"; }

  template <typename VectorT>
  bool solve(Alien::EigenProblemT<Alien::BackEnd::tag::simplecsr, VectorT>& problem);
  bool solve(Alien::EigenProblem& problem);

  template <typename VectorT>
  bool solve(
      Alien::GeneralizedEigenProblemT<Alien::BackEnd::tag::simplecsr, VectorT>& problem);
  bool solve(Alien::GeneralizedEigenProblem& problem);

  //! Indicateur de support de résolution parallèle
  bool hasParallelSupport() const { return true; }

  //! Etat du solveur
  const Status& getStatus() const;
  String getName() const { return "htssolver"; }

  //! Etat du solveur
 private:
 protected:
  std::unique_ptr<HartsSolver::HTSSolver> m_hts_solver;

 private:
  //! Structure interne du solveur

  bool m_use_mpi = false;
  Arccore::MessagePassing::IMessagePassingMng* m_parallel_mng = nullptr;
  RunTimeSystem::MachineInfo m_machine_info;
  // Status m_status;
  HartsSolver::HTSSolver::Status m_hts_status;
  Status m_status;

  //! Solver parameters
  Integer m_max_iteration = 0;
  Real m_tol = 0.;

  int m_current_ctx_id = -1;

  Integer m_output_level = 0;

  IOptionsHTSEigenSolver* m_options = nullptr;
};

#ifdef ALIEN_USE_HTSSOLVER
template <typename VectorT>
bool
HTSInternalEigenSolver::solve(
    Alien::EigenProblemT<Alien::BackEnd::tag::simplecsr, VectorT>& problem)
{
  using namespace HartsSolver;

  CSRMatrixType const& A = problem.getA();

  if (m_output_level > 0)
    alien_info([&] { cout() << "HTSEigenSolver::solve"; });
  typedef HartsSolver::CSRProfile MCProfileType;
  typedef HartsSolver::ProfileView MCProfileViewType;
  typedef MCProfileType::PermutationType MCProfilePermType;

  CSRMatrixType::ProfileType const& matrix_profile = A.internal()->getCSRProfile();
  int nrows = matrix_profile.getNRow();
  int const* kcol = matrix_profile.getRowOffset().unguardedBasePointer();
  int const* cols = matrix_profile.getCols().unguardedBasePointer();

  MCProfileType hts_profileA(nrows, kcol, cols);
  typedef HartsSolver::CSRMatrix<Real, 1> MCMatrixType;
  MCMatrixType hts_A(&hts_profileA);
  hts_A.setValues(A.internal()->getDataPtr());

  std::vector<double>& real_eigen_values = problem.getRealEigenValues();
  std::vector<double>& imaginary_eigen_values = problem.getImaginaryEigenValues();
  std::vector<VectorT>& eigen_vectors = problem.getEigenVectors();

  HartsSolver::HTSSolver::Status status;
  HartsSolver::HTSSolver::ContextType& context =
      m_hts_solver->getContext(m_current_ctx_id);
  m_hts_solver->solveEigenProblem(
      hts_A, real_eigen_values, imaginary_eigen_values, eigen_vectors, status, context);
  m_status.m_succeeded = status.error == 0;
  m_status.m_residual = status.residual;
  m_status.m_iteration_count = status.num_iter;
  m_status.m_error = status.error;
  m_status.m_nconv = eigen_vectors.size();

  return m_status.m_succeeded;
}

template <typename VectorT>
bool
HTSInternalEigenSolver::solve(
    Alien::GeneralizedEigenProblemT<Alien::BackEnd::tag::simplecsr, VectorT>& problem)
{
  using namespace HartsSolver;

  CSRMatrixType const& A = problem.getA();
  CSRMatrixType const& B = problem.getB();

  if (m_output_level > 0)
    alien_info([&] { cout() << "HTSEigenSolver::solve"; });
  typedef HartsSolver::CSRProfile MCProfileType;
  typedef HartsSolver::ProfileView MCProfileViewType;
  typedef MCProfileType::PermutationType MCProfilePermType;

  CSRMatrixType::ProfileType const& profileA = A.internal()->getCSRProfile();
  int nrowsA = profileA.getNRow();
  int const* kcolA = profileA.getRowOffset().unguardedBasePointer();
  int const* colsA = profileA.getCols().unguardedBasePointer();
  // MCProfilePermType* profile_permutation = nullptr ;
  MCProfileType hts_profileA(nrowsA, kcolA, colsA);
  typedef HartsSolver::CSRMatrix<Real, 1> MCMatrixType;
  MCMatrixType hts_A(&hts_profileA);
  hts_A.setValues(A.internal()->getDataPtr());

  CSRMatrixType::ProfileType const& profileB = B.internal()->getCSRProfile();
  int nrowsB = profileB.getNRow();
  int const* kcolB = profileB.getRowOffset().unguardedBasePointer();
  int const* colsB = profileB.getCols().unguardedBasePointer();
  MCProfileType hts_profileB(nrowsB, kcolB, colsB);
  MCMatrixType hts_B(&hts_profileB);
  hts_B.setValues(B.internal()->getDataPtr());

  std::vector<double>& real_eigen_values = problem.getRealEigenValues();
  std::vector<double>& imaginary_eigen_values = problem.getImaginaryEigenValues();
  std::vector<VectorT>& eigen_vectors = problem.getEigenVectors();
  HartsSolver::HTSSolver::Status status;
  HartsSolver::HTSSolver::ContextType& context =
      m_hts_solver->getContext(m_current_ctx_id);
  m_hts_solver->solveGeneralizedEigenProblem(hts_A, hts_B, real_eigen_values,
      imaginary_eigen_values, eigen_vectors, status, context);

  m_status.m_succeeded = status.error == 0;
  m_status.m_residual = status.residual;
  m_status.m_iteration_count = status.num_iter;
  m_status.m_error = status.error;
  m_status.m_nconv = eigen_vectors.size();

  return m_status.m_succeeded;
}

#endif

} // namespace Alien

#endif /* HTSEIGENSOLVER_H_ */
