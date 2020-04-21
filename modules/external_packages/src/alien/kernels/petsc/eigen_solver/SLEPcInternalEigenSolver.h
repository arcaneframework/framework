/*
 * PETSCInternalEIGENSolver.h
 *
 *  Created on: 22 déc. 2014
 *      Author: gratienj
 */

#ifndef ALIEN_KERNELS_PETSC_EIGENSOLVER_PETSCINTERNALEIGENSOLVER_H
#define ALIEN_KERNELS_PETSC_EIGENSOLVER_PETSCINTERNALEIGENSOLVER_H

#include <alien/utils/Precomp.h>
#include <alien/AlienExternalPackagesPrecomp.h>
#include <alien/core/backend/IInternalEigenSolverT.h>
#include <alien/kernels/petsc/eigen_solver/SLEPcEigenOptionTypes.h>
#include <alien/core/backend/EigenSolver.h>
#include <alien/core/backend/EigenSolverT.h>
#include <alien/utils/ObjectWithTrace.h>
#include <alien/kernels/simple_csr/SimpleCSRVector.h>
#include <alien/kernels/simple_csr/SimpleCSRMatrix.h>

class IOptionsSLEPcEigenSolver;

namespace Alien {

class PETScMatrix;
class PETScVector;

struct SLEPC
{
#ifdef ALIEN_USE_SLEPC
  static bool m_initialized;
#endif

  static void initialize(bool is_io_master);
  static void finalize();
};

class ALIEN_EXTERNAL_PACKAGES_EXPORT SLEPcInternalEigenSolver
    //: public IInternalEigenSolver<SimpleCSRMatrix<Real>, SimpleCSRVector<Real> >
    : public IGeneralizedEigenSolver,
      public ObjectWithTrace
{
 private:
  typedef IEigenSolver::Status Status;

 public:
  /** Constructeur de la classe */
  SLEPcInternalEigenSolver(
      Arccore::MessagePassing::IMessagePassingMng* parallel_mng = nullptr,
      IOptionsSLEPcEigenSolver* options = nullptr);

  /** Destructeur de la classe */
  virtual ~SLEPcInternalEigenSolver() {}

 public:
  //! Initialisation
  void init(int argv, char const** argc);
  void init();

  Arccore::String getBackEndName() const { return "petsc"; }

  template <typename VectorT>
  bool solve(Alien::EigenProblemT<Alien::BackEnd::tag::petsc, VectorT>& problem);
  bool solve(Alien::EigenProblem& problem);

  template <typename VectorT>
  bool solve(
      Alien::GeneralizedEigenProblemT<Alien::BackEnd::tag::petsc, VectorT>& problem);
  bool solve(Alien::GeneralizedEigenProblem& problem);

  //! Indicateur de support de résolution parallèle
  bool hasParallelSupport() const { return true; }

  //! Etat du solveur
  const Status& getStatus() const;
  Arccore::String getName() const { return "SLEPcEigenSolver"; }

  //! Etat du solveur
 private:
  class SLEPcStatus
  {
   public:
    bool m_succeed = false;
    int m_error = -1;
    int m_nconv = 0;
    int m_nb_iter = 0;
    double m_residual = -1;
  };

  typedef enum { Arnoldi, KrylovSchur, Arpack } eSolverType;

  int _solve(Arccore::Integer local_size, PETScMatrix const& A,
      std::vector<double>& r_eigen_values, std::vector<double>& i_eigen_values,
      std::vector<std::vector<double>>& eigen_vectors, SLEPcStatus& status);

  int _solve(Arccore::Integer local_size, PETScMatrix const& A, PETScMatrix const& B,
      std::vector<double>& r_eigen_values, std::vector<double>& i_eigen_values,
      std::vector<std::vector<double>>& eigen_vectors, SLEPcStatus& status);

 protected:
 private:
  //! Structure interne du solveur

  Status m_status;

  //! Solver parameters
  Arccore::Integer m_max_iteration = 0;
  Arccore::Real m_tol = 0.;
  eSolverType m_evtype = Arpack;
  Arccore::Integer m_nev = 0;
  Arccore::Integer m_evorder = 0;
  Arccore::Real m_evbound = 0;
  Arccore::Integer m_output_level = 0;

  Arccore::Integer m_ncv = 0;
  int m_max_it = 0;
  int m_mpd = 0;
  int m_nconv = 0;

  IOptionsSLEPcEigenSolver* m_options = nullptr;
  Arccore::MessagePassing::IMessagePassingMng* m_parallel_mng = nullptr;
};

#ifdef ALIEN_USE_SLEPC
template <typename VectorT>
bool
SLEPcInternalEigenSolver::solve(
    Alien::EigenProblemT<Alien::BackEnd::tag::petsc, VectorT>& problem)
{

  Arccore::Integer local_size = problem.localSize();

  PETScMatrix const& A = problem.getA();

  if (m_output_level > 0)
    alien_info([&] { cout() << "PETSCEigenSolver::solve"; });

  std::vector<double>& real_eigen_values = problem.getRealEigenValues();
  std::vector<double>& imaginary_eigen_values = problem.getImaginaryEigenValues();
  std::vector<VectorT>& eigen_vectors = problem.getEigenVectors();

  SLEPcStatus status;
  _solve(local_size, A, real_eigen_values, imaginary_eigen_values, eigen_vectors, status);

  m_status.m_succeeded = status.m_error == 0;
  m_status.m_residual = status.m_residual;
  m_status.m_iteration_count = status.m_nb_iter;
  m_status.m_error = status.m_error;
  m_status.m_nconv = eigen_vectors.size();

  return m_status.m_succeeded;
}

template <typename VectorT>
bool
SLEPcInternalEigenSolver::solve(
    Alien::GeneralizedEigenProblemT<Alien::BackEnd::tag::petsc, VectorT>& problem)
{

  Arccore::Integer local_size = problem.localSize();
  PETScMatrix const& A = problem.getA();
  PETScMatrix const& B = problem.getB();

  std::vector<double>& real_eigen_values = problem.getRealEigenValues();
  std::vector<double>& imaginary_eigen_values = problem.getImaginaryEigenValues();
  std::vector<VectorT>& eigen_vectors = problem.getEigenVectors();

  SLEPcStatus status;
  _solve(
      local_size, A, B, real_eigen_values, imaginary_eigen_values, eigen_vectors, status);

  m_status.m_succeeded = status.m_error == 0;
  m_status.m_residual = status.m_residual;
  m_status.m_iteration_count = status.m_nb_iter;
  m_status.m_error = status.m_error;
  m_status.m_nconv = eigen_vectors.size();

  return m_status.m_succeeded;
}

#endif

} // namespace Alien

#endif /* PETSCEIGENSOLVER_H_ */
