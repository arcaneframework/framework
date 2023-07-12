/*
 * TRILINOSInternalEIGENSolver.h
 *
 *  Created on: 22 déc. 2014
 *      Author: gratienj
 */

#ifndef ALIEN_KERNELS_TRILINOS_EIGENSOLVER_TRILINOSINTERNALEIGENSOLVER_H
#define ALIEN_KERNELS_TRILINOS_EIGENSOLVER_TRILINOSINTERNALEIGENSOLVER_H

#include <alien/utils/Precomp.h>
#include <alien/AlienTrilinosPrecomp.h>
#include <alien/kernels/trilinos/eigen_solver/TrilinosEigenOptionTypes.h>
#include <alien/expression/solver/IEigenSolver.h>
//#include <alien/core/backend/IInternalEigenSolverT.h>
#include <alien/core/backend/EigenSolver.h>
//#include <alien/core/backend/EigenSolverT.h>

#include <alien/utils/ObjectWithTrace.h>
#include <alien/kernels/trilinos/data_structure/TrilinosVector.h>
#include <alien/kernels/trilinos/data_structure/TrilinosMatrix.h>

class IOptionsTrilinosEigenSolver;

namespace Alien {

class ALIEN_TRILINOS_EXPORT TrilinosInternalEigenSolver
    //: public IInternalEigenSolver<SimpleCSRMatrix<Real>, SimpleCSRVector<Real> >
    : public IGeneralizedEigenSolver,
      public ObjectWithTrace
{
 private:
  typedef IEigenSolver::Status Status;

 public:
  /** Constructeur de la classe */
  TrilinosInternalEigenSolver(
      Arccore::MessagePassing::IMessagePassingMng* parallel_mng = nullptr,
      IOptionsTrilinosEigenSolver* options = nullptr);

  /** Destructeur de la classe */
  virtual ~TrilinosInternalEigenSolver() {}

 public:
  //! Initialisation
  void init(int argv, char const** argc);
  void init();

  String getBackEndName() const { return "tpetraserial"; }

  template <typename VectorT>
  bool solve(Alien::EigenProblemT<Alien::BackEnd::tag::tpetraserial, VectorT>& problem);
  bool solve(Alien::EigenProblem& problem);

  template <typename VectorT>
  bool solve(Alien::GeneralizedEigenProblemT<Alien::BackEnd::tag::tpetraserial, VectorT>&
          problem);
  bool solve(Alien::GeneralizedEigenProblem& problem);

  //! Indicateur de support de résolution parallèle
  bool hasParallelSupport() const { return true; }

  //! Etat du solveur
  const Status& getStatus() const;

  String getName() const { return "trilinos"; }

  //! Etat du solveur
 private:
 protected:
 private:
  //! Structure interne du solveur

  bool m_use_mpi = false;
  Arccore::MessagePassing::IMessagePassingMng* m_parallel_mng = nullptr;
  Status m_status;

  //! Solver parameters
  Integer m_max_iteration = 0;
  Real m_tol = 0.;

  int m_current_ctx_id = -1;

  Integer m_output_level = 0;

  IOptionsTrilinosEigenSolver* m_options = nullptr;
};

#ifdef ALIEN_USE_TRILINOSSOLVER
template <typename VectorT>
bool
TrilinosInternalEigenSolver::solve(
    Alien::EigenProblemT<Alien::BackEnd::tag::simplecsr, VectorT>& problem)
{

  return m_status.m_succeeded;
}

template <typename VectorT>
bool
TrilinosInternalEigenSolver::solve(
    Alien::GeneralizedEigenProblemT<Alien::BackEnd::tag::simplecsr, VectorT>& problem)
{

  return m_status.m_succeeded;
}

#endif

} // namespace Alien

#endif /* TRILINOSEIGENSOLVER_H_ */
