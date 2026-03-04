// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#pragma once



#include "alien/local_direct_solvers/IBaseLinearSolver.h"
#include "alien/local_direct_solvers/ILinearSystem.h"
#include "alien/local_direct_solvers/ILinearSystemBuilder.h"
#include "alien/local_direct_solvers/ILinearSystemVisitor.h"
#include "alien/local_direct_solvers/LocalLinearSystemT.h"

namespace Alien {

template<int N>
class EigenLUSolver : public LocalDirectSolverNamespace::ILinearSolver
{
  public:

  typedef Eigen::Matrix<Real,N,N,Eigen::RowMajor>                    MatrixType ;
  typedef Eigen::Matrix<Real,N,1>                                    VectorType ;

  typedef LinearSystemT<MatrixType,VectorType>                       LinearSystemType ;

  //! Constructeur de la classe
  EigenLUSolver()
  : ILinearSolver()
  {}

  //! Destructeur de la classe
  virtual ~EigenLUSolver() {}

public:
  String getName() const {
    return "EigenLUSolver" ;
  }

  //! Initialisation
  void init() {}

  //! Associe un build � ce solveur
  void setLinearSystemBuilder(ILinearSystemBuilder * builder)
  { m_builder = builder ; }

  //! Retourne le syst�me lin�aire associ� � ce solveur
  ILinearSystem* getLinearSystem() { return m_system ; }

  //! Construit le system lineaire
  bool buildLinearSystem(Integer nextBuildStage=BuildType::eBuildMatrix|BuildType::eBuildRhs)
  {
    bool flag = (m_builder?m_system->accept(m_builder):true) ;
    m_system_is_built = true ;
    m_system_is_locked = false ;
    return flag ;
  }

  bool buildLinearSystem(ILinearSystem*  system,Integer nextBuildStage=BuildType::eBuildMatrix|BuildType::eBuildRhs)
  {
    bool flag = system->accept(m_builder) ;
    m_system_is_built = true ;
    m_system_is_locked = false ;
    return flag ;
  }


  //! @name Etapes d'une r�solution
  //@{

  //! D�but de boucle locale (avant solve)
  void start() {
    delete m_system ;
    m_system = new LinearSystemType() ;
  }
  //! Fin de boucle locale (apr�s solve)
  void end() {
    delete m_system ;
    m_system = nullptr ;
    m_system_is_built = false ;
    m_system_is_locked = false ;
  }

  //! R�solution du system lineaire associ�
  bool solve() {

    if(!m_system_is_built)
    {
      cerr<<"Linear system is not built, buildLinearSystem shoud be called first"<<endl ;
      return  false ;
    }
    if(m_system_is_locked)
    {
      cerr<<"linear system has already be solved once and has not been modified since" ;
      return false ;
    }
    *(m_system->getX()) = m_system->getMatrix()->lu().solve(*m_system->getRhs()) ;
    m_status.succeeded = true ;
    m_status.error = 0 ;
    m_status.iteration_count = 0 ;
    m_status.residual = 0. ;
    m_system_is_locked = true ;
    return true ;
  }

  //! Etat final apr�s r�solution
  const ILinearSolver::Status & getStatus() const { return m_status ; }

  //! Applique la proc�dure d'extraction de la solution
  bool getSolution() {
    if(m_builder)
      return m_builder->commitSolution(m_system) ;
    else
      return false ;
  }

  void setNullSpaceConstantOption(bool flag) {
    throw Arccore::FatalErrorException(A_FUNCINFO,"not implemented");
  }
private :
  LinearSystemType* m_system = nullptr;

  //! flag pour verifier si le syst�me a �t� construit
  bool m_system_is_built = false;

  //! flag pour empecher de resoudre le meme syst�me deux fois
  bool m_system_is_locked = false;
  ILinearSystemBuilder* m_builder = nullptr;
  ILinearSolver::Status m_status ;
};
}
