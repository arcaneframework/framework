// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#pragma once

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/lu.hpp>

#include "alien/local_direct_solvers/ILinearSystem.h"
#include "alien/local_direct_solvers/ILinearSystemBuilder.h"
#include "alien/local_direct_solvers/ILinearSystemVisitor.h"
#include "alien/local_direct_solvers/IBaseLinearSolver.h"
#include "alien/local_direct_solvers/algorithms/LUSolver.h"

namespace Alien {

class LocalLinearSystem
: public ILinearSystem
{
public :
  typedef boost::numeric::ublas::vector<Arccore::Real> RealVector;
  typedef boost::numeric::ublas::matrix<Arccore::Real> RealMatrix;
  typedef RealVector vector_type ;
  typedef RealMatrix matrix_type ;

  LocalLinearSystem() {}

  LocalLinearSystem(RealMatrix* matrix,
                    RealVector* x,
                    RealVector* rhs)
  : m_matrix(matrix)
  , m_x(x)
  , m_rhs(rhs){}

  //! initialise le system lin�aire
  void init() {} ;

  //! permet d'ajouter des op�rateurs au l'interface
  bool accept(ILinearSystemVisitor* visitor)
  {
    return visitor->visit(this) ;
  }

  //! d�mare une �tape de r�solution de syst�me lin�aire
  void start() {}

  //! finalise une �tape de r�solution et lib�re les objets interm�diaires
  void end() {}

  //! retourne le nom du type de syst�me
  virtual const char * name() const { return "LocalLinearSystem" ; }

  RealMatrix* getMatrix() { return m_matrix ; }
  RealVector* getX() { return m_x ; }
  RealVector* getRhs() { return m_rhs ; }

  void setMatrix(RealMatrix* matrix) { m_matrix = matrix ; }
  void setX(RealVector* x) { m_x = x ; }
  void setRhs(RealVector* rhs) { m_rhs = rhs ; }
private :
  RealMatrix* m_matrix = nullptr;
  RealVector* m_x = nullptr;
  RealVector* m_rhs = nullptr;
};

class LocalLinearSystemBuilder
: public ILinearSystemBuilder
{
public :
  LocalLinearSystemBuilder(ILinearSystem* system)
  : ILinearSystemBuilder()
  {
    m_system = dynamic_cast<LocalLinearSystem*>(system) ;
    if(!system)
      throw Arccore::FatalErrorException(A_FUNCINFO,
                                        "Only LocalLinearSystem can be used");
  }

  LocalLinearSystemBuilder(LocalLinearSystem* system)
  : m_system(system) {}

  virtual ~LocalLinearSystemBuilder() {}

  void init() {}
  void start() {}
  void freeData() {}
  void end() {}

  bool connect(LocalLinearSystem* system)
  {
    (*system) = (*m_system) ;
    return true;
  }

  bool commitSolution(LocalLinearSystem* system)
  { return true; }

  bool visit(LocalLinearSystem* system)
  { return connect(system) ; }

private :
  LocalLinearSystem* m_system = nullptr;

};

class LocalDirectSolver
: public Alien::ILinearSolver
{
  public:
  //! Constructeur de la classe
  LocalDirectSolver()
    : Alien::ILinearSolver()
  {}

  //! Destructeur de la classe
  virtual ~LocalDirectSolver() {}

public:
  Arccore::String getName() const {
    return "UblasLUSolver" ;
  }

  //! Initialisation
  void init() {}

  //! Associe un build � ce solveur
  void setLinearSystemBuilder(ILinearSystemBuilder * builder)
  { m_builder = builder ; }

  //! Retourne le syst�me lin�aire associ� � ce solveur
  ILinearSystem* getLinearSystem() { return m_system ; }

  //! Construit le system lineaire
  bool buildLinearSystem(Arccore::Integer nextBuildStage=BuildType::eBuildMatrix|BuildType::eBuildRhs)
  {
    bool flag = (m_builder?m_system->accept(m_builder):true) ;
    m_system_is_built = true ;
    m_system_is_locked = false ;
    return flag ;
  }

  bool buildLinearSystem(ILinearSystem*  system,Arccore::Integer nextBuildStage=BuildType::eBuildMatrix|BuildType::eBuildRhs)
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
    m_system = new LocalLinearSystem() ;
  }
  //! Fin de boucle locale (apr�s solve)
  void end() {
    delete m_system ;
    m_system = nullptr ;
    m_system_is_built = false ;
    m_system_is_locked = false ;
  }

  //! R�solution du system lineaire associ�
  bool solve() ;

  //! Etat final apr�s r�solution
  const Alien::ILinearSolver::Status & getStatus() const { return m_status ; }

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
  LocalLinearSystem* m_system = nullptr;

  //! flag pour verifier si le syst�me a �t� construit
  bool m_system_is_built = false;

  //! flag pour empecher de resoudre le meme syst�me deux fois
  bool m_system_is_locked = false;
  ILinearSystemBuilder* m_builder = nullptr;
  Alien::ILinearSolver::Status m_status ;
};
}

