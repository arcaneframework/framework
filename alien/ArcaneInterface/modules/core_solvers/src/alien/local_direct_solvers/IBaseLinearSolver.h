// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#pragma once

#include <alien/expression/solver/ILinearSolver.h>


namespace Alien {

class IBaseLinearSolver
{
 public:
    //! Type de retour de l'�tat du solveur
    /*! \todo Doit devenir un type complexe int�rrogeable un peu �
     *  l'image des CaseOptions
     */

  typedef Alien::ILinearSolver::Status Status ;

 public:
  /** Constructeur de la classe */
  IBaseLinearSolver()
    {
      ;
    }

  /** Destructeur de la classe */
  virtual ~IBaseLinearSolver() { }

public:

  //! Initialisation
  virtual void init() = 0;

  //! Finalizing (after solve)
  virtual void end() = 0 ;

  virtual const Status & getStatus() const = 0;

} ;


class ILinearSystemBuilder;
class ILinearSystem;

struct BuildType {
  enum eBuild
    {
      eNone        = 0,    // rien a construire
      eBuildMatrix = 1<<0, // la matrice doit etre reconstruite
      eBuildRhs    = 1<<1  // Le second membre doit etre reconstruit
    };
};

/*---------------------------------------------------------------------------*/

namespace LocalDirectSolverNamespace {

class ILinearSolver : public IBaseLinearSolver
{
 public:
  /** Destructeur de la classe */
  virtual ~ILinearSolver() { }

  //! Setting linear system builder
  virtual void setLinearSystemBuilder(ILinearSystemBuilder * builder) = 0;

  //!option to imposed extra constraint to solution
  virtual void setNullSpaceConstantOption(bool flag) = 0 ;

  //! Retourne le nom du solveur
  virtual String getName() const = 0 ;

  //! Getting linear system
  virtual ILinearSystem * getLinearSystem() = 0 ;

  //! Building linear system Tempory patch to enable solving several systems
  virtual bool buildLinearSystem(ILinearSystem*  system,Integer nextBuildStage=BuildType::eBuildMatrix|BuildType::eBuildRhs) = 0 ; ;

  //! Building linear system
  virtual bool buildLinearSystem(Integer nextBuildStage=BuildType::eBuildMatrix|BuildType::eBuildRhs) = 0 ; ;

  //! @name Resolution step
  //@{

  //! Initializing (before solve)
  virtual void start() = 0 ;

  //! Solving
  virtual bool solve() = 0 ;

  //! Getting solution
  virtual bool getSolution() = 0 ;

  //@}

#define USE_MULTI_SOLVER_INSTANCE
#ifdef USE_MULTI_SOLVER_INSTANCE
  class Solver
  {
  public :
    Solver(){}
    virtual ~Solver(){}

    virtual ILinearSystem * getLinearSystem() = 0 ;

    virtual bool solve(ILinearSystem* system,Status& status) const = 0 ;

    virtual bool getSolution(ILinearSystemBuilder* builder,ILinearSystem* system) const = 0 ;

  };

  virtual Solver* create() const {
    return nullptr ;
  }
#endif /* USE_MULTI_SOLVER_INSTANCE */
};

/*---------------------------------------------------------------------------*/

};

}

