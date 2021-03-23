// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
#ifndef HYPRESOLVERIMPL_H
#define HYPRESOLVERIMPL_H

#include "Numerics/LinearSolver/ILinearSolver.h"

#include "Numerics/LinearSolver/HypreSolverImpl/HypreOptionTypes.h"

#include "HypreSolver_axl.h"

/**
 * Interface du service de résolution de système linéaire
 */
 
using namespace Arcane;

class HypreLinearSystem;
class HypreLinearSystemBuilder;

class HypreLinearSolver :
  public ArcaneHypreSolverObject
{
public:
  /** Constructeur de la classe */
  HypreLinearSolver(const ServiceBuildInfo & sbi);
  
  /** Destructeur de la classe */
  virtual ~HypreLinearSolver();
  
public:

  //! Définition du constructeur du système linéaire
  void setLinearSystemBuilder(ILinearSystemBuilder * builder); 

  //! Initialisation
  void init();

  void start();

  //! Résolution du système linéaire
  bool solve();
  
  //! construit le system lineaire
  bool buildLinearSystem();
  
  //! recupere la solution
  bool getSolution();
  
  void end();

  //! Etat du solveur
  const Status & getStatus() const { return m_status; }
  
  ILinearSystem * getLinearSystem();

private:

  void updateLinearSystem();
  void freeLinearSystem();

private:
  //! Structure interne du solveur
  /* Pour l'instant le système est embarqué dans le solveur */
  HypreLinearSystem * m_system;

  //! flag pour verifier si le système a été construit
  bool m_system_is_built ;

  //! flag pour empecher de resoudre le meme système deux fois
  bool m_system_is_locked ;

  //! Constructeur du système linéaire
  ILinearSystemBuilder * m_builder;

  Status m_status;

private:
  void checkError(const String & msg, int ierr, int skipError = 0) const;
};

#endif /* HYPRESOLVERIMPL_H */
