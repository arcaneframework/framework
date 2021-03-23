// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
#ifndef HYPRELINEARSYSTEM_H
#define HYPRELINEARSYSTEM_H

/**
 * Interface du service de résolution de système linéaire
 */

#include <arcane/ISubDomain.h>

using namespace Arcane;

class ILinearSystem ;
class HypreInternal ;
class HypreLinearSolver ;

class HypreLinearSystem : public ILinearSystem
{
public:
  //!Constructeur 
  HypreLinearSystem(HypreLinearSolver* solver);

  /** Destructeur */
  virtual ~HypreLinearSystem() ;

  //! initialise le system linéaire
  void init() {}
  
  //! permet d'ajouter des opérateurs à l'interface
  bool accept(ILinearSystemVisitor* visitor)
  { 
    return visitor->visit(this) ;
  }
  
  //! démare une étape de résolution de système linéaire
  void start() {}
  
  //! finalise une étape de résolution et libère les objets intermédiaires
  void end() {}

  //! retourne le nom du type de système
  const char * name() const { return "Hypre"; }
  
  bool initMatrix(const int ilower, const int iupper,
            const int jlower, const int jupper,
            const ConstArrayView<Integer> & lineSizes) ;

  bool addMatrixValues(const int nrow, const int * rows,
                       const int * ncols, const int * cols,
                       const Real * values) ;

  bool setMatrixValues(const int nrow, const int * rows,
                       const int * ncols, const int * cols,
                       const Real * values) ;

  bool addRHSValues(const int nrow, const int * rows,
                   const Real * values) ;

  bool setRHSValues(const int nrow, const int * rows,
                   const Real * values) ;

  bool setInitValues(const int nrow, const int * rows,
                    const Real * values) ;
  
  bool assemble() ;

  bool getSolutionValues(const int nrow, const int * rows,
                         Real * values) ;
  
  ISubDomain * getSubDomain() const;

protected:
  friend class HypreLinearSolver;

  //! Internal struct container
  /*! Hide all implementation constraint in this header */  
  HypreInternal * m_internal;

private :
  HypreLinearSolver* m_solver ;
};

#endif /* HYPRESYSTEMIMPL_H */
