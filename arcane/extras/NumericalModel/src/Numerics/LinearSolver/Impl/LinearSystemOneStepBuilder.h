// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
#ifndef LINEARSYSTEMONESTEPBUILDER_H
#define LINEARSYSTEMONESTEPBUILDER_H

#include "Numerics/LinearSolver/ILinearSystemBuilder.h"
#include "Numerics/LinearSolver/IIndexManager.h"

using namespace Arcane;

#include <map>

class HypreLinearSystem;
class PETScLinearSystem;
class IFPSolverLinearSystem;
class MTLLinearSystem;

class LinearSystemOneStepBuilder
  : public ILinearSystemBuilder
{
public:
  typedef IIndexManager::Entry Entry ;
  typedef IIndexManager::Equation Equation ;
  typedef IIndexManager::EntryIndex EntryIndex ;
  typedef IIndexManager::EquationIndex EquationIndex ;
  LinearSystemOneStepBuilder();

  virtual ~LinearSystemOneStepBuilder();

  void addData(const Integer iIndex,
               const Integer jIndex,
               const Real value);
  
  void addData(const Integer iIndex,
               const Real factor,
               const ConstArrayView<Integer> & jIndexes,
               const ConstArrayView<Real> & jValues);

  void addRHSData(const Integer iIndex,
                  const Real value);

  void setRHSData(const Integer iIndex,
                  const Real value);

  void init();

  void start();

  void end() ;
  
  void freeData();

  void initRhs();

  IIndexManager * getIndexManager() ;

private:
  //@{ @name Interface générale au solveur
  template<typename SystemT> void initUnknownT(SystemT * system) ;
  template<typename SystemT> bool commitSolutionT(SystemT * system) ;
  //@}

public:
  //@{ @name Interface à Hypre
  bool build(HypreLinearSystem * system);
  void initUnknown(HypreLinearSystem * system) ;
  bool commitSolution(HypreLinearSystem * system) ;
  bool connect(HypreLinearSystem * system) ;
  bool visit(HypreLinearSystem * system) ;
  //@}

  //! Dump matrix and vector data to Matlab format for debugging
  void dumpToMatlab(const std::string& file_name);

protected:
  //! @internal Structure de représentation d'un vecteur
  typedef std::map<Integer, Real> VectorData;
  
  //! @internal Structure de représentation du graphe de la matrice
  typedef std::map<Integer, VectorData > MatrixData;

  //! @internal Représentation de la matrice
  MatrixData m_matrix;

  //! @internal Représentation du vecteur second membre
  VectorData m_rhs_vector;

  //! @internal Représentation du vecteur solution
  VectorData m_x_vector;

  //! Définition des états du builder
  enum State { eNone, eInit, eStart, eBuild };

  //! Etat courant du builder
  State m_state;

  //! Définition des type de solveurs supportés
  enum SolverType { eUndefined, eHypre, ePETSc, eIFPSolver, eMTLSolver };

  //! Type du solver associé
  SolverType m_solver_type;

  //! Accès aux traces
  ITraceMng * m_trace;

  //! Accès au parallélisme Arcane
  IParallelMng * m_parallel_mng;

  //! Indexation des entrées
  IIndexManager * m_index_manager;
};

#endif /* LINEARSYSTEMONESTEPBUILDER_H */
