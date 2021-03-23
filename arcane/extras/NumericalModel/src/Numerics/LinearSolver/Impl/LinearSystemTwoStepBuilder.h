// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
#ifndef LINEARSYSTEMTWOSTEPBUILDER_H
#define LINEARSYSTEMTWOSTEPBUILDER_H

/*! 
  \class LinearSystemTwoStepBuilder
  \author Pascal Havé <pascal.have@ifp.fr>
  \author Daniele Di Pietro <daniele-antonio.di-pietro@ifp.fr>
  \brief Constructeur de systèmes linéaires en 2 temps (structure + data)
  \todo En faire une vraie méthode en 2 temps (actuellement la
  définition n'est pas utilisé en contrôle des données saisies) +
  Optimiser pour l'implémentation sous-jacente.
*/

#include "Numerics/LinearSolver/ILinearSystemBuilder.h"
#include "Numerics/LinearSolver/IIndexManager.h"

using namespace Arcane;

#include <vector>
#include <map>
#include <set>

class HypreLinearSystem;
class PETScLinearSystem;
class MTLLinearSystem;

class LinearSystemTwoStepBuilder : 
  public ILinearSystemBuilder
{
public:
  typedef IIndexManager::Entry Entry ;
  typedef IIndexManager::Equation Equation ;
  typedef IIndexManager::EntryIndex EntryIndex ;
  typedef IIndexManager::EquationIndex EquationIndex ;
  LinearSystemTwoStepBuilder();

  virtual ~LinearSystemTwoStepBuilder();

public:
  //! défini une entrée ixj non nulle du sytème linéaire
  void defineData(const IIndexManager::EquationIndex iIndex,
                  const IIndexManager::EntryIndex jIndex);

  //! défini un ensemble d'entrées ix(j1,j2...jn) non nulles du sytème linéaire
  void defineData(const IIndexManager::EquationIndex iIndex,
                  const ConstArrayView<IIndexManager::EntryIndex> & jIndexes);

  void addData(const Integer iIndex,
               const Integer jIndex,
               const Real value);
  
  void setData(const Integer iIndex,
               const Integer jIndex,
               const Real value);

  void addData(const Integer iIndex,
               const Real factor,
               const ConstArrayView<Integer> & jIndexes,
               const ConstArrayView<Real> & jValues);

  void setData(const Integer iIndex,
               const Real factor,
               const ConstArrayView<Integer> & jIndexes,
               const ConstArrayView<Real> & jValues);

  void addRHSData(const Integer iIndex,
                  const Real value);

  void setRHSData(const Integer iIndex,
                  const Real value);
  

  virtual void initRhs()
  {
    throw FatalErrorException(A_FUNCINFO,"not implemented");
  }
  
  inline void initData(const Real matrix_value, const Real rhs_value)
  {
    _fill(m_matrix,matrix_value) ;
    _fill(m_rhs_vector,rhs_value);
  }

  void init();

  void start();

  void end();
  
  void freeData();
  
  IIndexManager * getIndexManager();

  void setTraceMng(ITraceMng * traceMng);

  void setParallelMng(IParallelMng * parallelMng);

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

  //@{ @name Interface à PETSc
  bool build(PETScLinearSystem * system);
  void initUnknown(PETScLinearSystem * system) ;
  bool commitSolution(PETScLinearSystem * system) ;
  bool connect(PETScLinearSystem * system) ;
  bool visit(PETScLinearSystem * system) ;
  //@}

  //@{ @name Interface à MTL
  bool build(MTLLinearSystem * system);
  void initUnknown(MTLLinearSystem * system) ;
  bool commitSolution(MTLLinearSystem * system) ;
  bool connect(MTLLinearSystem * system) ;
  bool visit(MTLLinearSystem * system) ;
  //@}
 public:
  //! Dump matrix and vector data to Matlab format for debugging
  void dumpToMatlab(const std::string& file_name);

protected:
  typedef std::set<IIndexManager::EntryIndex> VectorDefinition;

  //! @internal Structure de représentation du graphe de la matrice
  typedef std::map<IIndexManager::EquationIndex, VectorDefinition> MatrixDefinition;

  //! @internal Structure de représentation d'un vecteur
  typedef std::map<Integer, Real> MapData;
  
  //! @internal Structure de représentation du graphe de la matrice
  typedef Array<MapData> MatrixData;

  //! @internal Structure de représentation d'un vecteur
  typedef RealArray VectorData;

  inline void _fill(VectorData& vector,const Real value) {
    for(VectorData::iterator iter = vector.begin();iter!=vector.end();++iter)
      *iter = value ;
  }
  inline void _fill(MapData& vector,const Real value) {
    for(MapData::iterator iter = vector.begin();iter!=vector.end();++iter)
      (*iter).second = value ;
  }
  inline void _fill(MatrixData& matrix,const Real value)
  {
    for(MatrixData::iterator irow = matrix.begin();irow!=matrix.end();++irow)
      _fill(*irow,value) ;
  }

  //! @internal Représentation de la matrice (CSR)
  MatrixDefinition m_def_matrix;

  //! @internal Représentation de la matrice (CSR)
  MatrixData m_matrix;

  //! @internal Représentation du vecteur second membre
  VectorData m_rhs_vector;

  //! @internal Représentation du vecteur solution
  VectorData m_x_vector;

  //! Stats locales de la matrice
  Integer m_global_size, m_local_offset, m_local_size;

  //! Définition des états du builder
  enum State { eNone, eInit, eStart, eBuild };

  //! Etat courant du builder
  State m_state;

  //! Définition des type de solveurs supportés
  enum SolverType { eUndefined, eHypre, ePETSc, eMTL};

  //! Type du solver associé
  SolverType m_solver_type;

  //! Accès aux traces
  ITraceMng * m_trace;

  //! Accès au parallélisme Arcane
  IParallelMng * m_parallel_mng;

  //! Indexation des entrées
  IIndexManager * m_index_manager;
};

#endif /* LINEARSYSTEMTWOSTEPBUILDER_H */
