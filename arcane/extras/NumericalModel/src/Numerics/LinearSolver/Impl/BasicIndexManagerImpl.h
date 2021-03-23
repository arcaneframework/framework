// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
#ifndef BASICINDEXMANAGERIMPL_H
#define BASICINDEXMANAGERIMPL_H

#include "Numerics/LinearSolver/IIndexManager.h"

#include <arcane/IMesh.h>
#include <set>
#include <map>
#include <vector>

using namespace Arcane ;


/*! \todo Il est possible d'optimiser les accès vectorielles en raprochant les
 *  structures internes des interfaces critiques
 *  (en particulier getEntryIndex vectoriel)
 */
class BasicIndexManagerImpl :
  public IIndexManager
{
public:
  class MyEntryEnumeratorImpl;
  class MyEntryImpl;
  class MyEquationImpl;

public:
  //! Constructeur de la classe
  BasicIndexManagerImpl(IParallelMng * parallelMng) :
    m_parallel_mng(parallelMng),
    m_state(Undef),
    m_trace(NULL),
    m_creation_index(0)
  {
    ;
  }

  //! Destructeur de la classe
  virtual ~BasicIndexManagerImpl();

  bool isReady() {
    return (m_state == Prepared ) ;
  }
  //! Initialisation
  void init();

  //! Définit le gestionnaire de trace
  void setTraceMng(ITraceMng * traceMng);

  //! Construit une nouvelle entrée abstraite
  Entry buildAbstractEntry(const String name, const IItemFamily * itemFamily);

  //! Construit une entrée associée à une variable
  Entry buildVariableEntry(IVariable * ivar, const eVariableAccess mode);

  //! Retourne l'entrée associée à un nom
  /*! Valable aussi pour un accès via un nom de variable */
  Entry getEntry(const String name) const;

  //! Retourne l'entrée associée à une variable
  Entry getVariableEntry(IVariable * ivar) const;

  //! Construit une équation indépendante
  /*! Ceci implique une politique de placement de l'équation dans le
   *  système mal définie
   */
  Equation buildEquation(const String name, const IItemFamily * itemFamily);

  //! Construit une équation dominée par une entrée
  /*! L'indexation de cette équation sera associée à celle de l'entrée
   *  fournit
   */
  Equation buildEquation(const String name, const Entry & entry);

  //! Retourne l'équation via son nom
  Equation getEquation(const String name);

  //! Demande d'indexation d'une entrée (colonne)
  /*! Utilisable uniquement avant prepare */
  EntryIndex defineEntryIndex(const Entry & entry, const Item & item);

  //! Demande d'indexation d'une entrée (colonnes)
  /*! Utilisable uniquement avant prepare */
  void defineEntryIndex(const Entry & entry, const ItemGroup & itemGroup);

  //! Demande d'indexation d'une équation (ligne) (avant prepare)
  /*! Utilisable uniquement avant prepare */
  EquationIndex defineEquationIndex(const Equation & entry, const Item & item);

  //! Demande d'indexation d'une équation (lignes) (avant prepare)
  /*! Utilisable uniquement avant prepare */
  void defineEquationIndex(const Equation & entry, const ItemGroup & itemGroup);

  //! Préparation : fixe l'indexation (fin des définitions)
  void prepare();

  //! Consultation d'indexation d'une entrée (après prepare)
  Integer getEntryIndex(const Entry & entry, const Item & item) const;

  //! Consultation de réindexation (après prepare)
  Integer getEquationIndex(const Equation & equation, const Item & item) const;

  //! Consultation vectorielle d'indexation d'une entrée (après prepare)
  void getEntryIndex(const Entry & entry, const ItemVectorView & items, ArrayView<Integer> indexes) const;

  //! Consultation vectorielle de réindexation (après prepare)
  void getEquationIndex(const Equation & equation, const ItemVectorView & items, ArrayView<Integer> indexes) const;

  //! Translation d'un EntryIndex en indice de colonne
  Integer getEntryIndex(const EntryIndex & index) const;

  //! Translation d'un EquationIndex en indice de ligne
  Integer getEquationIndex(const EquationIndex & index) const;

  //! Fournit une table de translation indexé par les items
  IntegerArray getEntryIndexes(const Entry & entry) const;

  //! Fournit une table de translation indexé par les items
  Array2<Integer> getVecEntryIndexes(ConstArrayView<Entry> entry) const;


  //! Fournit une table de translation indexé par les items
  IntegerArray getEquationIndexes(const Equation & equation) const;

  //! Fournit une table de translation indexé par les items
  Array2<Integer> getVecEquationIndexes(ConstArrayView<Equation> equation) const;

  //! Décrit si l'ordre des getEntryIndexes et getEquationIndexes associés sont compatibles
  bool hasCompatibleIndexOrder(const Entry & entry, const Equation & equation) const;

  //! Statistiques d'indexation
  /*! Uniquement valide après \a prepare */
  void stats(Integer & totalSize,
		     Integer & minLocalIndex,
		     Integer & localSize) const;

  //! Construction d'un enumerateur sur les \a Entry
  EntryEnumerator enumerateEntry();

private:
  IParallelMng * m_parallel_mng;

  enum State { Undef, Initialized, Prepared } m_state;
  ITraceMng * m_trace;

  typedef std::pair<MyEntryImpl *, Item> InternalEntryIndex;
  typedef std::pair<MyEquationImpl *, Item> InternalEquationIndex;

  struct EntryIndexComparator
  {
    inline bool operator()(const InternalEntryIndex & a,
                           const InternalEntryIndex & b) const;
  };

  struct EquationIndexComparator
  {
    inline bool operator()(const InternalEquationIndex & a,
                           const InternalEquationIndex & b) const;
  };

  typedef std::map<InternalEntryIndex, Integer, EntryIndexComparator> EntryIndexMap;
  EntryIndexMap m_entry_index; //!< Table des index d'entrées (>=0:local, <0:global) en phase1

  typedef std::map<InternalEquationIndex, Integer, EquationIndexComparator> EquationIndexMap;
  EquationIndexMap m_equation_index; //!< Table des index d'équations (>=0 car tous locales)

  Array<Integer> m_entry_reindex; //!< Table EntryIndex->Integer
  Array<Integer> m_equation_reindex; //!< Table EquationIndex->Integer

  Integer m_local_entry_count;
  Integer m_global_entry_count;
  Integer m_global_entry_offset;

  Integer m_local_equation_count;
  Integer m_global_equation_count;
  Integer m_global_equation_offset;

  //! Table d'accès rapide à l'Entry associé à une variable
  typedef std::map<const IVariable *, MyEntryImpl*> VarEntryMap;
  VarEntryMap m_var_entry;

  //! Table des Entry connues localement
  typedef std::map<String,MyEntryImpl*> EntrySet;
  EntrySet m_entry_set;

  //! Table des Equations
  typedef std::map<String,MyEquationImpl*> EquationSet;
  EquationSet m_equation_set;

  //! Index de creation des entrées
  Integer m_creation_index;

protected:
  //! \internal Structure interne de communication dans prepare()
  struct EntrySendRequest;
  struct EntryRecvRequest;

private:
  void parallel_prepare();
  void sequential_prepare();
};

#endif /* BASICINDEXMANAGERIMPL_H */
