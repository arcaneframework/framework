﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* BitonicSort.h                                               (C) 2000-2016 */
/*                                                                           */
/* Algorithme de tri bitonique parallèle                                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_PARALLEL_BITONICSORT_H
#define ARCANE_PARALLEL_BITONICSORT_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/Limits.h"

#include "arcane/IParallelSort.h"
#include "arcane/IParallelMng.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE_PARALLEL

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Fournit les opérations nécessaires pour le tri via la
 * classe \a BitonicSort.
 */
template<typename KeyType>
class BitonicSortDefaultTraits
{
 public:
  static bool compareLess(const KeyType& k1,const KeyType& k2)
  {
    return k1<k2;
  }
  static Request send(IParallelMng* pm,Int32 rank,ConstArrayView<KeyType> values)
  {
    return pm->send(values,rank,false);
  }
  static Request recv(IParallelMng* pm,Int32 rank,ArrayView<KeyType> values)
  {
    return pm->recv(values,rank,false);
  }
  //! Valeur max possible pour la clé.
  static KeyType maxValue()
  {
    //return ARCANE_INTEGER_MAX-1;
    return std::numeric_limits<KeyType>::max();
  }
  // Indique si la clé est valide. Elle doit être invalide si k==maxValue()
  static bool isValid(const KeyType& k)
  {
    return k!=maxValue();
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Algorithme de tri bitonique parallèle.
 *
 * Le type de la clé peut être quelconque, mais il doit posséder un opérateur
 * de comparaison. Les caractéristiques nécessaires sont données par la
 * classe KeyTypeTraints. . L'implémentation fournit les
 * opérations pour les type Int32, Int64 et Real via la classe
 * \a BitonicSortDefaultTraits. Pour les autres types, il est nécessaire de
 * spécialiser cette classe.
 *
 * La méthode sort() procède au tri. Après appel à cette méthode, il est
 * possible de récupérer la liste des clés via \a keys() et les rangs
 * et indices dans la liste d'origine de chaque élément de la clé via
 * les méthodes keyRangs() et keyIndexes(). Si ces informations ne sont
 * pas utiles, il est possible d'appeler setNeedIndexAndRank() pour
 * les désactiver ce qui permet d'accéler quelque peu le tri.
 *
 * Le tri se fait de telle sorte que les éléments sont triés dans l'ordre croissant
 * en commençant par le processeur de rang 0, puis de rang 1 et ainsi de
 * suite jusqu'à la fin. Par exemple, pour une liste de 5000 éléments répartis
 * sur 4 rangs, le processeur de rang 0 possédera à la fin du tri les 1250 éléments
 * les plus petits, le processeur de rang 1 les 1250 éléments suivantes et ainsi
 * de suite.
 *
 * Pour accélérer l'algorithme, il est préférable que tous les processeurs
 * aient environ le même nombre d'éléments dans leur liste au départ. A la fin
 * du tri, il est possible que tous les processeurs n'aient pas le même nombre
 * d'éléments dans la liste et notamment les processeurs de rang les plus élevés
 * peuvent ne pas avoir d'éléments.
 */
template<typename KeyType,typename KeyTypeTraits = BitonicSortDefaultTraits<KeyType> >
class BitonicSort
: public TraceAccessor
, public IParallelSort<KeyType>
{
 public:

  BitonicSort(IParallelMng* parallel_mng);

  /*!
   * \brief Trie en parallèle les éléments de \a keys sur tous les rangs.
   *
   * Cette opération est collective.
   */
  virtual void sort(ConstArrayView<KeyType> keys);

  //! Après un tri, retourne la liste des éléments de ce rang.
  virtual ConstArrayView<KeyType> keys() const { return m_keys; }

  //! Après un tri, retourne le tableau des rangs d'origine des éléments de keys().
  virtual Int32ConstArrayView keyRanks() const { return m_key_ranks; }
  
  //! Après un tri, retourne le tableau des indices dans la liste d'origine des éléments de keys().
  virtual Int32ConstArrayView keyIndexes() const { return m_key_indexes; }

 public:

  void setNeedIndexAndRank(bool want_index_and_rank)
  {
    m_want_index_and_rank = want_index_and_rank;
  }

 private:

  void _mergeLevels(Int32 begin,Int32 size);
  void _mergeProcessors(Int32 proc1,Int32 proc2);
  void _separator(Int32 begin,Int32 size);
  void _localHeapSort();

 private:

  //! Variable contenant la cle du tri
  UniqueArray<KeyType> m_keys;
  //! Tableau contenant le rang du processeur où se trouve la clé
  UniqueArray<Int32> m_key_ranks;
  //! Tableau contenant l'indice de la clé dans le processeur
  UniqueArray<Int32> m_key_indexes;
  //! Gestionnaire du parallèlisme
  IParallelMng* m_parallel_mng;
  //! Nombre d'éléments locaux
  Integer m_init_size;
  //! Nombre d'éléments locaux pour le tri bitonique
  Integer m_size;
  
  //! Indique si on souhaite les infos sur les rangs et index
  bool m_want_index_and_rank;

  //! Statistiques sur le nombre de niveaux de messages
  Integer m_nb_merge;

 private:

  void _init(ConstArrayView<KeyType> keys);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE_PARALLEL

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
