// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* BitonicSort.h                                               (C) 2000-2025 */
/*                                                                           */
/* Algorithme de tri bi-tonique parallèle.                                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_PARALLEL_BITONICSORT_H
#define ARCANE_CORE_PARALLEL_BITONICSORT_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/UniqueArray.h"

#include "arcane/core/IParallelSort.h"
#include "arcane/core/IParallelMng.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Parallel
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Fournit les opérations nécessaires pour le tri via la
 * classe \a BitonicSort.
 */
template <typename KeyType>
class BitonicSortDefaultTraits
{
 public:

  static bool compareLess(const KeyType& k1, const KeyType& k2)
  {
    return k1 < k2;
  }
  static Request send(IParallelMng* pm, Int32 rank, ConstArrayView<KeyType> values)
  {
    return pm->send(values, rank, false);
  }
  static Request recv(IParallelMng* pm, Int32 rank, ArrayView<KeyType> values)
  {
    return pm->recv(values, rank, false);
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
    return k != maxValue();
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Algorithme de tri bi-tonique parallèle.
 *
 * Le type de la clé peut être quelconque, mais il doit posséder un opérateur
 * de comparaison. Les caractéristiques nécessaires sont données par la
 * classe KeyTypeTraits. L'implémentation fournit les
 * opérations pour les types Int32, Int64 et Real via la classe
 * \a BitonicSortDefaultTraits. Pour les autres types, il est nécessaire de
 * spécialiser cette classe.
 *
 * La méthode sort() procède au tri. Après appel à cette méthode, il est
 * possible de récupérer la liste des clés via \a keys() et les rangs
 * et indices dans la liste d'origine de chaque élément de la clé via
 * les méthodes keyRangs() et keyIndexes(). Si ces informations ne sont
 * pas utiles, il est possible d'appeler setNeedIndexAndRank() pour
 * les désactiver ce qui permet d'accélérer quelque peu le tri.
 *
 * Le tri se fait de telle sorte que les éléments sont triés dans l'ordre croissant
 * en commençant par le processeur de rang 0, puis de rang 1 et ainsi de
 * suite jusqu'à la fin. Par exemple, pour une liste de 5000 éléments répartis
 * sur 4 rangs, le processeur de rang 0 possédera à la fin du tri les 1250 éléments
 * les plus petits, le processeur de rang 1 les 1250 éléments suivants et ainsi
 * de suite.
 *
 * Pour accélérer l'algorithme, il est préférable que tous les processeurs
 * aient environ le même nombre d'éléments dans leur liste au départ. A la fin
 * du tri, il est possible que tous les processeurs n'aient pas le même nombre
 * d'éléments dans la liste et notamment les processeurs de rang les plus élevés
 * peuvent ne pas avoir d'éléments.
 */
template <typename KeyType, typename KeyTypeTraits = BitonicSortDefaultTraits<KeyType>>
class BitonicSort
: public TraceAccessor
, public IParallelSort<KeyType>
{
 public:

  explicit BitonicSort(IParallelMng* parallel_mng);
  explicit BitonicSort(IParallelMng* parallel_mng, const KeyTypeTraits& traits);

 public:

  /*!
   * \brief Trie en parallèle les éléments de \a keys sur tous les rangs.
   *
   * Cette opération est collective.
   */
  inline void sort(ConstArrayView<KeyType> keys) override;

  //! Après un tri, retourne la liste des éléments de ce rang.
  ConstArrayView<KeyType> keys() const override { return m_keys; }

  //! Après un tri, retourne le tableau des rangs d'origine des éléments de keys().
  Int32ConstArrayView keyRanks() const override { return m_key_ranks; }

  //! Après un tri, retourne le tableau des indices dans la liste d'origine des éléments de keys().
  Int32ConstArrayView keyIndexes() const override { return m_key_indexes; }

 public:

  void setNeedIndexAndRank(bool want_index_and_rank)
  {
    m_want_index_and_rank = want_index_and_rank;
  }

 private:

  void _mergeLevels(Int32 begin, Int32 size);
  void _mergeProcessors(Int32 proc1, Int32 proc2);
  void _separator(Int32 begin, Int32 size);
  void _localHeapSort();

 private:

  //! Variable contenant la cle du tri
  UniqueArray<KeyType> m_keys;
  //! Tableau contenant le rang du processeur où se trouve la clé
  UniqueArray<Int32> m_key_ranks;
  //! Tableau contenant l'indice de la clé dans le processeur
  UniqueArray<Int32> m_key_indexes;
  //! Gestionnaire du parallèlisme
  IParallelMng* m_parallel_mng = nullptr;
  //! Nombre d'éléments locaux
  Int64 m_init_size = 0;
  //! Nombre d'éléments locaux pour le tri bi-tonique
  Int64 m_size = 0;

  //! Indique si on souhaite les infos sur les rangs et index
  bool m_want_index_and_rank = true;

  //! Statistiques sur le nombre de niveaux de messages
  Integer m_nb_merge = 0;

  KeyTypeTraits m_traits;

 private:

  void _init(ConstArrayView<KeyType> keys);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Parallel

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
