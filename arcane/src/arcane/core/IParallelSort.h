// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IParallelSort.h                                             (C) 2000-2024 */
/*                                                                           */
/* Interface d'un algorithme de tri parallèle.                               */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IPARALLELSORT_H
#define ARCANE_CORE_IPARALLELSORT_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Parallel
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface d'un algorithme de tri parallèle
 *
 * Le type de la clé doit être comparable,et posséder l'opérateur operator<.
 *
 * Pour l'instant, cette interface est implémentée pour les types suivants:
 * Int32, Int64 et Real.
 *
 * La méthode sort() procède au tri. Après le tri, il est possible
 * de récupérer pour chaque clé le rang et l'indice de son origine,
 * via keyRanks() et keyIndexes(). Les clés triées sont accessible
 * via keys().
 */
template <typename KeyType>
class IParallelSort
{
 public:

  virtual ~IParallelSort() = default;

 public:

  /*!
   * \brief Tri en parallèle les clés \a keys.
   *
   * Cette méthode est collective.
   * Le tri est global, chaque rang donnant sa liste de clés \a keys.
   */
  virtual void sort(ConstArrayView<KeyType> keys) = 0;

  //! Tableau des clés
  virtual ConstArrayView<KeyType> keys() const = 0;

  //! Tableau des rangs du processeur d'origine contenant la clé
  virtual Int32ConstArrayView keyRanks() const = 0;

  //! Tableau des indices de la clé dans le processeur d'origine.
  virtual Int32ConstArrayView keyIndexes() const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Parallel

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
