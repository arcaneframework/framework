// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* HashSuite.h                                                 (C) 2000-2025 */
/*                                                                           */
/* Fonction de hachage d'une suite de valeur.                                */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_HASHSUITE_H
#define ARCANE_UTILS_HASHSUITE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/HashFunction.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Classe permettant de calculer un hash de manière itératif.
 * \warning L'ordre dans lequel les valeurs sont données via la méthode add() est important.
 */

class IntegerHashSuite
{
 public:

  /*!
   * \brief Méthode permettant d'ajouter une valeur dans le calcul du hash.
   * \warning L'ordre dans lequel les valeurs sont données via la méthode
   * add() est important.
   * \param value La valeur à ajouter.
   */
  template <class T>
  void add(T value)
  {
    const UInt64 next_hash = static_cast<UInt64>(IntegerHashFunctionT<T>::hashfunc(value));
    m_hash ^= next_hash + 0x9e3779b9 + (m_hash << 6) + (m_hash >> 2);
  }

  /*!
   * \brief Méthode permettant de récupérer le hash calculé à partir de
   * toutes les valeurs passées à la méthode add().
   * \return Le hash.
   */
  Int64 hash() const
  {
    return static_cast<Int64>(m_hash);
  }

 private:

  UInt64 m_hash{0};
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
