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
 * \brief Fonctor pour une fonction de hachage.
 */
template <class Type>
class IntegerHashSuiteT
{
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Classe permettant de calculer un hash de manière itératif.
 * \warning L'ordre dans lequel les valeurs sont données via la méthode add() est important.
 */
template <>
class IntegerHashSuiteT<Int32>
{
 public:

  /*!
   * \brief Méthode permettant d'ajouter une valeur dans le calcul du hash.
   * \warning L'ordre dans lequel les valeurs sont données via la méthode
   * add() est important.
   * \param value La valeur à ajouter.
   */
  void add(Int32 value)
  {
    Int32 next_hash = IntegerHashFunctionT<Int32>::hashfunc(value);
    if (m_hash == -1) m_hash = next_hash;
    else m_hash ^= next_hash + 0x9e3779b9 + (m_hash << 6) + (m_hash >> 2);
  }

  /*!
   * \brief Méthode permettant de récupérer le hash calculé à partir de
   * toutes les valeurs passées à la méthode add().
   * \return Le hash.
   */
  Int32 hash()
  {
    return m_hash;
  }

private:
  Int32 m_hash{-1};
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Classe permettant de calculer un hash de manière itératif.
 * \warning L'ordre dans lequel les valeurs sont données via la méthode add() est important.
 */
template <>
class IntegerHashSuiteT<Int64>
{
public:
  /*!
   * \brief Méthode permettant d'ajouter une valeur dans le calcul du hash.
   * \warning L'ordre dans lequel les valeurs sont données via la méthode
   * add() est important.
   * \param value La valeur à ajouter.
   */
  void add(Int64 value)
  {
    Int64 next_hash = IntegerHashFunctionT<Int64>::hashfunc(value);
    if (m_hash == -1) m_hash = next_hash;
    else m_hash ^= next_hash + 0x9e3779b9 + (m_hash << 6) + (m_hash >> 2);
  }

  /*!
   * \brief Méthode permettant de récupérer le hash calculé à partir de
   * toutes les valeurs passées à la méthode add().
   * \return Le hash.
   */
  Int64 hash()
  {
    return m_hash;
  }

private:
  Int64 m_hash{-1};
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
