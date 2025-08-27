// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SimpleTableInternal.h                                       (C) 2000-2022 */
/*                                                                           */
/* Fichier contenant la structure SimpleTableInternal décrivant un tableau   */
/* de valeurs simple.                                                        */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifndef ARCANE_SIMPLETABLEINTERNAL_H
#define ARCANE_SIMPLETABLEINTERNAL_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/IParallelMng.h"
#include "arcane/core/ItemTypes.h"
#include "arcane/utils/Array.h"
#include "arcane/utils/Array2.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/**
 * @brief Structure représentant un tableau simple.
 * 
 * Un tableau simple ressemble à ça :
 * 
 * NomTableau | C1 | C2 | C3
 *    L1      |Val1|Val2|Val3
 *    L2      |Val4|Val5|Val6
 * 
 * Un nom de tableau, une liste de noms de lignes,
 * une liste de noms de colonnes et une liste 2D
 * de valeur (Real pour l'instant).
 * 
 */
struct ARCANE_CORE_EXPORT SimpleTableInternal
{
  SimpleTableInternal(IParallelMng* parallel_mng)
  : m_parallel_mng(parallel_mng)
  , m_values()
  , m_row_names()
  , m_column_names()
  , m_table_name("")
  , m_row_sizes()
  , m_column_sizes()
  , m_last_row(-1)
  , m_last_column(-1)
  {
  }
  ~SimpleTableInternal() = default;

  void clear()
  {
    m_values.clear();
    m_row_names.clear();
    m_column_names.clear();
    m_table_name = "";
    m_row_sizes.clear();
    m_column_sizes.clear();
    m_last_row = -1;
    m_last_column = -1;
  }
  IParallelMng* m_parallel_mng;

  UniqueArray2<Real> m_values;

  UniqueArray<String> m_row_names;
  UniqueArray<String> m_column_names;

  String m_table_name;

  // Tailles des lignes/colonnes
  // (et pas le nombre d'éléments, on compte les "trous" entre les éléments ici,
  // mais sans le trou de fin).
  // Ex. : {{"1", "2", "0", "3", "0", "0"},
  //        {"4", "5", "6", "0", "7", "8"},
  //        {"0", "0", "0", "0", "0", "0"}}

  //       m_row_sizes[0] = 4
  //       m_row_sizes[1] = 6
  //       m_row_sizes[2] = 0
  //       m_row_sizes.size() = 3

  //       m_column_sizes[3] = 1
  //       m_column_sizes[0; 1; 2; 4; 5] = 2
  //       m_column_sizes.size() = 6
  UniqueArray<Integer> m_row_sizes;
  UniqueArray<Integer> m_column_sizes;

  // Position du dernier élement ajouté.
  Integer m_last_row;
  Integer m_last_column;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
