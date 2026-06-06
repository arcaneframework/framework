// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SimpleTableInternal.h                                       (C) 2000-2025 */
/*                                                                           */
/* File containing the SimpleTableInternal structure describing a table      */
/* of simple values.                                                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_SIMPLETABLEINTERNAL_H
#define ARCANE_CORE_SIMPLETABLEINTERNAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Array.h"
#include "arcane/utils/Array2.h"

#include "arcane/core/IParallelMng.h"
#include "arcane/core/ItemTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/**
 * @brief Structure representing a simple table.
 * 
 * A simple table looks like this:
 * 
 * TableName | C1 | C2 | C3
 *    L1      |Val1|Val2|Val3
 *    L2      |Val4|Val5|Val6
 * 
 * A table name, a list of row names,
 * a list of column names, and a 2D list
 * of values (Real for now).
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

  // Row/column sizes
  // (and not the number of elements, we count the "gaps" between elements here,
  // but without the final gap).
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

  // Position of the last added element.
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
