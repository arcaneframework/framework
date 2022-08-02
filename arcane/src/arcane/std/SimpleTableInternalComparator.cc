// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* TODO                                   (C) 2000-2022 */
/*                                                                           */
/*    */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/std/SimpleTableInternalComparator.h"

#include <arcane/Directory.h>
#include <arcane/IMesh.h>
#include <arcane/IParallelMng.h>
#include <arcane/ISubDomain.h>
#include "arcane/utils/StringBuilder.h"
#include "arcane/utils/Numeric.h"
#include "arcane/utils/ITraceMng.h"

#include <string>
#include <regex>
#include <optional>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool SimpleTableInternalComparator::
compare(Integer epsilon)
{
  bool is_ok = true;

  const Integer dim1 = m_stm_ref.numRows();
  const Integer dim2 = m_stm_ref.numColumns();

  for (Integer i = 0; i < dim1; i++) {
    // On regarde si l'on doit comparer la ligne actuelle.
    String row = m_stm_ref.nameRow(i);
    if(!_exploreRows(row)) continue;


    for (Integer j = 0; j < dim2; j++) {
    // On regarde si l'on doit comparer la colonne actuelle.
      String column = m_stm_ref.nameColumn(j);
      if(!_exploreColumn(column)) continue;


      const Real val1 = m_stm_ref.elem(column, row, false);
      const Real val2 = m_stm_to_compare.elem(column, row, false);

      if(!math::isNearlyEqualWithEpsilon(val1, val2, epsilon)) {
        m_sti_ref->m_mesh->traceMng()->warning() << "Values not equals -- Column name: \"" << column << "\" -- Row name: \"" << row << "\"";
        is_ok = false;
      }
    }
  }
  return is_ok;
}

bool SimpleTableInternalComparator::
addColumnToCompare(String name_column)
{
  m_compared_columns.add(name_column);
  return true;
}
bool SimpleTableInternalComparator::
addRowToCompare(String name_row)
{
  m_compared_rows.add(name_row);
  return true;
}

bool SimpleTableInternalComparator::
removeColumnToCompare(String name_column)
{
  std::optional index = m_compared_columns.span().findFirst(name_column);
  if(index) {
    m_compared_columns.remove(index.value());
    return true;
  }
  return false;
}
bool SimpleTableInternalComparator::
removeRowToCompare(String name_row)
{
  std::optional index = m_compared_rows.span().findFirst(name_row);
  if(index) {
    m_compared_rows.remove(index.value());
    return true;
  }
  return false;
}

void SimpleTableInternalComparator::
editRegexColumns(String regex_column)
{
  m_regex_columns = regex_column;
}
void SimpleTableInternalComparator::
editRegexRows(String regex_row)
{
  m_regex_rows = regex_row;
}

void SimpleTableInternalComparator::
isARegexExclusiveColumns(bool is_exclusive)
{
  m_is_excluding_regex_columns = is_exclusive;
}
void SimpleTableInternalComparator::
isARegexExclusiveRows(bool is_exclusive)
{
  m_is_excluding_regex_rows = is_exclusive;
}


SimpleTableInternal* SimpleTableInternalComparator::
internalRef() 
{
  return m_sti_ref;
}

void SimpleTableInternalComparator::
setInternalRef(SimpleTableInternal* sti_ref) 
{
  ARCANE_CHECK_PTR(sti_ref);
  m_sti_ref = sti_ref;
  m_stm_ref.setInternal(m_sti_ref);
}

void SimpleTableInternalComparator::
setInternalRef(SimpleTableInternal& sti_ref) 
{
  m_sti_ref = &sti_ref;
  m_stm_ref.setInternal(m_sti_ref);
}


SimpleTableInternal* SimpleTableInternalComparator::
internalToCompare() 
{
  return m_sti_ref;
}

void SimpleTableInternalComparator::
setInternalToCompare(SimpleTableInternal* sti_to_compare) 
{
  ARCANE_CHECK_PTR(sti_to_compare);
  m_sti_to_compare = sti_to_compare;
  m_stm_to_compare.setInternal(m_sti_to_compare);
}

void SimpleTableInternalComparator::
setInternalToCompare(SimpleTableInternal& sti_to_compare) 
{
  m_sti_to_compare = &sti_to_compare;
  m_stm_to_compare.setInternal(m_sti_to_compare);
}


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool SimpleTableInternalComparator::
_exploreColumn(String column_name)
{
  // S'il n'y a pas de précisions, on compare toutes les colonnes.
  if(m_compared_columns.empty() && m_regex_columns.empty()) {
    return true;
  }

  // D'abord, on regarde si le nom de la colonne est dans le tableau. 
  if(m_compared_columns.contains(column_name))
  {
    return true;
  }

  // S'il n'est pas dans le tableau et qu'il n'a a pas de regex, on return false.
  else if(m_regex_columns.empty())
  {
    return false;
  }

  // Sinon, on regarde aussi la regex.
  // TODO : Voir s'il y a un interet de faire des regex en mode JS.
  std::regex self_regex(m_regex_columns.localstr(), std::regex_constants::ECMAScript | std::regex_constants::icase);

  // Si quelque chose dans le nom correspond à la regex.
  if (std::regex_search(column_name.localstr(), self_regex))
  {
    return !m_is_excluding_regex_columns;
  }

  // Sinon.
  return m_is_excluding_regex_columns;
}

bool SimpleTableInternalComparator::
_exploreRows(String row_name)
{
  // S'il n'y a pas de précisions, on compare toutes les colonnes.
  if(m_compared_rows.empty() && m_regex_rows.empty()) {
    return true;
  }

  // D'abord, on regarde si le nom de la colonne est dans le tableau. 
  if(m_compared_rows.contains(row_name))
  {
    return true;
  }
  // S'il n'est pas dans le tableau et qu'il n'a a pas de regex, on return false.
  else if(m_regex_rows.empty())
  {
    return false;
  }

  // Sinon, on regarde aussi la regex.
  // TODO : Voir s'il y a un interet de faire des regex en mode JS.
  std::regex self_regex(m_regex_rows.localstr(), std::regex_constants::ECMAScript | std::regex_constants::icase);
  if (std::regex_search(row_name.localstr(), self_regex))
  {
    return !m_is_excluding_regex_rows;
  }

  return m_is_excluding_regex_rows;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
