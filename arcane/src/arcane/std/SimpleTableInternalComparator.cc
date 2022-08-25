// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SimpleTableInternalComparator.cc                            (C) 2000-2022 */
/*                                                                           */
/* Comparateur de SimpleTableInternal.                                       */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/std/SimpleTableInternalComparator.h"

#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/Numeric.h"

#include <optional>
#include <regex>
#include <string>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool SimpleTableInternalComparator::
compare(Real epsilon, bool compare_dimension_too)
{
  bool is_ok = true;

  const Integer dim1 = m_simple_table_internal_mng_reference.numberOfRows();
  const Integer dim2 = m_simple_table_internal_mng_reference.numberOfColumns();

  const Integer dim1_to_compare = m_simple_table_internal_mng_to_compare.numberOfRows();
  const Integer dim2_to_compare = m_simple_table_internal_mng_to_compare.numberOfColumns();

  if (compare_dimension_too && (dim1 != dim1_to_compare || dim2 != dim2_to_compare)) {
    m_simple_table_internal_reference->m_parallel_mng->traceMng()->warning() << "Dimensions not equals -- Expected dimensions: "
                                                                             << dim1 << "x" << dim2 << " -- Found dimensions: "
                                                                             << dim1_to_compare << "x" << dim2_to_compare;
    return false;
  }

  for (Integer i = 0; i < dim1; i++) {
    String row = m_simple_table_internal_mng_reference.rowName(i);
    // On regarde si l'on doit comparer la ligne actuelle.
    // On regarde si la ligne est présente dans le STI to_compare.
    if (!_exploreRows(row) || m_simple_table_internal_mng_to_compare.rowPosition(row) == -1)
      continue;

    for (Integer j = 0; j < dim2; j++) {
      String column = m_simple_table_internal_mng_reference.columnName(j);
      // On regarde si l'on doit comparer la colonne actuelle.
      // On regarde si la colonne est présente dans le STI to_compare.
      if (!_exploreColumn(column) || m_simple_table_internal_mng_to_compare.columnPosition(column) == -1)
        continue;

      const Real val1 = m_simple_table_internal_mng_reference.element(column, row, false);
      const Real val2 = m_simple_table_internal_mng_to_compare.element(column, row, false);

      if (!_isNearlyEqualWithAcceptableError(val1, val2, epsilon)) {
        m_simple_table_internal_reference->m_parallel_mng->traceMng()->warning() 
          << std::scientific << std::setprecision(std::numeric_limits<Real>::digits10)
          << "Values not equals -- Column name: \"" << column << "\" -- Row name: \"" << row 
          << "\" -- Expected value: " << val1 << " -- Found value: " << val2;
        is_ok = false;
      }
    }
  }
  return is_ok;
}

bool SimpleTableInternalComparator::
_isNearlyEqualWithAcceptableError(Real a, Real b, Real error)
{
  return (math::floor(a / error) == math::floor(b / error));
}

void SimpleTableInternalComparator::
clearComparator()
{
  m_regex_rows = "";
  m_is_excluding_regex_rows = false;

  m_regex_columns = "";
  m_is_excluding_regex_columns = false;

  m_rows_to_compare.clear();
  m_columns_to_compare.clear();
}

bool SimpleTableInternalComparator::
addColumnForComparing(const String& column_name)
{
  m_columns_to_compare.add(column_name);
  return true;
}
bool SimpleTableInternalComparator::
addRowForComparing(const String& row_name)
{
  m_rows_to_compare.add(row_name);
  return true;
}

void SimpleTableInternalComparator::
isAnArrayExclusiveColumns(bool is_exclusive)
{
  m_is_excluding_array_columns = is_exclusive;
}
void SimpleTableInternalComparator::
isAnArrayExclusiveRows(bool is_exclusive)
{
  m_is_excluding_array_rows = is_exclusive;
}

void SimpleTableInternalComparator::
editRegexColumns(const String& regex_column)
{
  m_regex_columns = regex_column;
}
void SimpleTableInternalComparator::
editRegexRows(const String& regex_row)
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

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<SimpleTableInternal> SimpleTableInternalComparator::
internalRef()
{
  return m_simple_table_internal_reference;
}

void SimpleTableInternalComparator::
setInternalRef(const Ref<SimpleTableInternal>& sti_ref)
{
  if (sti_ref.isNull())
    ARCANE_FATAL("La réference passée en paramètre est Null.");
  m_simple_table_internal_reference = sti_ref;
  m_simple_table_internal_mng_reference.setInternal(m_simple_table_internal_reference);
}

Ref<SimpleTableInternal> SimpleTableInternalComparator::
internalToCompare()
{
  return m_simple_table_internal_to_compare;
}

void SimpleTableInternalComparator::
setInternalToCompare(const Ref<SimpleTableInternal>& sti_to_compare)
{
  if (sti_to_compare.isNull())
    ARCANE_FATAL("La réference passée en paramètre est Null.");
  m_simple_table_internal_to_compare = sti_to_compare;
  m_simple_table_internal_mng_to_compare.setInternal(m_simple_table_internal_to_compare);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool SimpleTableInternalComparator::
_exploreColumn(const String& column_name)
{
  // S'il n'y a pas de précisions, on compare toutes les colonnes.
  if (m_columns_to_compare.empty() && m_regex_columns.empty()) {
    return true;
  }

  if (m_columns_to_compare.contains(column_name)) {
    return !m_is_excluding_array_columns;
  }

  // S'il n'est pas dans le tableau et qu'il n'a a pas de regex, on return false.
  else if (m_regex_columns.empty()) {
    return m_is_excluding_array_columns;
  }

  // Sinon, on regarde aussi la regex.
  // TODO : Voir s'il y a un interet de faire des regex en mode JS.
  std::regex self_regex(m_regex_columns.localstr(), std::regex_constants::ECMAScript | std::regex_constants::icase);

  // Si quelque chose dans le nom correspond à la regex.
  if (std::regex_search(column_name.localstr(), self_regex)) {
    return !m_is_excluding_regex_columns;
  }

  // Sinon.
  return m_is_excluding_regex_columns;
}

bool SimpleTableInternalComparator::
_exploreRows(const String& row_name)
{
  // S'il n'y a pas de précisions, on compare toutes les colonnes.
  if (m_rows_to_compare.empty() && m_regex_rows.empty()) {
    return true;
  }

  // D'abord, on regarde si le nom de la colonne est dans le tableau.
  if (m_rows_to_compare.contains(row_name)) {
    return !m_is_excluding_array_rows;
  }
  // S'il n'est pas dans le tableau et qu'il n'a a pas de regex, on return false.
  else if (m_regex_rows.empty()) {
    return m_is_excluding_array_rows;
  }

  // Sinon, on regarde aussi la regex.
  // TODO : Voir s'il y a un interet de faire des regex en mode JS.
  std::regex self_regex(m_regex_rows.localstr(), std::regex_constants::ECMAScript | std::regex_constants::icase);
  if (std::regex_search(row_name.localstr(), self_regex)) {
    return !m_is_excluding_regex_rows;
  }

  return m_is_excluding_regex_rows;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
