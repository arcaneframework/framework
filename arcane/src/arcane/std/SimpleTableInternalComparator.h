// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SimpleTableInternalComparator.h                             (C) 2000-2022 */
/*                                                                           */
/* Comparateur de SimpleTableInternal.                                       */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifndef ARCANE_STD_SIMPLETABLEINTERNALCOMPARATOR_H
#define ARCANE_STD_SIMPLETABLEINTERNALCOMPARATOR_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ISimpleTableInternalComparator.h"
#include "arcane/ISimpleTableInternalMng.h"
#include "arcane/ISimpleTableReaderWriter.h"

#include "arcane/std/SimpleTableInternalMng.h"
#include "arcane/utils/Array.h"
#include "arcane/utils/FatalErrorException.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class SimpleTableInternalComparator
: public ISimpleTableInternalComparator
{
 public:
  SimpleTableInternalComparator(const Ref<SimpleTableInternal>& sti_ref, const Ref<SimpleTableInternal>& sti_to_compare)
  : m_simple_table_internal_reference(sti_ref)
  , m_simple_table_internal_to_compare(sti_to_compare)
  , m_simple_table_internal_mng_reference(m_simple_table_internal_reference)
  , m_simple_table_internal_mng_to_compare(m_simple_table_internal_to_compare)
  , m_regex_rows("")
  , m_is_excluding_regex_rows(false)
  , m_regex_columns("")
  , m_is_excluding_regex_columns(false)
  , m_is_excluding_array_rows(false)
  , m_is_excluding_array_columns(false)
  {
    if (sti_ref.isNull() || sti_to_compare.isNull())
      ARCANE_FATAL("La réference passée en paramètre est Null.");
  }

  SimpleTableInternalComparator()
  : m_simple_table_internal_reference()
  , m_simple_table_internal_to_compare()
  , m_simple_table_internal_mng_reference()
  , m_simple_table_internal_mng_to_compare()
  , m_regex_rows("")
  , m_is_excluding_regex_rows(false)
  , m_regex_columns("")
  , m_is_excluding_regex_columns(false)
  , m_is_excluding_array_rows(false)
  , m_is_excluding_array_columns(false)
  {
  }

  virtual ~SimpleTableInternalComparator() = default;

 public:
  bool compare(Integer epsilon, bool compare_dimension_too) override;

  void clearComparator() override;

  bool addColumnForComparing(const String& column_name) override;
  bool addRowForComparing(const String& row_name) override;

  void isAnArrayExclusiveColumns(bool is_exclusive) override;
  void isAnArrayExclusiveRows(bool is_exclusive) override;

  void editRegexColumns(const String& regex_column) override;
  void editRegexRows(const String& regex_row) override;

  void isARegexExclusiveColumns(bool is_exclusive) override;
  void isARegexExclusiveRows(bool is_exclusive) override;

  Ref<SimpleTableInternal> internalRef() override;
  void setInternalRef(const Ref<SimpleTableInternal>& simple_table_internal) override;

  Ref<SimpleTableInternal> internalToCompare() override;
  void setInternalToCompare(const Ref<SimpleTableInternal>& simple_table_internal) override;

 protected:
  bool _exploreColumn(const String& column_name);
  bool _exploreRows(const String& row_name);

 protected:
  Ref<SimpleTableInternal> m_simple_table_internal_reference;
  Ref<SimpleTableInternal> m_simple_table_internal_to_compare;

  SimpleTableInternalMng m_simple_table_internal_mng_reference;
  SimpleTableInternalMng m_simple_table_internal_mng_to_compare;

  String m_regex_rows;
  bool m_is_excluding_regex_rows;

  String m_regex_columns;
  bool m_is_excluding_regex_columns;

  StringUniqueArray m_rows_to_compare;
  bool m_is_excluding_array_rows;

  StringUniqueArray m_columns_to_compare;
  bool m_is_excluding_array_columns;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
