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

#ifndef ARCANE_STD_SIMPLETABLEINTERNALCOMPARATOR_H
#define ARCANE_STD_SIMPLETABLEINTERNALCOMPARATOR_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ISimpleTableInternalMng.h"
#include "arcane/ISimpleTableReaderWriter.h"
#include "arcane/ISimpleTableInternalComparator.h"

#include "arcane/std/SimpleTableInternalMng.h"
#include "arcane/utils/Array.h"

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
  SimpleTableInternalComparator(SimpleTableInternal* sti_ref, SimpleTableInternal* sti_to_compare)
  : m_sti_ref(sti_ref)
  , m_sti_to_compare(sti_to_compare)
  , m_stm_ref(m_sti_ref)
  , m_stm_to_compare(m_sti_to_compare)
  , m_regex_rows("")
  , m_is_excluding_regex_rows(false)
  , m_regex_columns("")
  , m_is_excluding_regex_columns(false)
  , m_is_excluding_array_rows(false)
  , m_is_excluding_array_columns(false)
  {
  }

  SimpleTableInternalComparator()
  : m_sti_ref(nullptr)
  , m_sti_to_compare(nullptr)
  , m_stm_ref(m_sti_ref)
  , m_stm_to_compare(m_sti_to_compare)
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
  bool compare(Integer epsilon, bool compare_dim) override;

  void clearComparator() override;

  bool addColumnForComparing(const String& name_column) override;
  bool addRowForComparing(const String& name_row) override;

  void isAnArrayExclusiveColumns(bool is_exclusive) override;
  void isAnArrayExclusiveRows(bool is_exclusive) override;

  void editRegexColumns(const String& regex_column) override;
  void editRegexRows(const String& regex_row) override;

  void isARegexExclusiveColumns(bool is_exclusive) override;
  void isARegexExclusiveRows(bool is_exclusive) override;

  SimpleTableInternal* internalRef() override;
  void setInternalRef(SimpleTableInternal* sti) override;

  SimpleTableInternal* internalToCompare() override;
  void setInternalToCompare(SimpleTableInternal* sti) override;

 protected:
  bool _exploreColumn(const String& column_name);
  bool _exploreRows(const String& row_name);

 protected:
  SimpleTableInternal* m_sti_ref;
  SimpleTableInternal* m_sti_to_compare;

  SimpleTableInternalMng m_stm_ref;
  SimpleTableInternalMng m_stm_to_compare;

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
