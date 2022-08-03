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

#include "arcane/ISimpleTableMng.h"
#include "arcane/std/SimpleTableMng.h"
#include "arcane/ISimpleTableReaderWriter.h"
#include "arcane/ISimpleTableInternalComparator.h"
#include "arcane/Directory.h"
#include "arcane/utils/Array.h"
#include "arcane/utils/Array2.h"

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
  {

  }

  // SimpleTableInternalComparator()
  // : m_sti_ref(nullptr)
  // , m_sti_to_compare(nullptr)
  // {
  //   std::cout << "Attention, STIC vide !" << std::endl;
  // }

  virtual ~SimpleTableInternalComparator() = default;

 public:
  bool compare(Integer epsilon) override;

  void clear() override;

  bool addColumnToCompare(String name_column) override;
  bool addRowToCompare(String name_row) override;

  bool removeColumnToCompare(String name_column) override;
  bool removeRowToCompare(String name_row) override;

  void editRegexColumns(String regex_column) override;
  void editRegexRows(String regex_row) override;

  void isARegexExclusiveColumns(bool is_exclusive) override;
  void isARegexExclusiveRows(bool is_exclusive) override;

  SimpleTableInternal* internalRef() override;
  void setInternalRef(SimpleTableInternal* sti) override;
  void setInternalRef(SimpleTableInternal& sti) override;

  SimpleTableInternal* internalToCompare() override;
  void setInternalToCompare(SimpleTableInternal* sti) override;
  void setInternalToCompare(SimpleTableInternal& sti) override;

 private:
  bool _exploreColumn(String column_name);
  bool _exploreRows(String row_name);

 private:
  String m_regex_rows;
  bool m_is_excluding_regex_rows;

  String m_regex_columns;
  bool m_is_excluding_regex_columns;

  StringUniqueArray m_compared_rows;
  StringUniqueArray m_compared_columns;

  SimpleTableInternal* m_sti_ref;
  SimpleTableInternal* m_sti_to_compare;

  SimpleTableMng m_stm_ref;
  SimpleTableMng m_stm_to_compare;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
