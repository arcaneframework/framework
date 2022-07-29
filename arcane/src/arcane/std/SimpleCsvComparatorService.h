// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SimpleCsvComparatorService.hh                                   (C) 2000-2022 */
/*                                                                           */
/* Service permettant de construire et de sortir un tableau au formet csv.   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifndef ARCANE_STD_SIMPLECSVCOMPARATORSERVICE_H
#define ARCANE_STD_SIMPLECSVCOMPARATORSERVICE_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ISimpleTableOutput.h"
#include "arcane/ISimpleTableComparator.h"

#include <arcane/Directory.h>
#include <arcane/utils/Iostream.h>

#include "arcane/std/SimpleCsvComparator_axl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class SimpleCsvComparatorService
: public ArcaneSimpleCsvComparatorObject
{
 public:
  explicit SimpleCsvComparatorService(const ServiceBuildInfo& sbi)
  : ArcaneSimpleCsvComparatorObject(sbi)
  , m_iSTO(nullptr)
  , m_is_file_open(false)
  {
    m_with_option = (sbi.creationType() == ST_CaseOption);
  }

  virtual ~SimpleCsvComparatorService() = default;

 public:
  
  void init(ISimpleTableOutput* ptr_sto) override;
  void editRefFileEntry(String path, String name) override;
  bool writeRefFile(Integer only_proc) override;
  bool readRefFile(Integer only_proc) override;
  bool isRefExist(Integer only_proc) override;
  void print() override;
  bool compareWithRef(Integer epsilon) override;

  bool addColumnToCompare(String name_column) override;
  bool addRowToCompare(String name_row) override;

  bool removeColumnToCompare(String name_column) override;
  bool removeRowToCompare(String name_row) override;

  bool editRegexColumns(String regex_column) override;
  bool editRegexRows(String regex_row) override;

  bool isARegexExclusiveColumns(bool is_exclusive) override;
  bool isARegexExclusiveRows(bool is_exclusive) override;

 protected:
  void _openFile(String name_file);
  bool _exploreColumn(Integer pos);
  bool _exploreRows(Integer pos);

 protected:
  Directory m_path_ref;
  String m_path_ref_str;
  String m_name_ref;

  std::ifstream m_ifstream;
  bool m_is_file_open;

  ISimpleTableOutput* m_iSTO;

  RealUniqueArray2 m_values_csv;

  StringUniqueArray m_name_rows;
  StringUniqueArray m_name_columns_with_name_of_tab;

  String m_regex_rows;
  bool m_is_excluding_regex_rows;

  String m_regex_columns;
  bool m_is_excluding_regex_columns;

  StringUniqueArray m_compared_rows;
  StringUniqueArray m_compared_columns;


  bool m_with_option;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE_SIMPLECSVCOMPARATOR(SimpleCsvComparator, SimpleCsvComparatorService);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
