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
  , m_output_dir("_ref")
  , m_tab_name("Table")
  , m_is_file_open(false)
  , m_is_file_read(false)
  , m_iSTO(nullptr)
  , m_regex_rows()
  , m_is_excluding_regex_rows(false)
  , m_regex_columns()
  , m_is_excluding_regex_columns(false)
  {
    m_with_option = (sbi.creationType() == ST_CaseOption);
  }

  virtual ~SimpleCsvComparatorService() = default;

 public:
  
  void init(ISimpleTableOutput* ptr_sto) override;
  void clear() override;
  void editRootDir(Directory root_dir) override;
  bool writeRefFile(Integer only_proc) override;
  bool readRefFile(Integer only_proc) override;
  bool isRefExist(Integer only_proc) override;
  bool compareWithRef(Integer only_proc, Integer epsilon) override;

  bool addColumnToCompare(String name_column) override;
  bool addRowToCompare(String name_row) override;

  bool removeColumnToCompare(String name_column) override;
  bool removeRowToCompare(String name_row) override;

  void editRegexColumns(String regex_column) override;
  void editRegexRows(String regex_row) override;

  void isARegexExclusiveColumns(bool is_exclusive) override;
  void isARegexExclusiveRows(bool is_exclusive) override;

 protected:
  void _openFile(String name_file);
  void _closeFile();
  bool _exploreColumn(Integer pos);
  bool _exploreRows(Integer pos);

 protected:
  Directory m_ref_path;
  Directory m_root_path;
  String m_output_dir;
  String m_tab_name;
  String m_file_name;

  std::ifstream m_ifstream;
  bool m_is_file_open;
  bool m_is_file_read;

  ISimpleTableOutput* m_iSTO;
  bool m_is_csv_implem;

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
