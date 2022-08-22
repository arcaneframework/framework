// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SimpleCsvComparatorService.h                                (C) 2000-2022 */
/*                                                                           */
/* Service permettant de comparer un ISimpleTableOutput avec un fichier de   */
/* référence en format csv.                                                  */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifndef ARCANE_STD_SIMPLECSVCOMPARATORSERVICE_H
#define ARCANE_STD_SIMPLECSVCOMPARATORSERVICE_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ISimpleTableComparator.h"
#include "arcane/ISimpleTableInternalMng.h"
#include "arcane/ISimpleTableOutput.h"

#include "arcane/std/SimpleCsvReaderWriter.h"
#include "arcane/std/SimpleTableInternalComparator.h"

#include "arcane/Directory.h"
#include "arcane/IMesh.h"

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
  , m_simple_table_output_ptr(nullptr)
  , m_reference_path()
  , m_root_path()
  , m_output_directory("_ref")
  , m_file_name("")
  , m_table_name("")
  , m_is_file_open(false)
  , m_is_file_read(false)
  , m_simple_table_internal_reference(nullptr)
  , m_simple_table_internal_to_compare(mesh()->parallelMng())
  , m_simple_table_internal_comparator(m_simple_table_internal_reference, &m_simple_table_internal_to_compare)
  , m_simple_csv_reader_writer(&m_simple_table_internal_to_compare)
  {
    m_with_option = (sbi.creationType() == ST_CaseOption);
  }

  virtual ~SimpleCsvComparatorService() = default;

 public:
  void init(ISimpleTableOutput* simple_table_output_ptr) override;
  void clear() override;
  void editRootDirectory(const Directory& root_directory) override;
  void print(Integer rank) override;
  bool writeReferenceFile(Integer rank) override;
  bool readReferenceFile(Integer rank) override;
  bool isReferenceExist(Integer rank) override;
  bool compareWithReference(Integer rank, Integer epsilon, bool compare_dimension_too) override;

  bool addColumnForComparing(const String& column_name) override;
  bool addRowForComparing(const String& row_name) override;

  void isAnArrayExclusiveColumns(bool is_exclusive) override;
  void isAnArrayExclusiveRows(bool is_exclusive) override;

  void editRegexColumns(const String& regex_column) override;
  void editRegexRows(const String& regex_row) override;

  void isARegexExclusiveColumns(bool is_exclusive) override;
  void isARegexExclusiveRows(bool is_exclusive) override;

 private:
  bool _exploreColumn(Integer position);
  bool _exploreRows(Integer position);

 private:
  ISimpleTableOutput* m_simple_table_output_ptr;

  Directory m_reference_path;
  Directory m_root_path;

  String m_output_directory;
  String m_file_name;
  String m_table_name;

  std::ifstream m_ifstream;
  bool m_is_file_open;
  bool m_is_file_read;

  SimpleTableInternal* m_simple_table_internal_reference;
  SimpleTableInternal m_simple_table_internal_to_compare;
  SimpleTableInternalComparator m_simple_table_internal_comparator;
  SimpleCsvReaderWriter m_simple_csv_reader_writer;

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
