// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SimpleCsvOutputService.h                                    (C) 2000-2022 */
/*                                                                           */
/* Service permettant de construire et de sortir un tableau au formet csv.   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifndef ARCANE_STD_SIMPLECSVOUTPUTSERVICE_H
#define ARCANE_STD_SIMPLECSVOUTPUTSERVICE_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ISimpleTableOutput.h"

#include "arcane/std/SimpleCsvReaderWriter.h"
#include "arcane/std/SimpleTableInternalMng.h"
#include "arcane/std/SimpleTableWriterHelper.h"

#include "arcane/Directory.h"
#include "arcane/IMesh.h"

#include "arcane/std/SimpleCsvOutput_axl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class SimpleCsvOutputService
: public ArcaneSimpleCsvOutputObject
{
 public:
  explicit SimpleCsvOutputService(const ServiceBuildInfo& sbi)
  : ArcaneSimpleCsvOutputObject(sbi)
  , m_is_already_init(false)
  {
    m_with_option = (sbi.creationType() == ST_CaseOption);
  }

  virtual ~SimpleCsvOutputService() = default;

 public:
  bool init() override;
  bool init(const String& table_name) override;
  bool init(const String& table_name, const String& directory_name) override;

  void clear() override;

  Integer addRow(const String& row_name) override { return m_simple_table_internal_mng.addRow(row_name); };
  Integer addRow(const String& row_name, ConstArrayView<Real> elements) override { return m_simple_table_internal_mng.addRow(row_name, elements); };
  bool addRows(StringConstArrayView rows_names) override { return m_simple_table_internal_mng.addRows(rows_names); };

  Integer addColumn(const String& column_name) override { return m_simple_table_internal_mng.addColumn(column_name); };
  Integer addColumn(const String& column_name, ConstArrayView<Real> elements) override { return m_simple_table_internal_mng.addColumn(column_name, elements); };
  bool addColumns(StringConstArrayView columns_names) override { return m_simple_table_internal_mng.addColumns(columns_names); };

  bool addElementInRow(Integer position, Real element) override { return m_simple_table_internal_mng.addElementInRow(position, element); };
  bool addElementInRow(const String& row_name, Real element, bool create_if_not_exist) override { return m_simple_table_internal_mng.addElementInRow(row_name, element, create_if_not_exist); };
  bool addElementInSameRow(Real element) override { return m_simple_table_internal_mng.addElementInSameRow(element); };

  bool addElementsInRow(Integer position, ConstArrayView<Real> elements) override { return m_simple_table_internal_mng.addElementsInRow(position, elements); };
  bool addElementsInRow(const String& row_name, ConstArrayView<Real> elements, bool create_if_not_exist) override { return m_simple_table_internal_mng.addElementsInRow(row_name, elements, create_if_not_exist); };
  bool addElementsInSameRow(ConstArrayView<Real> elements) override { return m_simple_table_internal_mng.addElementsInSameRow(elements); };

  bool addElementInColumn(Integer position, Real element) override { return m_simple_table_internal_mng.addElementInColumn(position, element); };
  bool addElementInColumn(const String& column_name, Real element, bool create_if_not_exist) override { return m_simple_table_internal_mng.addElementInColumn(column_name, element, create_if_not_exist); };
  bool addElementInSameColumn(Real element) override { return m_simple_table_internal_mng.addElementInSameColumn(element); };

  bool addElementsInColumn(Integer position, ConstArrayView<Real> elements) override { return m_simple_table_internal_mng.addElementsInColumn(position, elements); };
  bool addElementsInColumn(const String& column_name, ConstArrayView<Real> elements, bool create_if_not_exist) override { return m_simple_table_internal_mng.addElementsInColumn(column_name, elements, create_if_not_exist); };
  bool addElementsInSameColumn(ConstArrayView<Real> elements) override { return m_simple_table_internal_mng.addElementsInSameColumn(elements); };

  bool editElementUp(Real element, bool update_last_position) override { return m_simple_table_internal_mng.editElementUp(element, update_last_position); };
  bool editElementDown(Real element, bool update_last_position) override { return m_simple_table_internal_mng.editElementDown(element, update_last_position); };
  bool editElementLeft(Real element, bool update_last_position) override { return m_simple_table_internal_mng.editElementLeft(element, update_last_position); };
  bool editElementRight(Real element, bool update_last_position) override { return m_simple_table_internal_mng.editElementRight(element, update_last_position); };

  Real elementUp(bool update_last_position) override { return m_simple_table_internal_mng.elementUp(update_last_position); };
  Real elementDown(bool update_last_position) override { return m_simple_table_internal_mng.elementDown(update_last_position); };
  Real elementLeft(bool update_last_position) override { return m_simple_table_internal_mng.elementLeft(update_last_position); };
  Real elementRight(bool update_last_position) override { return m_simple_table_internal_mng.elementRight(update_last_position); };

  bool editElement(Real element) override { return m_simple_table_internal_mng.editElement(element); };
  bool editElement(Integer position_x, Integer position_y, Real element) override { return m_simple_table_internal_mng.editElement(position_x, position_y, element); };
  bool editElement(const String& column_name, const String& row_name, Real element) override { return m_simple_table_internal_mng.editElement(column_name, row_name, element); };

  Real element() override { return m_simple_table_internal_mng.element(); };
  Real element(Integer position_x, Integer position_y, bool update_last_position) override { return m_simple_table_internal_mng.element(position_x, position_y, update_last_position); };
  Real element(const String& column_name, const String& row_name, bool update_last_position) override { return m_simple_table_internal_mng.element(column_name, row_name, update_last_position); };

  RealUniqueArray row(Integer position) override { return m_simple_table_internal_mng.row(position); };
  RealUniqueArray column(Integer position) override { return m_simple_table_internal_mng.column(position); };

  RealUniqueArray row(const String& row_name) override { return m_simple_table_internal_mng.row(row_name); };
  RealUniqueArray column(const String& column_name) override { return m_simple_table_internal_mng.column(column_name); };

  Integer rowSize(Integer position) override { return m_simple_table_internal_mng.rowSize(position); };
  Integer columnSize(Integer position) override { return m_simple_table_internal_mng.columnSize(position); };

  Integer rowSize(const String& row_name) override { return m_simple_table_internal_mng.rowSize(row_name); };
  Integer columnSize(const String& column_name) override { return m_simple_table_internal_mng.columnSize(column_name); };

  Integer rowPosition(const String& row_name) override { return m_simple_table_internal_mng.rowPosition(row_name); };
  Integer columnPosition(const String& column_name) override { return m_simple_table_internal_mng.columnPosition(column_name); };

  Integer numberOfRows() override { return m_simple_table_internal_mng.numberOfRows(); };
  Integer numberOfColumns() override { return m_simple_table_internal_mng.numberOfColumns(); };

  String rowName(Integer position) override { return m_simple_table_internal_mng.rowName(position); };
  String columnName(Integer position) override { return m_simple_table_internal_mng.columnName(position); };

  bool editRowName(Integer position, const String& new_name) override { return m_simple_table_internal_mng.editRowName(position, new_name); };
  bool editRowName(const String& row_name, const String& new_name) override { return m_simple_table_internal_mng.editRowName(row_name, new_name); };

  bool editColumnName(Integer position, const String& new_name) override { return m_simple_table_internal_mng.editColumnName(position, new_name); };
  bool editColumnName(const String& column_name, const String& new_name) override { return m_simple_table_internal_mng.editColumnName(column_name, new_name); };

  Integer addAverageColumn(const String& column_name) override { return m_simple_table_internal_mng.addAverageColumn(column_name); };

  void print(Integer rank) override { return m_simple_table_output_mng.print(rank); };
  bool writeFile(Integer rank) override { return m_simple_table_output_mng.writeFile(rank); };
  bool writeFile(const Directory& root_directory, Integer rank) override { return m_simple_table_output_mng.writeFile(root_directory, rank); };
  bool writeFile(const String& directory, Integer rank) override;

  Integer precision() override { return m_simple_table_output_mng.precision(); };
  void setPrecision(Integer precision) override { return m_simple_table_output_mng.setPrecision(precision); };

  bool isFixed() override { return m_simple_table_output_mng.isFixed(); };
  void setFixed(bool fixed) override { return m_simple_table_output_mng.setFixed(fixed); };

  bool isForcedToUseScientificNotation() override { return m_simple_table_output_mng.isForcedToUseScientificNotation(); };
  void setForcedToUseScientificNotation(bool use_scientific) override { return m_simple_table_output_mng.setForcedToUseScientificNotation(use_scientific); };

  String outputDirectory() override { return m_simple_table_output_mng.outputDirectory(); };
  void setOutputDirectory(const String& directory) override { return m_simple_table_output_mng.setOutputDirectory(directory); };

  String tableName() override { return m_simple_table_output_mng.tableName(); };
  void setTableName(const String& name) override { return m_simple_table_output_mng.setTableName(name); };
  String fileName() override { return m_simple_table_output_mng.fileName(); };

  Directory outputPath() override { return m_simple_table_output_mng.outputPath(); };
  Directory rootPath() override { return m_simple_table_output_mng.rootPath(); };

  String fileType() override { return m_simple_table_output_mng.fileType(); };

  bool isOneFileByRanksPermited() override { return m_simple_table_output_mng.isOneFileByRanksPermited(); };

  Ref<SimpleTableInternal> internal() override { return m_internal; };
  Ref<ISimpleTableReaderWriter> readerWriter() override { return m_simple_csv_reader_writer; };

 private:
  SimpleTableInternalMng m_simple_table_internal_mng;
  SimpleTableWriterHelper m_simple_table_output_mng;

  Ref<SimpleTableInternal> m_internal;
  Ref<SimpleCsvReaderWriter> m_simple_csv_reader_writer;

  bool m_is_already_init;
  bool m_with_option;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE_SIMPLECSVOUTPUT(SimpleCsvOutput, SimpleCsvOutputService);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
