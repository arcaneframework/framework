// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SimpleCsvOutputService.hh                                   (C) 2000-2022 */
/*                                                                           */
/* Service permettant de construire et de sortir un tableau au formet csv.   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifndef ARCANE_STD_SIMPLECSVOUTPUTSERVICE_H
#define ARCANE_STD_SIMPLECSVOUTPUTSERVICE_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/std/SimpleTableMng.h"
#include "arcane/std/SimpleTableWriterHelper.h"
#include "arcane/std/SimpleCsvReaderWriter.h"
#include "arcane/ISimpleTableOutput.h"
#include "arcane/Directory.h"
#include "arcane/IParallelMng.h"
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
  , m_internal(subDomain())
  , m_scrw(&m_internal)
  , m_stm(&m_internal)
  , m_stom(&m_scrw)
  {
    m_with_option = (sbi.creationType() == ST_CaseOption);
  }

  virtual ~SimpleCsvOutputService() = default;

 public:
  bool init() override;
  bool init(String name_table) override;
  bool init(String name_table, String name_dir) override { return m_stom.init(name_table, name_dir); };

  void clear() override { return m_stm.clear(); };

  Integer addRow(String name_row) override { return m_stm.addRow(name_row); };
  Integer addRow(String name_row, ConstArrayView<Real> elems) override { return m_stm.addRow(name_row, elems); };
  bool addRows(StringConstArrayView name_rows) override { return m_stm.addRows(name_rows); };

  Integer addColumn(String name_column) override { return m_stm.addColumn(name_column); };
  Integer addColumn(String name_column, ConstArrayView<Real> elems) override { return m_stm.addColumn(name_column, elems); };
  bool addColumns(StringConstArrayView name_columns) override { return m_stm.addColumns(name_columns); };

  bool addElemRow(Integer pos, Real elem) override { return m_stm.addElemRow(pos, elem); };
  bool addElemRow(String name_row, Real elem, bool create_if_not_exist) override { return m_stm.addElemRow(name_row, elem, create_if_not_exist); };
  bool addElemSameRow(Real elem) override { return m_stm.addElemSameRow(elem); };

  bool addElemsRow(Integer pos, ConstArrayView<Real> elems) override { return m_stm.addElemsRow(pos, elems); };
  bool addElemsRow(String name_row, ConstArrayView<Real> elems, bool create_if_not_exist) override { return m_stm.addElemsRow(name_row, elems, create_if_not_exist); };
  bool addElemsSameRow(ConstArrayView<Real> elems) override { return m_stm.addElemsSameRow(elems); };

  bool addElemColumn(Integer pos, Real elem) override { return m_stm.addElemColumn(pos, elem); };
  bool addElemColumn(String name_column, Real elem, bool create_if_not_exist) override { return m_stm.addElemColumn(name_column, elem, create_if_not_exist); };
  bool addElemSameColumn(Real elem) override { return m_stm.addElemSameColumn(elem); };

  bool addElemsColumn(Integer pos, ConstArrayView<Real> elems) override { return m_stm.addElemsColumn(pos, elems); };
  bool addElemsColumn(String name_column, ConstArrayView<Real> elems, bool create_if_not_exist) override { return m_stm.addElemsColumn(name_column, elems, create_if_not_exist); };
  bool addElemsSameColumn(ConstArrayView<Real> elems) override { return m_stm.addElemsSameColumn(elems); };

  bool editElemUp(Real elem, bool update_last_pos) override { return m_stm.editElemUp(elem, update_last_pos); };
  bool editElemDown(Real elem, bool update_last_pos) override { return m_stm.editElemDown(elem, update_last_pos); };
  bool editElemLeft(Real elem, bool update_last_pos) override { return m_stm.editElemLeft(elem, update_last_pos); };
  bool editElemRight(Real elem, bool update_last_pos) override { return m_stm.editElemRight(elem, update_last_pos); };

  Real elemUp(bool update_last_pos) override { return m_stm.elemUp(update_last_pos); };
  Real elemDown(bool update_last_pos) override { return m_stm.elemDown(update_last_pos); };
  Real elemLeft(bool update_last_pos) override { return m_stm.elemLeft(update_last_pos); };
  Real elemRight(bool update_last_pos) override { return m_stm.elemRight(update_last_pos); };

  bool editElem(Real elem) override { return m_stm.editElem(elem); };
  bool editElem(Integer pos_x, Integer pos_y, Real elem) override { return m_stm.editElem(pos_x, pos_y, elem); };
  bool editElem(String name_column, String name_row, Real elem) override { return m_stm.editElem(name_column, name_row, elem); };

  Real elem() override { return m_stm.elem(); };
  Real elem(Integer pos_x, Integer pos_y, bool update_last_pos) override { return m_stm.elem(pos_x, pos_y, update_last_pos); };
  Real elem(String name_column, String name_row, bool update_last_pos) override { return m_stm.elem(name_column, name_row, update_last_pos); };

  RealUniqueArray row(Integer pos) override { return m_stm.row(pos); };
  RealUniqueArray column(Integer pos) override { return m_stm.column(pos); };

  RealUniqueArray row(String name_row) override { return m_stm.row(name_row); };
  RealUniqueArray column(String name_column) override { return m_stm.column(name_column); };

  Integer sizeRow(Integer pos) override { return m_stm.sizeRow(pos); };
  Integer sizeColumn(Integer pos) override { return m_stm.sizeColumn(pos); };

  Integer sizeRow(String name_row) override { return m_stm.sizeRow(name_row); };
  Integer sizeColumn(String name_column) override { return m_stm.sizeColumn(name_column); };

  Integer posRow(String name_row) override { return m_stm.posRow(name_row); };
  Integer posColumn(String name_column) override { return m_stm.posColumn(name_column); };

  Integer numRows() override { return m_stm.numRows(); };
  Integer numColumns() override { return m_stm.numColumns(); };

  String nameRow(Integer pos) override { return m_stm.nameRow(pos); };
  String nameColumn(Integer pos) override { return m_stm.nameColumn(pos); };

  bool editNameRow(Integer pos, String new_name) override { return m_stm.editNameRow(pos, new_name); };
  bool editNameRow(String name_row, String new_name) override { return m_stm.editNameRow(name_row, new_name); };

  bool editNameColumn(Integer pos, String new_name) override { return m_stm.editNameColumn(pos, new_name); };
  bool editNameColumn(String name_column, String new_name) override { return m_stm.editNameColumn(name_column, new_name); };

  Integer addAverageColumn(String name_column) override { return m_stm.addAverageColumn(name_column); };

  void print(Integer only_proc) override { return m_stom.print(only_proc); };
  bool writeFile(Integer only_proc) override { return m_stom.writeFile(only_proc); };
  bool writeFile(Directory root_dir, Integer only_proc) override { return m_stom.writeFile(root_dir, only_proc); };
  bool writeFile(String dir, Integer only_proc) override;

  Integer precision() override { return m_stom.precision(); };
  void setPrecision(Integer precision) override { return m_stom.setPrecision(precision); };

  bool fixed() override { return m_stom.fixed(); };
  void setFixed(bool fixed) override { return m_stom.setFixed(fixed); };

  String outputDir() override { return m_stom.outputDir(); };
  void setOutputDir(String dir) override { return m_stom.setOutputDir(dir); };

  String tabName() override { return m_stom.tabName(); };
  void setTabName(String name) override { return m_stom.setTabName(name); };
  String fileName() override { return m_stom.fileName(); };
  
  Directory outputPath() override { return m_stom.outputPath(); };
  Directory rootPath() override { return m_stom.rootPath(); };
  
  String outputFileType() override { return m_stom.typeFile(); };

  bool isOneFileByProcsPermited() override { return m_stom.isOneFileByProcsPermited(); };

  SimpleTableInternal* internal() override { return &m_internal; };
  ISimpleTableReaderWriter* readerWriter() override { return &m_scrw; };

 private:
  SimpleTableInternal m_internal;
  SimpleCsvReaderWriter m_scrw;
  SimpleTableMng m_stm;
  SimpleTableWriterHelper m_stom;
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
