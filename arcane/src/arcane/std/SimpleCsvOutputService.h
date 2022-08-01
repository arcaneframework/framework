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

#include "arcane/ISimpleTableOutput.h"
#include "arcane/Directory.h"
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
  , m_name_tab_computed(false)
  , m_name_tab_only_once(true)
  , m_precision_print(6)
  , m_is_fixed_print(true)
  , m_name_rows(0)
  , m_name_columns(0)
  , m_size_rows(0)
  , m_size_columns(0)
  , m_last_row(-1)
  , m_last_column(-1)
  {
    m_with_option = (sbi.creationType() == ST_CaseOption);
  }

  virtual ~SimpleCsvOutputService() = default;

 public:
  bool init() override;
  bool init(String name_table) override;
  bool init(String name_table, String name_dir) override;

  void clear() override;

  Integer addRow(String name_row) override;
  Integer addRow(String name_row, ConstArrayView<Real> elems) override;
  bool addRows(StringConstArrayView name_rows) override;

  Integer addColumn(String name_column) override;
  Integer addColumn(String name_column, ConstArrayView<Real> elems) override;
  bool addColumns(StringConstArrayView name_columns) override;

  bool addElemRow(Integer pos, Real elem) override;
  bool addElemRow(String name_row, Real elem, bool create_if_not_exist) override;
  bool addElemSameRow(Real elem) override;

  bool addElemsRow(Integer pos, ConstArrayView<Real> elems) override;
  bool addElemsRow(String name_row, ConstArrayView<Real> elems, bool create_if_not_exist) override;
  bool addElemsSameRow(ConstArrayView<Real> elems) override;

  bool addElemColumn(Integer pos, Real elem) override;
  bool addElemColumn(String name_column, Real elem, bool create_if_not_exist) override;
  bool addElemSameColumn(Real elem) override;

  bool addElemsColumn(Integer pos, ConstArrayView<Real> elems) override;
  bool addElemsColumn(String name_column, ConstArrayView<Real> elems, bool create_if_not_exist) override;
  bool addElemsSameColumn(ConstArrayView<Real> elems) override;

  bool editElemUp(Real elem, bool update_last_pos) override;
  bool editElemDown(Real elem, bool update_last_pos) override;
  bool editElemLeft(Real elem, bool update_last_pos) override;
  bool editElemRight(Real elem, bool update_last_pos) override;

  Real elemUp(bool update_last_pos) override;
  Real elemDown(bool update_last_pos) override;
  Real elemLeft(bool update_last_pos) override;
  Real elemRight(bool update_last_pos) override;

  bool editElem(Real elem) override;
  bool editElem(Integer pos_x, Integer pos_y, Real elem) override;
  bool editElem(String name_column, String name_row, Real elem) override;

  Real elem() override;
  Real elem(Integer pos_x, Integer pos_y, bool update_last_pos) override;
  Real elem(String name_column, String name_row, bool update_last_pos) override;

  RealUniqueArray row(Integer pos) override;
  RealUniqueArray column(Integer pos) override;

  RealUniqueArray row(String name_row) override;
  RealUniqueArray column(String name_column) override;

  Integer sizeRow(Integer pos) override;
  Integer sizeColumn(Integer pos) override;

  Integer sizeRow(String name_row) override;
  Integer sizeColumn(String name_column) override;

  Integer posRow(String name_row) override;
  Integer posColumn(String name_column) override;

  Integer numRows() override;
  Integer numColumns() override;

  bool editNameRow(Integer pos, String new_name) override;
  bool editNameRow(String name_row, String new_name) override;

  bool editNameColumn(Integer pos, String new_name) override;
  bool editNameColumn(String name_column, String new_name) override;

  Integer addAverageColumn(String name_column) override;

  void print(Integer only_proc) override;
  bool writeFile(Integer only_proc) override;
  bool writeFile(Directory root_dir, Integer only_proc) override;
  bool writeFile(String dir, Integer only_proc) override;

  Integer precision() override;
  void setPrecision(Integer precision) override;

  bool fixed() override;
  void setFixed(bool fixed) override;

  String outputDir() override;
  void setOutputDir(String dir) override;

  String tabName() override;
  void setTabName(String name) override;
  String fileName() override;
  
  Directory outputPath() override;
  Directory rootPath() override;
  
  String outputFileType() override;

  bool isOneFileByProcsPermited() override;

 private:
  bool _writeFile(Directory output_dir, Integer only_proc);
  String _computeFinal();
  void _print(std::ostream& stream);
  void _computeName();
  bool _createDirectory(Directory dir);
  bool _createOutputDirectory();
  bool _createRoot();

 private:
  String m_name_output_dir;

  Directory m_root;

  String m_name_tab;
  String m_name_csv;
  bool m_name_tab_computed;
  bool m_name_tab_only_once;

  String m_separator;
  Integer m_precision_print;
  bool m_is_fixed_print;

  const String m_output_file_type = "csv";

  UniqueArray2<Real> m_values_csv;

  UniqueArray<String> m_name_rows;
  UniqueArray<String> m_name_columns;

  // Tailles des lignes/colonnes
  // (et pas le nombre d'éléments, on compte les "trous" entre les éléments ici,
  // mais sans le trou de fin).
  // Ex. : {{"1", "2", "0", "3", "0", "0"},
  //        {"4", "5", "6", "0", "7", "8"},
  //        {"0", "0", "0", "0", "0", "0"}}

  //       m_size_rows[0] = 4
  //       m_size_rows[1] = 6
  //       m_size_rows[2] = 0
  //       m_size_rows.size() = 3

  //       m_size_columns[3] = 1
  //       m_size_columns[0; 1; 2; 4; 5] = 2
  //       m_size_columns.size() = 6

  UniqueArray<Integer> m_size_rows;
  UniqueArray<Integer> m_size_columns;

  // Dernier élement ajouté.
  Integer m_last_row;
  Integer m_last_column;

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
