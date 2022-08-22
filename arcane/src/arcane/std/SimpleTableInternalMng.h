// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SimpleTableInternalMng.h                                    (C) 2000-2022 */
/*                                                                           */
/* Classe permettant de modifier facilement un SimpleTableInternal.          */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifndef ARCANE_STD_SIMPLETABLEINTERNALMNG_H
#define ARCANE_STD_SIMPLETABLEINTERNALMNG_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ISimpleTableInternalMng.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class SimpleTableInternalMng
: public ISimpleTableInternalMng
{
 public:
  SimpleTableInternalMng(SimpleTableInternal* simple_table_internal)
  : m_simple_table_internal(simple_table_internal)
  {
  }

  SimpleTableInternalMng()
  : m_simple_table_internal(nullptr)
  {
  }

  virtual ~SimpleTableInternalMng() = default;

 public:
  void clearInternal() override;

  Integer addRow(const String& row_name) override;
  Integer addRow(const String& row_name, ConstArrayView<Real> elements) override;
  bool addRows(StringConstArrayView rows_names) override;

  Integer addColumn(const String& column_name) override;
  Integer addColumn(const String& column_name, ConstArrayView<Real> elements) override;
  bool addColumns(StringConstArrayView columns_names) override;

  bool addElementInRow(Integer position, Real element) override;
  bool addElementInRow(const String& row_name, Real element, bool create_if_not_exist) override;
  bool addElementInSameRow(Real element) override;

  bool addElementsInRow(Integer position, ConstArrayView<Real> elements) override;
  bool addElementsInRow(const String& row_name, ConstArrayView<Real> elements, bool create_if_not_exist) override;
  bool addElementsInSameRow(ConstArrayView<Real> elements) override;

  bool addElementInColumn(Integer position, Real element) override;
  bool addElementInColumn(const String& column_name, Real element, bool create_if_not_exist) override;
  bool addElementInSameColumn(Real element) override;

  bool addElementsInColumn(Integer position, ConstArrayView<Real> elements) override;
  bool addElementsInColumn(const String& column_name, ConstArrayView<Real> elements, bool create_if_not_exist) override;
  bool addElementsInSameColumn(ConstArrayView<Real> elements) override;

  bool editElementUp(Real element, bool update_last_position) override;
  bool editElementDown(Real element, bool update_last_position) override;
  bool editElementLeft(Real element, bool update_last_position) override;
  bool editElementRight(Real element, bool update_last_position) override;

  Real elementUp(bool update_last_position) override;
  Real elementDown(bool update_last_position) override;
  Real elementLeft(bool update_last_position) override;
  Real elementRight(bool update_last_position) override;

  bool editElement(Real element) override;
  bool editElement(Integer position_x, Integer position_y, Real element) override;
  bool editElement(const String& column_name, const String& row_name, Real element) override;

  Real element() override;
  Real element(Integer position_x, Integer position_y, bool update_last_position) override;
  Real element(const String& column_name, const String& row_name, bool update_last_position) override;

  RealUniqueArray row(Integer position) override;
  RealUniqueArray column(Integer position) override;

  RealUniqueArray row(const String& row_name) override;
  RealUniqueArray column(const String& column_name) override;

  Integer rowSize(Integer position) override;
  Integer columnSize(Integer position) override;

  Integer rowSize(const String& row_name) override;
  Integer columnSize(const String& column_name) override;

  Integer rowPosition(const String& row_name) override;
  Integer columnPosition(const String& column_name) override;

  Integer numberOfRows() override;
  Integer numberOfColumns() override;

  String rowName(Integer position) override;
  String columnName(Integer position) override;

  bool editRowName(Integer position, const String& new_name) override;
  bool editRowName(const String& row_name, const String& new_name) override;

  bool editColumnName(Integer position, const String& new_name) override;
  bool editColumnName(const String& column_name, const String& new_name) override;

  Integer addAverageColumn(const String& column_name) override;

  SimpleTableInternal* internal() override;
  void setInternal(SimpleTableInternal* simple_table_internal) override;

 protected:
  SimpleTableInternal* m_simple_table_internal;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
