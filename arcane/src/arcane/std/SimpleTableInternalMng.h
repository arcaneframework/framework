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
  SimpleTableInternalMng(SimpleTableInternal* sti)
  : m_sti(sti)
  {
  }

  SimpleTableInternalMng()
  : m_sti(nullptr)
  {
  }

  virtual ~SimpleTableInternalMng() = default;

 public:
  void clearInternal() override;

  Integer addRow(const String& name_row) override;
  Integer addRow(const String& name_row, ConstArrayView<Real> elems) override;
  bool addRows(StringConstArrayView name_rows) override;

  Integer addColumn(const String& name_column) override;
  Integer addColumn(const String& name_column, ConstArrayView<Real> elems) override;
  bool addColumns(StringConstArrayView name_columns) override;

  bool addElemRow(Integer pos, Real elem) override;
  bool addElemRow(const String& name_row, Real elem, bool create_if_not_exist) override;
  bool addElemSameRow(Real elem) override;

  bool addElemsRow(Integer pos, ConstArrayView<Real> elems) override;
  bool addElemsRow(const String& name_row, ConstArrayView<Real> elems, bool create_if_not_exist) override;
  bool addElemsSameRow(ConstArrayView<Real> elems) override;

  bool addElemColumn(Integer pos, Real elem) override;
  bool addElemColumn(const String& name_column, Real elem, bool create_if_not_exist) override;
  bool addElemSameColumn(Real elem) override;

  bool addElemsColumn(Integer pos, ConstArrayView<Real> elems) override;
  bool addElemsColumn(const String& name_column, ConstArrayView<Real> elems, bool create_if_not_exist) override;
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
  bool editElem(const String& name_column, const String& name_row, Real elem) override;

  Real elem() override;
  Real elem(Integer pos_x, Integer pos_y, bool update_last_pos) override;
  Real elem(const String& name_column, const String& name_row, bool update_last_pos) override;

  RealUniqueArray row(Integer pos) override;
  RealUniqueArray column(Integer pos) override;

  RealUniqueArray row(const String& name_row) override;
  RealUniqueArray column(const String& name_column) override;

  Integer sizeRow(Integer pos) override;
  Integer sizeColumn(Integer pos) override;

  Integer sizeRow(const String& name_row) override;
  Integer sizeColumn(const String& name_column) override;

  Integer posRow(const String& name_row) override;
  Integer posColumn(const String& name_column) override;

  Integer numRows() override;
  Integer numColumns() override;

  String nameRow(Integer pos) override;
  String nameColumn(Integer pos) override;

  bool editNameRow(Integer pos, const String& new_name) override;
  bool editNameRow(const String& name_row, const String& new_name) override;

  bool editNameColumn(Integer pos, const String& new_name) override;
  bool editNameColumn(const String& name_column, const String& new_name) override;

  Integer addAverageColumn(const String& name_column) override;

  SimpleTableInternal* internal() override;
  void setInternal(SimpleTableInternal* sti) override;

 protected:
  SimpleTableInternal* m_sti;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
