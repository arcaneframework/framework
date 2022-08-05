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

#include "arcane/std/SimpleTableInternalMng.h"

#include <arcane/Directory.h>
#include <arcane/IMesh.h>
#include <arcane/IParallelMng.h>
#include "arcane/utils/StringBuilder.h"

#include <optional>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SimpleTableInternalMng::
clearInternal()
{
  ARCANE_CHECK_PTR(m_sti);
  m_sti->clear();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer SimpleTableInternalMng::
addRow(const String& name_row)
{
  ARCANE_CHECK_PTR(m_sti);
  Integer pos = m_sti->m_values_csv.dim1Size();
  m_sti->m_values_csv.resize(pos + 1);

  m_sti->m_name_rows.add(name_row);
  m_sti->m_size_rows.add(0);

  m_sti->m_last_row = pos;

  return pos;
}

Integer SimpleTableInternalMng::
addRow(const String& name_row, ConstArrayView<Real> elems)
{
  ARCANE_CHECK_PTR(m_sti);
  Integer pos = m_sti->m_values_csv.dim1Size();
  m_sti->m_values_csv.resize(pos + 1);

  m_sti->m_name_rows.add(name_row);
  m_sti->m_size_rows.add(0);

  addElemsRow(pos, elems);

  return pos;
}

bool SimpleTableInternalMng::
addRows(StringConstArrayView name_rows)
{
  ARCANE_CHECK_PTR(m_sti);
  Integer size = name_rows.size();
  if (size == 0)
    return true;

  Integer pos = m_sti->m_values_csv.dim1Size();
  m_sti->m_values_csv.resize(pos + size);

  m_sti->m_name_rows.addRange(name_rows);
  m_sti->m_size_rows.addRange(IntegerUniqueArray(size, 0));

  m_sti->m_last_row = pos;

  return true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer SimpleTableInternalMng::
addColumn(const String& name_column)
{
  ARCANE_CHECK_PTR(m_sti);
  Integer pos = m_sti->m_values_csv.dim2Size();
  m_sti->m_values_csv.resize(m_sti->m_values_csv.dim1Size(), pos + 1);

  m_sti->m_name_columns.add(name_column);
  m_sti->m_size_columns.add(0);

  m_sti->m_last_column = pos;

  return pos;
}

Integer SimpleTableInternalMng::
addColumn(const String& name_column, ConstArrayView<Real> elems)
{
  ARCANE_CHECK_PTR(m_sti);
  Integer pos = m_sti->m_values_csv.dim2Size();
  m_sti->m_values_csv.resize(m_sti->m_values_csv.dim1Size(), pos + 1);

  m_sti->m_name_columns.add(name_column);
  m_sti->m_size_columns.add(0);

  addElemsColumn(pos, elems);

  return pos;
}

bool SimpleTableInternalMng::
addColumns(StringConstArrayView name_columns)
{
  ARCANE_CHECK_PTR(m_sti);
  Integer size = name_columns.size();
  if (size == 0)
    return true;

  Integer pos = m_sti->m_values_csv.dim2Size();
  m_sti->m_values_csv.resize(m_sti->m_values_csv.dim1Size(), pos + size);

  m_sti->m_name_columns.addRange(name_columns);
  m_sti->m_size_columns.addRange(IntegerUniqueArray(size, 0));

  m_sti->m_last_column = pos;

  return true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool SimpleTableInternalMng::
addElemRow(Integer pos, Real elem)
{
  ARCANE_CHECK_PTR(m_sti);
  if (pos < 0 || pos >= m_sti->m_values_csv.dim1Size())
    return false;

  ArrayView<Real> view = m_sti->m_values_csv[pos];
  Integer size_row = m_sti->m_size_rows[pos];

  if (m_sti->m_values_csv.dim2Size() < size_row + 1)
    return false;

  view[size_row] = elem;

  m_sti->m_last_row = pos;
  m_sti->m_last_column = size_row;

  m_sti->m_size_rows[pos]++;
  // Il peut y avoir des élements sur la ligne d'après à la même colonne.
  // Exemple : addElemRow(pos=L01, elem=NEW):
  // aaa|C00|C01|C02
  // L00|123|456|789
  // L01|147|NEW|
  // L02|159|753|852
  // Il y a 753 donc la taille de la colonne reste égale à 3.
  m_sti->m_size_columns[size_row] = std::max(pos + 1, m_sti->m_size_columns[size_row]);

  return true;
}

bool SimpleTableInternalMng::
addElemRow(const String& name_row, Real elem, bool create_if_not_exist)
{
  ARCANE_CHECK_PTR(m_sti);
  std::optional<Integer> pos = m_sti->m_name_rows.span().findFirst(name_row);

  if (pos)
    return addElemRow(pos.value(), elem);
  else if (create_if_not_exist)
    return addElemRow(addRow(name_row), elem);
  else
    return false;
}

bool SimpleTableInternalMng::
addElemSameRow(Real elem)
{
  ARCANE_CHECK_PTR(m_sti);
  if (m_sti->m_last_row == -1 || m_sti->m_last_column == -1)
    return false;
  return addElemRow(m_sti->m_last_row, elem);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool SimpleTableInternalMng::
addElemsRow(Integer pos, ConstArrayView<Real> elems)
{
  ARCANE_CHECK_PTR(m_sti);
  if (pos < 0 || pos >= m_sti->m_values_csv.dim1Size())
    return false;

  ArrayView<Real> view = m_sti->m_values_csv[pos];
  Integer size_row = m_sti->m_size_rows[pos];
  Integer min_size = (elems.size() <= m_sti->m_values_csv.dim2Size() - size_row
                      ? elems.size()
                      : m_sti->m_values_csv.dim2Size() - size_row);

  for (Integer i = 0; i < min_size; i++) {
    view[i + size_row] = elems[i];
    m_sti->m_size_columns[i + size_row] = std::max(pos + 1, m_sti->m_size_columns[i + size_row]);
  }
  m_sti->m_size_rows[pos] += min_size;

  m_sti->m_last_row = pos;
  m_sti->m_last_column = m_sti->m_size_rows[pos] - 1;

  return elems.size() <= m_sti->m_values_csv.dim2Size() - size_row;
}

bool SimpleTableInternalMng::
addElemsRow(const String& name_row, ConstArrayView<Real> elems, bool create_if_not_exist)
{
  ARCANE_CHECK_PTR(m_sti);
  std::optional<Integer> pos = m_sti->m_name_rows.span().findFirst(name_row);

  if (pos)
    return addElemsRow(pos.value(), elems);
  // Permet d'avoir un return bool (sinon on pourrait simplement faire addRow(name_row, elems)).
  else if (create_if_not_exist)
    return addElemsRow(addRow(name_row), elems);
  else
    return false;
}

bool SimpleTableInternalMng::
addElemsSameRow(ConstArrayView<Real> elems)
{
  ARCANE_CHECK_PTR(m_sti);
  if (m_sti->m_last_row == -1 || m_sti->m_last_column == -1)
    return false;
  return addElemsRow(m_sti->m_last_row, elems);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool SimpleTableInternalMng::
addElemColumn(Integer pos, Real elem)
{
  ARCANE_CHECK_PTR(m_sti);
  if (pos < 0 || pos >= m_sti->m_values_csv.dim2Size())
    return false;

  Integer size_column = m_sti->m_size_columns[pos];

  if (m_sti->m_values_csv.dim1Size() < size_column + 1)
    return false;

  m_sti->m_values_csv[size_column][pos] = elem;

  m_sti->m_last_column = pos;
  m_sti->m_last_row = size_column;

  m_sti->m_size_columns[pos]++;
  m_sti->m_size_rows[size_column] = std::max(pos + 1, m_sti->m_size_rows[size_column]);

  return true;
}

bool SimpleTableInternalMng::
addElemColumn(const String& name_column, Real elem, bool create_if_not_exist)
{
  ARCANE_CHECK_PTR(m_sti);
  std::optional<Integer> pos = m_sti->m_name_columns.span().findFirst(name_column);

  if (pos)
    return addElemColumn(pos.value(), elem);
  else if (create_if_not_exist)
    return addElemColumn(addColumn(name_column), elem);
  else
    return false;
}

bool SimpleTableInternalMng::
addElemSameColumn(Real elem)
{
  ARCANE_CHECK_PTR(m_sti);
  if (m_sti->m_last_row == -1 || m_sti->m_last_column == -1)
    return false;
  return addElemColumn(m_sti->m_last_column, elem);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool SimpleTableInternalMng::
addElemsColumn(Integer pos, ConstArrayView<Real> elems)
{
  ARCANE_CHECK_PTR(m_sti);
  if (pos < 0 || pos >= m_sti->m_values_csv.dim2Size())
    return false;

  Integer size_column = m_sti->m_size_columns[pos];
  Integer min_size = (elems.size() <= m_sti->m_values_csv.dim1Size() - size_column
                      ? elems.size()
                      : m_sti->m_values_csv.dim1Size() - size_column);

  for (Integer i = 0; i < min_size; i++) {
    m_sti->m_values_csv[i + size_column][pos] = elems[i];
    m_sti->m_size_rows[i + size_column] = std::max(pos + 1, m_sti->m_size_rows[i + size_column]);
  }
  m_sti->m_size_columns[pos] += min_size;

  m_sti->m_last_column = pos;
  m_sti->m_last_row = m_sti->m_size_columns[pos] - 1;

  return elems.size() <= m_sti->m_values_csv.dim1Size() - size_column;
}

bool SimpleTableInternalMng::
addElemsColumn(const String& name_column, ConstArrayView<Real> elems, bool create_if_not_exist)
{
  ARCANE_CHECK_PTR(m_sti);
  std::optional<Integer> pos = m_sti->m_name_columns.span().findFirst(name_column);

  if (pos)
    return addElemsColumn(pos.value(), elems);
  // Permet d'avoir un return bool (sinon on pourrait simplement faire addColumn(name_column, elems)).
  else if (create_if_not_exist)
    return addElemsColumn(addColumn(name_column), elems);
  else
    return false;
}

bool SimpleTableInternalMng::
addElemsSameColumn(ConstArrayView<Real> elems)
{
  ARCANE_CHECK_PTR(m_sti);
  if (m_sti->m_last_row == -1 || m_sti->m_last_column == -1)
    return false;
  return addElemsColumn(m_sti->m_last_column, elems);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool SimpleTableInternalMng::
editElemUp(Real elem, bool update_last_pos)
{
  ARCANE_CHECK_PTR(m_sti);
  if (m_sti->m_last_row == -1 || m_sti->m_last_column == -1 || m_sti->m_last_row - 1 < 0)
    return false;
  m_sti->m_last_row--;

  // Pas besoin d'ajuster la taille de la colonne car on est sûr que m_sti->m_size_columns[m_sti->m_last_column] >= m_sti->m_last_row.
  if (m_sti->m_size_rows[m_sti->m_last_row] <= m_sti->m_last_column)
    m_sti->m_size_rows[m_sti->m_last_row] = m_sti->m_last_column + 1;

  m_sti->m_values_csv[m_sti->m_last_row][m_sti->m_last_column] = elem;
  if (!update_last_pos)
    m_sti->m_last_row++;
  return true;
}

bool SimpleTableInternalMng::
editElemDown(Real elem, bool update_last_pos)
{
  ARCANE_CHECK_PTR(m_sti);
  if (m_sti->m_last_row == -1 || m_sti->m_last_column == -1 || m_sti->m_last_row + 1 >= m_sti->m_values_csv.dim1Size())
    return false;
  m_sti->m_last_row++;

  if (m_sti->m_size_rows[m_sti->m_last_row] <= m_sti->m_last_column)
    m_sti->m_size_rows[m_sti->m_last_row] = m_sti->m_last_column + 1;
  if (m_sti->m_size_columns[m_sti->m_last_column] <= m_sti->m_last_row)
    m_sti->m_size_columns[m_sti->m_last_column] = m_sti->m_last_row + 1;

  m_sti->m_values_csv[m_sti->m_last_row][m_sti->m_last_column] = elem;
  if (!update_last_pos)
    m_sti->m_last_row--;
  return true;
}

bool SimpleTableInternalMng::
editElemLeft(Real elem, bool update_last_pos)
{
  ARCANE_CHECK_PTR(m_sti);
  if (m_sti->m_last_row == -1 || m_sti->m_last_column == -1 || m_sti->m_last_column - 1 < 0)
    return false;
  m_sti->m_last_column--;

  // Pas besoin d'ajuster la taille de la ligne car on est sûr que m_sti->m_size_rows[m_sti->m_last_row] >= m_sti->m_last_column.
  if (m_sti->m_size_columns[m_sti->m_last_column] <= m_sti->m_last_row)
    m_sti->m_size_columns[m_sti->m_last_column] = m_sti->m_last_row + 1;

  m_sti->m_values_csv[m_sti->m_last_row][m_sti->m_last_column] = elem;
  if (!update_last_pos)
    m_sti->m_last_column++;
  return true;
}

bool SimpleTableInternalMng::
editElemRight(Real elem, bool update_last_pos)
{
  ARCANE_CHECK_PTR(m_sti);
  if (m_sti->m_last_row == -1 || m_sti->m_last_column == -1 || m_sti->m_last_column + 1 >= m_sti->m_values_csv.dim2Size())
    return false;
  m_sti->m_last_column++;

  if (m_sti->m_size_rows[m_sti->m_last_row] <= m_sti->m_last_column)
    m_sti->m_size_rows[m_sti->m_last_row] = m_sti->m_last_column + 1;
  if (m_sti->m_size_columns[m_sti->m_last_column] <= m_sti->m_last_row)
    m_sti->m_size_columns[m_sti->m_last_column] = m_sti->m_last_row + 1;

  m_sti->m_values_csv[m_sti->m_last_row][m_sti->m_last_column] = elem;
  if (!update_last_pos)
    m_sti->m_last_column--;
  return true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Real SimpleTableInternalMng::
elemUp(bool update_last_pos)
{
  ARCANE_CHECK_PTR(m_sti);
  if (m_sti->m_last_row == -1 || m_sti->m_last_column == -1 || m_sti->m_last_row - 1 < 0)
    return 0;

  // Par rapport à editElemUp(), si on ne veut pas mettre à jour la dernière position,
  // on ne verifie pas ni modifie m_sti->m_size_rows.
  if (update_last_pos) {
    m_sti->m_last_row--;
    // Pas besoin d'ajuster la taille de la colonne car on est sûr que m_sti->m_size_columns[m_sti->m_last_column] >= m_sti->m_last_row.
    if (m_sti->m_size_rows[m_sti->m_last_row] <= m_sti->m_last_column)
      m_sti->m_size_rows[m_sti->m_last_row] = m_sti->m_last_column + 1;
    return m_sti->m_values_csv[m_sti->m_last_row][m_sti->m_last_column];
  }

  return m_sti->m_values_csv[m_sti->m_last_row - 1][m_sti->m_last_column];
}

Real SimpleTableInternalMng::
elemDown(bool update_last_pos)
{
  ARCANE_CHECK_PTR(m_sti);
  if (m_sti->m_last_row == -1 || m_sti->m_last_column == -1 || m_sti->m_last_row + 1 >= m_sti->m_values_csv.dim1Size())
    return 0;

  // Par rapport à editElemDown(), si on ne veut pas mettre à jour la dernière position,
  // on ne verifie pas ni modifie m_sti->m_size_rows.
  if (update_last_pos) {
    m_sti->m_last_row++;

    if (m_sti->m_size_rows[m_sti->m_last_row] <= m_sti->m_last_column)
      m_sti->m_size_rows[m_sti->m_last_row] = m_sti->m_last_column + 1;
    if (m_sti->m_size_columns[m_sti->m_last_column] <= m_sti->m_last_row)
      m_sti->m_size_columns[m_sti->m_last_column] = m_sti->m_last_row + 1;
    return m_sti->m_values_csv[m_sti->m_last_row][m_sti->m_last_column];
  }
  return m_sti->m_values_csv[m_sti->m_last_row + 1][m_sti->m_last_column];
}

Real SimpleTableInternalMng::
elemLeft(bool update_last_pos)
{
  ARCANE_CHECK_PTR(m_sti);
  if (m_sti->m_last_row == -1 || m_sti->m_last_column == -1 || m_sti->m_last_column - 1 < 0)
    return 0;

  // Par rapport à editElemLeft(), si on ne veut pas mettre à jour la dernière position,
  // on ne verifie pas ni modifie m_sti->m_size_columns.
  if (update_last_pos) {
    m_sti->m_last_column--;

    // Pas besoin d'ajuster la taille de la ligne car on est sûr que m_sti->m_size_rows[m_sti->m_last_row] >= m_sti->m_last_column.
    if (m_sti->m_size_columns[m_sti->m_last_column] <= m_sti->m_last_row)
      m_sti->m_size_columns[m_sti->m_last_column] = m_sti->m_last_row + 1;
    return m_sti->m_values_csv[m_sti->m_last_row][m_sti->m_last_column];
  }
  return m_sti->m_values_csv[m_sti->m_last_row][m_sti->m_last_column - 1];
}

Real SimpleTableInternalMng::
elemRight(bool update_last_pos)
{
  ARCANE_CHECK_PTR(m_sti);
  if (m_sti->m_last_row == -1 || m_sti->m_last_column == -1 || m_sti->m_last_column + 1 >= m_sti->m_values_csv.dim2Size())
    return 0;

  // Par rapport à editElemRight(), si on ne veut pas mettre à jour la dernière position,
  // on ne verifie pas ni modifie m_sti->m_size_columns.
  if (update_last_pos) {
    m_sti->m_last_column++;

    if (m_sti->m_size_rows[m_sti->m_last_row] <= m_sti->m_last_column)
      m_sti->m_size_rows[m_sti->m_last_row] = m_sti->m_last_column + 1;
    if (m_sti->m_size_columns[m_sti->m_last_column] <= m_sti->m_last_row)
      m_sti->m_size_columns[m_sti->m_last_column] = m_sti->m_last_row + 1;
    return m_sti->m_values_csv[m_sti->m_last_row][m_sti->m_last_column];
  }
  return m_sti->m_values_csv[m_sti->m_last_row][m_sti->m_last_column + 1];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool SimpleTableInternalMng::
editElem(Real elem)
{
  ARCANE_CHECK_PTR(m_sti);
  m_sti->m_values_csv[m_sti->m_last_row][m_sti->m_last_column] = elem;
  return true;
}

bool SimpleTableInternalMng::
editElem(Integer pos_x, Integer pos_y, Real elem)
{
  ARCANE_CHECK_PTR(m_sti);
  if (pos_x < 0 || pos_x >= m_sti->m_values_csv.dim2Size() || pos_y < 0 || pos_y >= m_sti->m_values_csv.dim1Size())
    return false;

  if (m_sti->m_size_columns[pos_x] <= pos_y)
    m_sti->m_size_columns[pos_x] = pos_y + 1;
  if (m_sti->m_size_rows[pos_y] <= pos_x)
    m_sti->m_size_rows[pos_y] = pos_x + 1;

  m_sti->m_values_csv[pos_y][pos_x] = elem;

  m_sti->m_last_row = pos_y;
  m_sti->m_last_column = pos_x;

  return true;
}

bool SimpleTableInternalMng::
editElem(const String& name_column, const String& name_row, Real elem)
{
  ARCANE_CHECK_PTR(m_sti);
  std::optional<Integer> pos_x = m_sti->m_name_columns.span().findFirst(name_column);
  std::optional<Integer> pos_y = m_sti->m_name_rows.span().findFirst(name_row);

  if (pos_x && pos_y)
    return editElem(pos_x.value(), pos_y.value(), elem);
  return false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Real SimpleTableInternalMng::
elem()
{
  ARCANE_CHECK_PTR(m_sti);
  return m_sti->m_values_csv[m_sti->m_last_row][m_sti->m_last_column];
}

Real SimpleTableInternalMng::
elem(Integer pos_x, Integer pos_y, bool update_last_pos)
{
  ARCANE_CHECK_PTR(m_sti);
  if (pos_x < 0 || pos_x >= m_sti->m_values_csv.dim2Size() || pos_y < 0 || pos_y >= m_sti->m_values_csv.dim1Size())
    return 0;

  if (update_last_pos) {
    m_sti->m_last_column = pos_x;
    m_sti->m_last_row = pos_y;
  }

  return m_sti->m_values_csv[pos_y][pos_x];
}

Real SimpleTableInternalMng::
elem(const String& name_column, const String& name_row, bool update_last_pos)
{
  ARCANE_CHECK_PTR(m_sti);
  std::optional<Integer> pos_x = m_sti->m_name_columns.span().findFirst(name_column);
  std::optional<Integer> pos_y = m_sti->m_name_rows.span().findFirst(name_row);

  if (pos_x && pos_y)
    return elem(pos_x.value(), pos_y.value(), update_last_pos);
  return 0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

RealUniqueArray SimpleTableInternalMng::
row(Integer pos)
{
  ARCANE_CHECK_PTR(m_sti);
  Integer size = sizeRow(pos);
  RealUniqueArray copie(size);
  for (Integer i = 0; i < size; i++) {
    copie[i] = m_sti->m_values_csv[pos][i];
  }
  return copie;
}

RealUniqueArray SimpleTableInternalMng::
row(const String& name_row)
{
  ARCANE_CHECK_PTR(m_sti);
  std::optional<Integer> pos_y = m_sti->m_name_rows.span().findFirst(name_row);
  if (pos_y)
    return row(pos_y.value());
  return RealUniqueArray(0);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

RealUniqueArray SimpleTableInternalMng::
column(Integer pos)
{
  ARCANE_CHECK_PTR(m_sti);
  Integer size = sizeColumn(pos);

  RealUniqueArray copie(size);
  for (Integer i = 0; i < size; i++) {
    copie[i] = m_sti->m_values_csv[i][pos];
  }
  return copie;
}

RealUniqueArray SimpleTableInternalMng::
column(const String& name_column)
{
  ARCANE_CHECK_PTR(m_sti);
  std::optional<Integer> pos_x = m_sti->m_name_columns.span().findFirst(name_column);
  if (pos_x)
    return column(pos_x.value());
  return RealUniqueArray(0);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer SimpleTableInternalMng::
sizeRow(Integer pos)
{
  ARCANE_CHECK_PTR(m_sti);
  if (pos < 0 || pos >= m_sti->m_values_csv.dim1Size())
    return 0;
  return m_sti->m_size_rows[pos];
}

Integer SimpleTableInternalMng::
sizeRow(const String& name_row)
{
  ARCANE_CHECK_PTR(m_sti);
  std::optional<Integer> pos_y = m_sti->m_name_rows.span().findFirst(name_row);
  if (pos_y)
    return sizeRow(pos_y.value());
  return 0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer SimpleTableInternalMng::
sizeColumn(Integer pos)
{
  ARCANE_CHECK_PTR(m_sti);
  if (pos < 0 || pos >= m_sti->m_values_csv.dim2Size())
    return 0;
  return m_sti->m_size_columns[pos];
}

Integer SimpleTableInternalMng::
sizeColumn(const String& name_column)
{
  ARCANE_CHECK_PTR(m_sti);
  std::optional<Integer> pos_x = m_sti->m_name_columns.span().findFirst(name_column);
  if (pos_x)
    return sizeColumn(pos_x.value());
  return 0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer SimpleTableInternalMng::
posRow(const String& name_row)
{
  ARCANE_CHECK_PTR(m_sti);
  std::optional<Integer> pos_y = m_sti->m_name_rows.span().findFirst(name_row);
  if (pos_y)
    return pos_y.value();
  return -1;
}

Integer SimpleTableInternalMng::
posColumn(const String& name_column)
{
  ARCANE_CHECK_PTR(m_sti);
  std::optional<Integer> pos_x = m_sti->m_name_columns.span().findFirst(name_column);
  if (pos_x)
    return pos_x.value();
  return -1;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer SimpleTableInternalMng::
numRows()
{
  ARCANE_CHECK_PTR(m_sti);
  return m_sti->m_values_csv.dim1Size();
}

Integer SimpleTableInternalMng::
numColumns()
{
  ARCANE_CHECK_PTR(m_sti);
  return m_sti->m_values_csv.dim2Size();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String SimpleTableInternalMng::
nameRow(Integer pos)
{
  ARCANE_CHECK_PTR(m_sti);
  if (pos < 0 || pos >= m_sti->m_values_csv.dim1Size())
    return "";

  return m_sti->m_name_rows[pos];
}

String SimpleTableInternalMng::
nameColumn(Integer pos)
{
  ARCANE_CHECK_PTR(m_sti);
  if (pos < 0 || pos >= m_sti->m_values_csv.dim2Size())
    return "";

  return m_sti->m_name_columns[pos];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool SimpleTableInternalMng::
editNameRow(Integer pos, const String& new_name)
{
  ARCANE_CHECK_PTR(m_sti);
  if (pos < 0 || pos >= m_sti->m_values_csv.dim1Size())
    return false;
  m_sti->m_name_rows[pos] = new_name;
  return true;
}

bool SimpleTableInternalMng::
editNameRow(const String& name_row, const String& new_name)
{
  ARCANE_CHECK_PTR(m_sti);
  std::optional<Integer> pos_y = m_sti->m_name_rows.span().findFirst(name_row);
  if (pos_y)
    return editNameRow(pos_y.value(), new_name);
  return false;
}

bool SimpleTableInternalMng::
editNameColumn(Integer pos, const String& new_name)
{
  ARCANE_CHECK_PTR(m_sti);
  if (pos < 0 || pos >= m_sti->m_values_csv.dim2Size())
    return false;
  m_sti->m_name_columns[pos] = new_name;
  return true;
}

bool SimpleTableInternalMng::
editNameColumn(const String& name_column, const String& new_name)
{
  ARCANE_CHECK_PTR(m_sti);
  std::optional<Integer> pos_x = m_sti->m_name_columns.span().findFirst(name_column);
  if (pos_x)
    return editNameColumn(pos_x.value(), new_name);
  return false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer SimpleTableInternalMng::
addAverageColumn(const String& name_column)
{
  ARCANE_CHECK_PTR(m_sti);
  Integer pos = addColumn(name_column);
  for (Integer i = 0; i < m_sti->m_values_csv.dim1Size(); i++) {
    Real avg = 0.0;
    ConstArrayView<Real> view = m_sti->m_values_csv[i];
    for (Integer j = 0; j < view.size() - 1; j++) {
      avg += view[j];
    }
    avg /= view.size() - 1;
    addElemColumn(pos, avg);
  }
  return pos;
}

SimpleTableInternal* SimpleTableInternalMng::
internal() 
{
  return m_sti;
}

void SimpleTableInternalMng::
setInternal(SimpleTableInternal* sti) 
{
  ARCANE_CHECK_PTR(sti);
  m_sti = sti;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
