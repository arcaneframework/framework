// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SimpleTableInternalMng.cc                                   (C) 2000-2022 */
/*                                                                           */
/* Classe permettant de modifier facilement un SimpleTableInternal.          */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/std/SimpleTableInternalMng.h"

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
  m_simple_table_internal->clear();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer SimpleTableInternalMng::
addRow(const String& row_name)
{
  Integer position = m_simple_table_internal->m_values.dim1Size();
  m_simple_table_internal->m_values.resize(position + 1);

  m_simple_table_internal->m_row_names.add(row_name);
  m_simple_table_internal->m_row_sizes.add(0);

  m_simple_table_internal->m_last_row = position;

  return position;
}

Integer SimpleTableInternalMng::
addRow(const String& row_name, ConstArrayView<Real> elements)
{
  Integer position = m_simple_table_internal->m_values.dim1Size();
  m_simple_table_internal->m_values.resize(position + 1);

  m_simple_table_internal->m_row_names.add(row_name);
  m_simple_table_internal->m_row_sizes.add(0);

  addElementsInRow(position, elements);

  return position;
}

bool SimpleTableInternalMng::
addRows(StringConstArrayView rows_names)
{
  Integer size = rows_names.size();
  if (size == 0)
    return true;

  Integer position = m_simple_table_internal->m_values.dim1Size();
  m_simple_table_internal->m_values.resize(position + size);

  m_simple_table_internal->m_row_names.addRange(rows_names);
  m_simple_table_internal->m_row_sizes.addRange(IntegerUniqueArray(size, 0));

  m_simple_table_internal->m_last_row = position;

  return true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer SimpleTableInternalMng::
addColumn(const String& column_name)
{
  Integer position = m_simple_table_internal->m_values.dim2Size();
  m_simple_table_internal->m_values.resize(m_simple_table_internal->m_values.dim1Size(), position + 1);

  m_simple_table_internal->m_column_names.add(column_name);
  m_simple_table_internal->m_column_sizes.add(0);

  m_simple_table_internal->m_last_column = position;

  return position;
}

Integer SimpleTableInternalMng::
addColumn(const String& column_name, ConstArrayView<Real> elements)
{
  Integer position = m_simple_table_internal->m_values.dim2Size();
  m_simple_table_internal->m_values.resize(m_simple_table_internal->m_values.dim1Size(), position + 1);

  m_simple_table_internal->m_column_names.add(column_name);
  m_simple_table_internal->m_column_sizes.add(0);

  addElementsInColumn(position, elements);

  return position;
}

bool SimpleTableInternalMng::
addColumns(StringConstArrayView columns_names)
{
  Integer size = columns_names.size();
  if (size == 0)
    return true;

  Integer position = m_simple_table_internal->m_values.dim2Size();
  m_simple_table_internal->m_values.resize(m_simple_table_internal->m_values.dim1Size(), position + size);

  m_simple_table_internal->m_column_names.addRange(columns_names);
  m_simple_table_internal->m_column_sizes.addRange(IntegerUniqueArray(size, 0));

  m_simple_table_internal->m_last_column = position;

  return true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool SimpleTableInternalMng::
addElementInRow(Integer position, Real element)
{
  if (position < 0 || position >= m_simple_table_internal->m_values.dim1Size())
    return false;

  ArrayView<Real> view = m_simple_table_internal->m_values[position];
  Integer size_row = m_simple_table_internal->m_row_sizes[position];

  if (m_simple_table_internal->m_values.dim2Size() < size_row + 1)
    return false;

  view[size_row] = element;

  m_simple_table_internal->m_last_row = position;
  m_simple_table_internal->m_last_column = size_row;

  m_simple_table_internal->m_row_sizes[position]++;
  // Il peut y avoir des élements sur la ligne d'après à la même colonne.
  // Exemple : addElementInRow(position=L01, element=NEW):
  // aaa|C00|C01|C02
  // L00|123|456|789
  // L01|147|NEW|
  // L02|159|753|852
  // Il y a 753 donc la taille de la colonne reste égale à 3.
  m_simple_table_internal->m_column_sizes[size_row] = std::max(position + 1, m_simple_table_internal->m_column_sizes[size_row]);

  return true;
}

bool SimpleTableInternalMng::
addElementInRow(const String& row_name, Real element, bool create_if_not_exist)
{
  std::optional<Integer> position = m_simple_table_internal->m_row_names.span().findFirst(row_name);

  if (position)
    return addElementInRow(position.value(), element);
  else if (create_if_not_exist)
    return addElementInRow(addRow(row_name), element);
  else
    return false;
}

bool SimpleTableInternalMng::
addElementInSameRow(Real element)
{
  if (m_simple_table_internal->m_last_row == -1 || m_simple_table_internal->m_last_column == -1)
    return false;
  return addElementInRow(m_simple_table_internal->m_last_row, element);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool SimpleTableInternalMng::
addElementsInRow(Integer position, ConstArrayView<Real> elements)
{
  if (position < 0 || position >= m_simple_table_internal->m_values.dim1Size())
    return false;

  ArrayView<Real> view = m_simple_table_internal->m_values[position];
  Integer size_row = m_simple_table_internal->m_row_sizes[position];
  Integer min_size = (elements.size() <= m_simple_table_internal->m_values.dim2Size() - size_row
                      ? elements.size()
                      : m_simple_table_internal->m_values.dim2Size() - size_row);

  for (Integer i = 0; i < min_size; i++) {
    view[i + size_row] = elements[i];
    m_simple_table_internal->m_column_sizes[i + size_row] = std::max(position + 1, m_simple_table_internal->m_column_sizes[i + size_row]);
  }
  m_simple_table_internal->m_row_sizes[position] += min_size;

  m_simple_table_internal->m_last_row = position;
  m_simple_table_internal->m_last_column = m_simple_table_internal->m_row_sizes[position] - 1;

  return elements.size() <= m_simple_table_internal->m_values.dim2Size() - size_row;
}

bool SimpleTableInternalMng::
addElementsInRow(const String& row_name, ConstArrayView<Real> elements, bool create_if_not_exist)
{
  std::optional<Integer> position = m_simple_table_internal->m_row_names.span().findFirst(row_name);

  if (position)
    return addElementsInRow(position.value(), elements);
  // Permet d'avoir un return bool (sinon on pourrait simplement faire addRow(row_name, elements)).
  else if (create_if_not_exist)
    return addElementsInRow(addRow(row_name), elements);
  else
    return false;
}

bool SimpleTableInternalMng::
addElementsInSameRow(ConstArrayView<Real> elements)
{
  if (m_simple_table_internal->m_last_row == -1 || m_simple_table_internal->m_last_column == -1)
    return false;
  return addElementsInRow(m_simple_table_internal->m_last_row, elements);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool SimpleTableInternalMng::
addElementInColumn(Integer position, Real element)
{
  if (position < 0 || position >= m_simple_table_internal->m_values.dim2Size())
    return false;

  Integer size_column = m_simple_table_internal->m_column_sizes[position];

  if (m_simple_table_internal->m_values.dim1Size() < size_column + 1)
    return false;

  m_simple_table_internal->m_values[size_column][position] = element;

  m_simple_table_internal->m_last_column = position;
  m_simple_table_internal->m_last_row = size_column;

  m_simple_table_internal->m_column_sizes[position]++;
  m_simple_table_internal->m_row_sizes[size_column] = std::max(position + 1, m_simple_table_internal->m_row_sizes[size_column]);

  return true;
}

bool SimpleTableInternalMng::
addElementInColumn(const String& column_name, Real element, bool create_if_not_exist)
{
  std::optional<Integer> position = m_simple_table_internal->m_column_names.span().findFirst(column_name);

  if (position)
    return addElementInColumn(position.value(), element);
  else if (create_if_not_exist)
    return addElementInColumn(addColumn(column_name), element);
  else
    return false;
}

bool SimpleTableInternalMng::
addElementInSameColumn(Real element)
{
  if (m_simple_table_internal->m_last_row == -1 || m_simple_table_internal->m_last_column == -1)
    return false;
  return addElementInColumn(m_simple_table_internal->m_last_column, element);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool SimpleTableInternalMng::
addElementsInColumn(Integer position, ConstArrayView<Real> elements)
{
  if (position < 0 || position >= m_simple_table_internal->m_values.dim2Size())
    return false;

  Integer size_column = m_simple_table_internal->m_column_sizes[position];
  Integer min_size = (elements.size() <= m_simple_table_internal->m_values.dim1Size() - size_column
                      ? elements.size()
                      : m_simple_table_internal->m_values.dim1Size() - size_column);

  for (Integer i = 0; i < min_size; i++) {
    m_simple_table_internal->m_values[i + size_column][position] = elements[i];
    m_simple_table_internal->m_row_sizes[i + size_column] = std::max(position + 1, m_simple_table_internal->m_row_sizes[i + size_column]);
  }
  m_simple_table_internal->m_column_sizes[position] += min_size;

  m_simple_table_internal->m_last_column = position;
  m_simple_table_internal->m_last_row = m_simple_table_internal->m_column_sizes[position] - 1;

  return elements.size() <= m_simple_table_internal->m_values.dim1Size() - size_column;
}

bool SimpleTableInternalMng::
addElementsInColumn(const String& column_name, ConstArrayView<Real> elements, bool create_if_not_exist)
{
  std::optional<Integer> position = m_simple_table_internal->m_column_names.span().findFirst(column_name);

  if (position)
    return addElementsInColumn(position.value(), elements);
  // Permet d'avoir un return bool (sinon on pourrait simplement faire addColumn(column_name, elements)).
  else if (create_if_not_exist)
    return addElementsInColumn(addColumn(column_name), elements);
  else
    return false;
}

bool SimpleTableInternalMng::
addElementsInSameColumn(ConstArrayView<Real> elements)
{
  if (m_simple_table_internal->m_last_row == -1 || m_simple_table_internal->m_last_column == -1)
    return false;
  return addElementsInColumn(m_simple_table_internal->m_last_column, elements);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool SimpleTableInternalMng::
editElementUp(Real element, bool update_last_position)
{
  if (m_simple_table_internal->m_last_row == -1 || m_simple_table_internal->m_last_column == -1 || m_simple_table_internal->m_last_row - 1 < 0)
    return false;
  m_simple_table_internal->m_last_row--;

  // Pas besoin d'ajuster la taille de la colonne car on est sûr que m_simple_table_internal->m_column_sizes[m_simple_table_internal->m_last_column] >= m_simple_table_internal->m_last_row.
  if (m_simple_table_internal->m_row_sizes[m_simple_table_internal->m_last_row] <= m_simple_table_internal->m_last_column)
    m_simple_table_internal->m_row_sizes[m_simple_table_internal->m_last_row] = m_simple_table_internal->m_last_column + 1;

  m_simple_table_internal->m_values[m_simple_table_internal->m_last_row][m_simple_table_internal->m_last_column] = element;
  if (!update_last_position)
    m_simple_table_internal->m_last_row++;
  return true;
}

bool SimpleTableInternalMng::
editElementDown(Real element, bool update_last_position)
{
  if (m_simple_table_internal->m_last_row == -1 || m_simple_table_internal->m_last_column == -1 || m_simple_table_internal->m_last_row + 1 >= m_simple_table_internal->m_values.dim1Size())
    return false;
  m_simple_table_internal->m_last_row++;

  if (m_simple_table_internal->m_row_sizes[m_simple_table_internal->m_last_row] <= m_simple_table_internal->m_last_column)
    m_simple_table_internal->m_row_sizes[m_simple_table_internal->m_last_row] = m_simple_table_internal->m_last_column + 1;
  if (m_simple_table_internal->m_column_sizes[m_simple_table_internal->m_last_column] <= m_simple_table_internal->m_last_row)
    m_simple_table_internal->m_column_sizes[m_simple_table_internal->m_last_column] = m_simple_table_internal->m_last_row + 1;

  m_simple_table_internal->m_values[m_simple_table_internal->m_last_row][m_simple_table_internal->m_last_column] = element;
  if (!update_last_position)
    m_simple_table_internal->m_last_row--;
  return true;
}

bool SimpleTableInternalMng::
editElementLeft(Real element, bool update_last_position)
{
  if (m_simple_table_internal->m_last_row == -1 || m_simple_table_internal->m_last_column == -1 || m_simple_table_internal->m_last_column - 1 < 0)
    return false;
  m_simple_table_internal->m_last_column--;

  // Pas besoin d'ajuster la taille de la ligne car on est sûr que m_simple_table_internal->m_row_sizes[m_simple_table_internal->m_last_row] >= m_simple_table_internal->m_last_column.
  if (m_simple_table_internal->m_column_sizes[m_simple_table_internal->m_last_column] <= m_simple_table_internal->m_last_row)
    m_simple_table_internal->m_column_sizes[m_simple_table_internal->m_last_column] = m_simple_table_internal->m_last_row + 1;

  m_simple_table_internal->m_values[m_simple_table_internal->m_last_row][m_simple_table_internal->m_last_column] = element;
  if (!update_last_position)
    m_simple_table_internal->m_last_column++;
  return true;
}

bool SimpleTableInternalMng::
editElementRight(Real element, bool update_last_position)
{
  if (m_simple_table_internal->m_last_row == -1 || m_simple_table_internal->m_last_column == -1 || m_simple_table_internal->m_last_column + 1 >= m_simple_table_internal->m_values.dim2Size())
    return false;
  m_simple_table_internal->m_last_column++;

  if (m_simple_table_internal->m_row_sizes[m_simple_table_internal->m_last_row] <= m_simple_table_internal->m_last_column)
    m_simple_table_internal->m_row_sizes[m_simple_table_internal->m_last_row] = m_simple_table_internal->m_last_column + 1;
  if (m_simple_table_internal->m_column_sizes[m_simple_table_internal->m_last_column] <= m_simple_table_internal->m_last_row)
    m_simple_table_internal->m_column_sizes[m_simple_table_internal->m_last_column] = m_simple_table_internal->m_last_row + 1;

  m_simple_table_internal->m_values[m_simple_table_internal->m_last_row][m_simple_table_internal->m_last_column] = element;
  if (!update_last_position)
    m_simple_table_internal->m_last_column--;
  return true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Real SimpleTableInternalMng::
elementUp(bool update_last_position)
{
  if (m_simple_table_internal->m_last_row == -1 || m_simple_table_internal->m_last_column == -1 || m_simple_table_internal->m_last_row - 1 < 0)
    return 0;

  // Par rapport à editElementUp(), si on ne veut pas mettre à jour la dernière position,
  // on ne verifie pas ni modifie m_simple_table_internal->m_row_sizes.
  if (update_last_position) {
    m_simple_table_internal->m_last_row--;
    // Pas besoin d'ajuster la taille de la colonne car on est sûr que m_simple_table_internal->m_column_sizes[m_simple_table_internal->m_last_column] >= m_simple_table_internal->m_last_row.
    if (m_simple_table_internal->m_row_sizes[m_simple_table_internal->m_last_row] <= m_simple_table_internal->m_last_column)
      m_simple_table_internal->m_row_sizes[m_simple_table_internal->m_last_row] = m_simple_table_internal->m_last_column + 1;
    return m_simple_table_internal->m_values[m_simple_table_internal->m_last_row][m_simple_table_internal->m_last_column];
  }

  return m_simple_table_internal->m_values[m_simple_table_internal->m_last_row - 1][m_simple_table_internal->m_last_column];
}

Real SimpleTableInternalMng::
elementDown(bool update_last_position)
{
  if (m_simple_table_internal->m_last_row == -1 || m_simple_table_internal->m_last_column == -1 || m_simple_table_internal->m_last_row + 1 >= m_simple_table_internal->m_values.dim1Size())
    return 0;

  // Par rapport à editElementDown(), si on ne veut pas mettre à jour la dernière position,
  // on ne verifie pas ni modifie m_simple_table_internal->m_row_sizes.
  if (update_last_position) {
    m_simple_table_internal->m_last_row++;

    if (m_simple_table_internal->m_row_sizes[m_simple_table_internal->m_last_row] <= m_simple_table_internal->m_last_column)
      m_simple_table_internal->m_row_sizes[m_simple_table_internal->m_last_row] = m_simple_table_internal->m_last_column + 1;
    if (m_simple_table_internal->m_column_sizes[m_simple_table_internal->m_last_column] <= m_simple_table_internal->m_last_row)
      m_simple_table_internal->m_column_sizes[m_simple_table_internal->m_last_column] = m_simple_table_internal->m_last_row + 1;
    return m_simple_table_internal->m_values[m_simple_table_internal->m_last_row][m_simple_table_internal->m_last_column];
  }
  return m_simple_table_internal->m_values[m_simple_table_internal->m_last_row + 1][m_simple_table_internal->m_last_column];
}

Real SimpleTableInternalMng::
elementLeft(bool update_last_position)
{
  if (m_simple_table_internal->m_last_row == -1 || m_simple_table_internal->m_last_column == -1 || m_simple_table_internal->m_last_column - 1 < 0)
    return 0;

  // Par rapport à editElementLeft(), si on ne veut pas mettre à jour la dernière position,
  // on ne verifie pas ni modifie m_simple_table_internal->m_column_sizes.
  if (update_last_position) {
    m_simple_table_internal->m_last_column--;

    // Pas besoin d'ajuster la taille de la ligne car on est sûr que m_simple_table_internal->m_row_sizes[m_simple_table_internal->m_last_row] >= m_simple_table_internal->m_last_column.
    if (m_simple_table_internal->m_column_sizes[m_simple_table_internal->m_last_column] <= m_simple_table_internal->m_last_row)
      m_simple_table_internal->m_column_sizes[m_simple_table_internal->m_last_column] = m_simple_table_internal->m_last_row + 1;
    return m_simple_table_internal->m_values[m_simple_table_internal->m_last_row][m_simple_table_internal->m_last_column];
  }
  return m_simple_table_internal->m_values[m_simple_table_internal->m_last_row][m_simple_table_internal->m_last_column - 1];
}

Real SimpleTableInternalMng::
elementRight(bool update_last_position)
{
  if (m_simple_table_internal->m_last_row == -1 || m_simple_table_internal->m_last_column == -1 || m_simple_table_internal->m_last_column + 1 >= m_simple_table_internal->m_values.dim2Size())
    return 0;

  // Par rapport à editElementRight(), si on ne veut pas mettre à jour la dernière position,
  // on ne verifie pas ni modifie m_simple_table_internal->m_column_sizes.
  if (update_last_position) {
    m_simple_table_internal->m_last_column++;

    if (m_simple_table_internal->m_row_sizes[m_simple_table_internal->m_last_row] <= m_simple_table_internal->m_last_column)
      m_simple_table_internal->m_row_sizes[m_simple_table_internal->m_last_row] = m_simple_table_internal->m_last_column + 1;
    if (m_simple_table_internal->m_column_sizes[m_simple_table_internal->m_last_column] <= m_simple_table_internal->m_last_row)
      m_simple_table_internal->m_column_sizes[m_simple_table_internal->m_last_column] = m_simple_table_internal->m_last_row + 1;
    return m_simple_table_internal->m_values[m_simple_table_internal->m_last_row][m_simple_table_internal->m_last_column];
  }
  return m_simple_table_internal->m_values[m_simple_table_internal->m_last_row][m_simple_table_internal->m_last_column + 1];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool SimpleTableInternalMng::
editElement(Real element)
{
  m_simple_table_internal->m_values[m_simple_table_internal->m_last_row][m_simple_table_internal->m_last_column] = element;
  return true;
}

bool SimpleTableInternalMng::
editElement(Integer position_x, Integer position_y, Real element)
{
  if (position_x < 0 || position_x >= m_simple_table_internal->m_values.dim2Size() || position_y < 0 || position_y >= m_simple_table_internal->m_values.dim1Size())
    return false;

  if (m_simple_table_internal->m_column_sizes[position_x] <= position_y)
    m_simple_table_internal->m_column_sizes[position_x] = position_y + 1;
  if (m_simple_table_internal->m_row_sizes[position_y] <= position_x)
    m_simple_table_internal->m_row_sizes[position_y] = position_x + 1;

  m_simple_table_internal->m_values[position_y][position_x] = element;

  m_simple_table_internal->m_last_row = position_y;
  m_simple_table_internal->m_last_column = position_x;

  return true;
}

bool SimpleTableInternalMng::
editElement(const String& column_name, const String& row_name, Real element)
{
  std::optional<Integer> position_x = m_simple_table_internal->m_column_names.span().findFirst(column_name);
  std::optional<Integer> position_y = m_simple_table_internal->m_row_names.span().findFirst(row_name);

  if (position_x && position_y)
    return editElement(position_x.value(), position_y.value(), element);
  return false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Real SimpleTableInternalMng::
element()
{
  return m_simple_table_internal->m_values[m_simple_table_internal->m_last_row][m_simple_table_internal->m_last_column];
}

Real SimpleTableInternalMng::
element(Integer position_x, Integer position_y, bool update_last_position)
{
  if (position_x < 0 || position_x >= m_simple_table_internal->m_values.dim2Size() || position_y < 0 || position_y >= m_simple_table_internal->m_values.dim1Size())
    return 0;

  if (update_last_position) {
    m_simple_table_internal->m_last_column = position_x;
    m_simple_table_internal->m_last_row = position_y;
  }

  return m_simple_table_internal->m_values[position_y][position_x];
}

Real SimpleTableInternalMng::
element(const String& column_name, const String& row_name, bool update_last_position)
{
  std::optional<Integer> position_x = m_simple_table_internal->m_column_names.span().findFirst(column_name);
  std::optional<Integer> position_y = m_simple_table_internal->m_row_names.span().findFirst(row_name);

  if (position_x && position_y)
    return element(position_x.value(), position_y.value(), update_last_position);
  return 0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

RealUniqueArray SimpleTableInternalMng::
row(Integer position)
{
  Integer size = rowSize(position);
  RealUniqueArray copie(size);
  for (Integer i = 0; i < size; i++) {
    copie[i] = m_simple_table_internal->m_values[position][i];
  }
  return copie;
}

RealUniqueArray SimpleTableInternalMng::
row(const String& row_name)
{
  std::optional<Integer> position_y = m_simple_table_internal->m_row_names.span().findFirst(row_name);
  if (position_y)
    return row(position_y.value());
  return RealUniqueArray(0);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

RealUniqueArray SimpleTableInternalMng::
column(Integer position)
{
  Integer size = columnSize(position);

  RealUniqueArray copie(size);
  for (Integer i = 0; i < size; i++) {
    copie[i] = m_simple_table_internal->m_values[i][position];
  }
  return copie;
}

RealUniqueArray SimpleTableInternalMng::
column(const String& column_name)
{
  std::optional<Integer> position_x = m_simple_table_internal->m_column_names.span().findFirst(column_name);
  if (position_x)
    return column(position_x.value());
  return RealUniqueArray(0);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer SimpleTableInternalMng::
rowSize(Integer position)
{
  if (position < 0 || position >= m_simple_table_internal->m_values.dim1Size())
    return 0;
  return m_simple_table_internal->m_row_sizes[position];
}

Integer SimpleTableInternalMng::
rowSize(const String& row_name)
{
  std::optional<Integer> position_y = m_simple_table_internal->m_row_names.span().findFirst(row_name);
  if (position_y)
    return rowSize(position_y.value());
  return 0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer SimpleTableInternalMng::
columnSize(Integer position)
{
  if (position < 0 || position >= m_simple_table_internal->m_values.dim2Size())
    return 0;
  return m_simple_table_internal->m_column_sizes[position];
}

Integer SimpleTableInternalMng::
columnSize(const String& column_name)
{
  std::optional<Integer> position_x = m_simple_table_internal->m_column_names.span().findFirst(column_name);
  if (position_x)
    return columnSize(position_x.value());
  return 0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer SimpleTableInternalMng::
rowPosition(const String& row_name)
{
  std::optional<Integer> position_y = m_simple_table_internal->m_row_names.span().findFirst(row_name);
  if (position_y)
    return position_y.value();
  return -1;
}

Integer SimpleTableInternalMng::
columnPosition(const String& column_name)
{
  std::optional<Integer> position_x = m_simple_table_internal->m_column_names.span().findFirst(column_name);
  if (position_x)
    return position_x.value();
  return -1;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer SimpleTableInternalMng::
numberOfRows()
{
  return m_simple_table_internal->m_values.dim1Size();
}

Integer SimpleTableInternalMng::
numberOfColumns()
{
  return m_simple_table_internal->m_values.dim2Size();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String SimpleTableInternalMng::
rowName(Integer position)
{
  if (position < 0 || position >= m_simple_table_internal->m_values.dim1Size())
    return "";

  return m_simple_table_internal->m_row_names[position];
}

String SimpleTableInternalMng::
columnName(Integer position)
{
  if (position < 0 || position >= m_simple_table_internal->m_values.dim2Size())
    return "";

  return m_simple_table_internal->m_column_names[position];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool SimpleTableInternalMng::
editRowName(Integer position, const String& new_name)
{
  if (position < 0 || position >= m_simple_table_internal->m_values.dim1Size())
    return false;
  m_simple_table_internal->m_row_names[position] = new_name;
  return true;
}

bool SimpleTableInternalMng::
editRowName(const String& row_name, const String& new_name)
{
  std::optional<Integer> position_y = m_simple_table_internal->m_row_names.span().findFirst(row_name);
  if (position_y)
    return editRowName(position_y.value(), new_name);
  return false;
}

bool SimpleTableInternalMng::
editColumnName(Integer position, const String& new_name)
{
  if (position < 0 || position >= m_simple_table_internal->m_values.dim2Size())
    return false;
  m_simple_table_internal->m_column_names[position] = new_name;
  return true;
}

bool SimpleTableInternalMng::
editColumnName(const String& column_name, const String& new_name)
{
  std::optional<Integer> position_x = m_simple_table_internal->m_column_names.span().findFirst(column_name);
  if (position_x)
    return editColumnName(position_x.value(), new_name);
  return false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer SimpleTableInternalMng::
addAverageColumn(const String& column_name)
{
  Integer position = addColumn(column_name);
  for (Integer i = 0; i < m_simple_table_internal->m_values.dim1Size(); i++) {
    Real avg = 0.0;
    ConstArrayView<Real> view = m_simple_table_internal->m_values[i];
    for (Integer j = 0; j < view.size() - 1; j++) {
      avg += view[j];
    }
    avg /= view.size() - 1;
    addElementInColumn(position, avg);
  }
  return position;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<SimpleTableInternal> SimpleTableInternalMng::
internal()
{
  return m_simple_table_internal;
}

void SimpleTableInternalMng::
setInternal(const Ref<SimpleTableInternal>& simple_table_internal)
{
  if (simple_table_internal.isNull())
    ARCANE_FATAL("La réference passée en paramètre est Null.");
  m_simple_table_internal = simple_table_internal;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
