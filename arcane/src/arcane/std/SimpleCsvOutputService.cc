// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SimpleCsvOutputService.cc                                   (C) 2000-2022 */
/*                                                                           */
/* Service permettant de construire et de sortir un tableau au formet csv.   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/std/SimpleCsvOutputService.h"

#include <arcane/Directory.h>
#include <arcane/IMesh.h>
#include <arcane/IParallelMng.h>

#include <optional>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SimpleCsvOutputService::
init()
{
  if (m_with_option && options()->getTableName() != "") {
    init(options()->getTableName());
  }
  else {
    init("Table_@proc_id@");
  }
}

void SimpleCsvOutputService::
init(String name_table)
{
  m_name_tab = _computeAt(name_table, m_name_tab_only_P0);
  m_name_tab_computed = true;

  m_separator = ";";

  if (m_with_option && options()->getTableDir() != "") {
    m_path = _computeAt(options()->getTableDir(), m_path_only_P0);
  }
  else {
    m_path = _computeAt("./csv/", m_path_only_P0);
  }
  m_path_computed = true;

  m_precision_print = 6;
  m_is_fixed_print = true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SimpleCsvOutputService::
clear()
{
  m_values_csv.clear();

  m_name_rows.clear();
  m_name_columns.clear();

  m_size_rows.clear();
  m_size_columns.clear();

  m_last_row = -1;
  m_last_column = -1;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer SimpleCsvOutputService::
addRow(String name_row)
{
  Integer pos = m_values_csv.dim1Size();
  m_values_csv.resize(pos + 1);

  m_name_rows.add(name_row);
  m_size_rows.add(0);

  m_last_row = pos;

  return pos;
}

Integer SimpleCsvOutputService::
addRow(String name_row, ConstArrayView<Real> elems)
{
  Integer pos = m_values_csv.dim1Size();
  m_values_csv.resize(pos + 1);

  m_name_rows.add(name_row);
  m_size_rows.add(0);

  addElemsRow(pos, elems);

  return pos;
}

bool SimpleCsvOutputService::
addRows(StringConstArrayView name_rows)
{
  Integer size = name_rows.size();
  if (size == 0)
    return true;

  Integer pos = m_values_csv.dim1Size();
  m_values_csv.resize(pos + size);

  m_name_rows.addRange(name_rows);
  m_size_rows.addRange(IntegerUniqueArray(size, 0));

  m_last_row = pos;

  return true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer SimpleCsvOutputService::
addColumn(String name_column)
{
  Integer pos = m_values_csv.dim2Size();
  m_values_csv.resize(m_values_csv.dim1Size(), pos + 1);

  m_name_columns.add(name_column);
  m_size_columns.add(0);

  m_last_column = pos;

  return pos;
}

Integer SimpleCsvOutputService::
addColumn(String name_column, ConstArrayView<Real> elems)
{
  Integer pos = m_values_csv.dim2Size();
  m_values_csv.resize(m_values_csv.dim1Size(), pos + 1);

  m_name_columns.add(name_column);
  m_size_columns.add(0);

  addElemsColumn(pos, elems);

  return pos;
}

bool SimpleCsvOutputService::
addColumns(StringConstArrayView name_columns)
{
  Integer size = name_columns.size();
  if (size == 0)
    return true;

  Integer pos = m_values_csv.dim2Size();
  m_values_csv.resize(m_values_csv.dim1Size(), pos + size);

  m_name_columns.addRange(name_columns);
  m_size_columns.addRange(IntegerUniqueArray(size, 0));

  m_last_column = pos;

  return true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool SimpleCsvOutputService::
addElemRow(Integer pos, Real elem)
{
  if (pos < 0 || pos >= m_values_csv.dim1Size())
    return false;

  ArrayView<Real> view = m_values_csv[pos];
  Integer size_row = m_size_rows[pos];

  if (m_values_csv.dim2Size() < size_row + 1)
    return false;

  view[size_row] = elem;

  m_last_row = pos;
  m_last_column = size_row;

  m_size_rows[pos]++;
  // Il peut y avoir des élements sur la ligne d'après à la même colonne.
  // Exemple : addElemRow(pos=L01, elem=NEW):
  // aaa|C00|C01|C02
  // L00|123|456|789
  // L01|147|NEW|
  // L02|159|753|852
  // Il y a 753 donc la taille de la colonne reste égale à 3.
  m_size_columns[size_row] = std::max(pos + 1, m_size_columns[size_row]);

  return true;
}

bool SimpleCsvOutputService::
addElemRow(String name_row, Real elem, bool create_if_not_exist)
{
  std::optional<Integer> pos = m_name_rows.span().findFirst(name_row);

  if (pos)
    return addElemRow(pos.value(), elem);
  else if (create_if_not_exist)
    return addElemRow(addRow(name_row), elem);
  else
    return false;
}

bool SimpleCsvOutputService::
addElemSameRow(Real elem)
{
  if (m_last_row == -1 || m_last_column == -1)
    return false;
  return addElemRow(m_last_row, elem);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool SimpleCsvOutputService::
addElemsRow(Integer pos, ConstArrayView<Real> elems)
{
  if (pos < 0 || pos >= m_values_csv.dim1Size())
    return false;

  ArrayView<Real> view = m_values_csv[pos];
  Integer size_row = m_size_rows[pos];
  Integer min_size = (elems.size() <= m_values_csv.dim2Size() - size_row
                      ? elems.size()
                      : m_values_csv.dim2Size() - size_row);

  for (Integer i = 0; i < min_size; i++) {
    view[i + size_row] = elems[i];
    m_size_columns[i + size_row] = std::max(pos + 1, m_size_columns[i + size_row]);
  }
  m_size_rows[pos] += min_size;

  m_last_row = pos;
  m_last_column = m_size_rows[pos] - 1;

  return elems.size() <= m_values_csv.dim2Size() - size_row;
}

bool SimpleCsvOutputService::
addElemsRow(String name_row, ConstArrayView<Real> elems, bool create_if_not_exist)
{
  std::optional<Integer> pos = m_name_rows.span().findFirst(name_row);

  if (pos)
    return addElemsRow(pos.value(), elems);
  // Permet d'avoir un return bool (sinon on pourrait simplement faire addRow(name_row, elems)).
  else if (create_if_not_exist)
    return addElemsRow(addRow(name_row), elems);
  else
    return false;
}

bool SimpleCsvOutputService::
addElemsSameRow(ConstArrayView<Real> elems)
{
  if (m_last_row == -1 || m_last_column == -1)
    return false;
  return addElemsRow(m_last_row, elems);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool SimpleCsvOutputService::
addElemColumn(Integer pos, Real elem)
{
  if (pos < 0 || pos >= m_values_csv.dim2Size())
    return false;

  Integer size_column = m_size_columns[pos];

  if (m_values_csv.dim1Size() < size_column + 1)
    return false;

  m_values_csv[size_column][pos] = elem;

  m_last_column = pos;
  m_last_row = size_column;

  m_size_columns[pos]++;
  m_size_rows[size_column] = std::max(pos + 1, m_size_rows[size_column]);

  return true;
}

bool SimpleCsvOutputService::
addElemColumn(String name_column, Real elem, bool create_if_not_exist)
{
  std::optional<Integer> pos = m_name_columns.span().findFirst(name_column);

  if (pos)
    return addElemColumn(pos.value(), elem);
  else if (create_if_not_exist)
    return addElemColumn(addColumn(name_column), elem);
  else
    return false;
}

bool SimpleCsvOutputService::
addElemSameColumn(Real elem)
{
  if (m_last_row == -1 || m_last_column == -1)
    return false;
  return addElemColumn(m_last_column, elem);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool SimpleCsvOutputService::
addElemsColumn(Integer pos, ConstArrayView<Real> elems)
{
  if (pos < 0 || pos >= m_values_csv.dim2Size())
    return false;

  Integer size_column = m_size_columns[pos];
  Integer min_size = (elems.size() <= m_values_csv.dim1Size() - size_column
                      ? elems.size()
                      : m_values_csv.dim1Size() - size_column);

  for (Integer i = 0; i < min_size; i++) {
    m_values_csv[i + size_column][pos] = elems[i];
    m_size_rows[i + size_column] = std::max(pos + 1, m_size_rows[i + size_column]);
  }
  m_size_columns[pos] += min_size;

  m_last_column = pos;
  m_last_row = m_size_columns[pos] - 1;

  return elems.size() <= m_values_csv.dim1Size() - size_column;
}

bool SimpleCsvOutputService::
addElemsColumn(String name_column, ConstArrayView<Real> elems, bool create_if_not_exist)
{
  std::optional<Integer> pos = m_name_columns.span().findFirst(name_column);

  if (pos)
    return addElemsColumn(pos.value(), elems);
  // Permet d'avoir un return bool (sinon on pourrait simplement faire addColumn(name_column, elems)).
  else if (create_if_not_exist)
    return addElemsColumn(addColumn(name_column), elems);
  else
    return false;
}

bool SimpleCsvOutputService::
addElemsSameColumn(ConstArrayView<Real> elems)
{
  if (m_last_row == -1 || m_last_column == -1)
    return false;
  return addElemsColumn(m_last_column, elems);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool SimpleCsvOutputService::
editElemUp(Real elem, bool update_last_pos)
{
  if (m_last_row == -1 || m_last_column == -1 || m_last_row - 1 < 0)
    return false;
  m_last_row--;

  // Pas besoin d'ajuster la taille de la colonne car on est sûr que m_size_columns[m_last_column] >= m_last_row.
  if (m_size_rows[m_last_row] <= m_last_column)
    m_size_rows[m_last_row] = m_last_column + 1;

  m_values_csv[m_last_row][m_last_column] = elem;
  if (!update_last_pos)
    m_last_row++;
  return true;
}

bool SimpleCsvOutputService::
editElemDown(Real elem, bool update_last_pos)
{
  if (m_last_row == -1 || m_last_column == -1 || m_last_row + 1 >= m_values_csv.dim1Size())
    return false;
  m_last_row++;

  if (m_size_rows[m_last_row] <= m_last_column)
    m_size_rows[m_last_row] = m_last_column + 1;
  if (m_size_columns[m_last_column] <= m_last_row)
    m_size_columns[m_last_column] = m_last_row + 1;

  m_values_csv[m_last_row][m_last_column] = elem;
  if (!update_last_pos)
    m_last_row--;
  return true;
}

bool SimpleCsvOutputService::
editElemLeft(Real elem, bool update_last_pos)
{
  if (m_last_row == -1 || m_last_column == -1 || m_last_column - 1 < 0)
    return false;
  m_last_column--;

  // Pas besoin d'ajuster la taille de la ligne car on est sûr que m_size_rows[m_last_row] >= m_last_column.
  if (m_size_columns[m_last_column] <= m_last_row)
    m_size_columns[m_last_column] = m_last_row + 1;

  m_values_csv[m_last_row][m_last_column] = elem;
  if (!update_last_pos)
    m_last_column++;
  return true;
}

bool SimpleCsvOutputService::
editElemRight(Real elem, bool update_last_pos)
{
  if (m_last_row == -1 || m_last_column == -1 || m_last_column + 1 >= m_values_csv.dim2Size())
    return false;
  m_last_column++;

  if (m_size_rows[m_last_row] <= m_last_column)
    m_size_rows[m_last_row] = m_last_column + 1;
  if (m_size_columns[m_last_column] <= m_last_row)
    m_size_columns[m_last_column] = m_last_row + 1;

  m_values_csv[m_last_row][m_last_column] = elem;
  if (!update_last_pos)
    m_last_column--;
  return true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Real SimpleCsvOutputService::
elemUp(bool update_last_pos)
{
  if (m_last_row == -1 || m_last_column == -1 || m_last_row - 1 < 0)
    return 0;

  // Par rapport à editElemUp(), si on ne veut pas mettre à jour la dernière position,
  // on ne verifie pas ni modifie m_size_rows.
  if (update_last_pos) {
    m_last_row--;
    // Pas besoin d'ajuster la taille de la colonne car on est sûr que m_size_columns[m_last_column] >= m_last_row.
    if (m_size_rows[m_last_row] <= m_last_column)
      m_size_rows[m_last_row] = m_last_column + 1;
    return m_values_csv[m_last_row][m_last_column];
  }

  return m_values_csv[m_last_row - 1][m_last_column];
}

Real SimpleCsvOutputService::
elemDown(bool update_last_pos)
{
  if (m_last_row == -1 || m_last_column == -1 || m_last_row + 1 >= m_values_csv.dim1Size())
    return 0;

  // Par rapport à editElemDown(), si on ne veut pas mettre à jour la dernière position,
  // on ne verifie pas ni modifie m_size_rows.
  if (update_last_pos) {
    m_last_row++;

    if (m_size_rows[m_last_row] <= m_last_column)
      m_size_rows[m_last_row] = m_last_column + 1;
    if (m_size_columns[m_last_column] <= m_last_row)
      m_size_columns[m_last_column] = m_last_row + 1;
    return m_values_csv[m_last_row][m_last_column];
  }
  return m_values_csv[m_last_row + 1][m_last_column];
}

Real SimpleCsvOutputService::
elemLeft(bool update_last_pos)
{
  if (m_last_row == -1 || m_last_column == -1 || m_last_column - 1 < 0)
    return 0;

  // Par rapport à editElemLeft(), si on ne veut pas mettre à jour la dernière position,
  // on ne verifie pas ni modifie m_size_columns.
  if (update_last_pos) {
    m_last_column--;

    // Pas besoin d'ajuster la taille de la ligne car on est sûr que m_size_rows[m_last_row] >= m_last_column.
    if (m_size_columns[m_last_column] <= m_last_row)
      m_size_columns[m_last_column] = m_last_row + 1;
    return m_values_csv[m_last_row][m_last_column];
  }
  return m_values_csv[m_last_row][m_last_column - 1];
}

Real SimpleCsvOutputService::
elemRight(bool update_last_pos)
{
  if (m_last_row == -1 || m_last_column == -1 || m_last_column + 1 >= m_values_csv.dim2Size())
    return 0;

  // Par rapport à editElemRight(), si on ne veut pas mettre à jour la dernière position,
  // on ne verifie pas ni modifie m_size_columns.
  if (update_last_pos) {
    m_last_column++;

    if (m_size_rows[m_last_row] <= m_last_column)
      m_size_rows[m_last_row] = m_last_column + 1;
    if (m_size_columns[m_last_column] <= m_last_row)
      m_size_columns[m_last_column] = m_last_row + 1;
    return m_values_csv[m_last_row][m_last_column];
  }
  return m_values_csv[m_last_row][m_last_column + 1];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool SimpleCsvOutputService::
editElem(Real elem)
{
  m_values_csv[m_last_row][m_last_column] = elem;
  return true;
}

bool SimpleCsvOutputService::
editElem(Integer pos_x, Integer pos_y, Real elem)
{
  if (pos_x < 0 || pos_x >= m_values_csv.dim2Size() || pos_y < 0 || pos_y >= m_values_csv.dim1Size())
    return false;

  if (m_size_columns[pos_x] <= pos_y)
    m_size_columns[pos_x] = pos_y + 1;
  if (m_size_rows[pos_y] <= pos_x)
    m_size_rows[pos_y] = pos_x + 1;

  m_values_csv[pos_y][pos_x] = elem;

  m_last_row = pos_y;
  m_last_column = pos_x;

  return true;
}

bool SimpleCsvOutputService::
editElem(String name_column, String name_row, Real elem)
{
  std::optional<Integer> pos_x = m_name_columns.span().findFirst(name_column);
  std::optional<Integer> pos_y = m_name_rows.span().findFirst(name_row);

  if (pos_x && pos_y)
    return editElem(pos_x.value(), pos_y.value(), elem);
  return false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Real SimpleCsvOutputService::
elem()
{
  return m_values_csv[m_last_row][m_last_column];
}

Real SimpleCsvOutputService::
elem(Integer pos_x, Integer pos_y, bool update_last_pos)
{
  if (pos_x < 0 || pos_x >= m_values_csv.dim2Size() || pos_y < 0 || pos_y >= m_values_csv.dim1Size())
    return 0;

  if (update_last_pos) {
    m_last_column = pos_x;
    m_last_row = pos_y;
  }

  return m_values_csv[pos_y][pos_x];
}

Real SimpleCsvOutputService::
elem(String name_column, String name_row, bool update_last_pos)
{
  std::optional<Integer> pos_x = m_name_columns.span().findFirst(name_column);
  std::optional<Integer> pos_y = m_name_rows.span().findFirst(name_row);

  if (pos_x && pos_y)
    return elem(pos_x.value(), pos_y.value(), update_last_pos);
  return 0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

RealUniqueArray SimpleCsvOutputService::
row(Integer pos)
{
  Integer size = sizeRow(pos);
  RealUniqueArray copie(size);
  for (Integer i = 0; i < size; i++) {
    copie[i] = m_values_csv[pos][i];
  }
  return copie;
}

RealUniqueArray SimpleCsvOutputService::
row(String name_row)
{
  std::optional<Integer> pos_y = m_name_rows.span().findFirst(name_row);
  if (pos_y)
    return row(pos_y.value());
  return RealUniqueArray(0);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

RealUniqueArray SimpleCsvOutputService::
column(Integer pos)
{
  Integer size = sizeColumn(pos);

  RealUniqueArray copie(size);
  for (Integer i = 0; i < size; i++) {
    copie[i] = m_values_csv[i][pos];
  }
  return copie;
}

RealUniqueArray SimpleCsvOutputService::
column(String name_column)
{
  std::optional<Integer> pos_x = m_name_columns.span().findFirst(name_column);
  if (pos_x)
    return column(pos_x.value());
  return RealUniqueArray(0);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer SimpleCsvOutputService::
sizeRow(Integer pos)
{
  if (pos < 0 || pos >= m_values_csv.dim1Size())
    return 0;
  return m_size_rows[pos];
}

Integer SimpleCsvOutputService::
sizeRow(String name_row)
{
  std::optional<Integer> pos_y = m_name_rows.span().findFirst(name_row);
  if (pos_y)
    return sizeRow(pos_y.value());
  return 0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer SimpleCsvOutputService::
sizeColumn(Integer pos)
{
  if (pos < 0 || pos >= m_values_csv.dim2Size())
    return 0;
  return m_size_columns[pos];
}

Integer SimpleCsvOutputService::
sizeColumn(String name_column)
{
  std::optional<Integer> pos_x = m_name_columns.span().findFirst(name_column);
  if (pos_x)
    return sizeColumn(pos_x.value());
  return 0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer SimpleCsvOutputService::
posRow(String name_row)
{
  std::optional<Integer> pos_y = m_name_rows.span().findFirst(name_row);
  if (pos_y)
    return pos_y.value();
  return -1;
}

Integer SimpleCsvOutputService::
posColumn(String name_column)
{
  std::optional<Integer> pos_x = m_name_columns.span().findFirst(name_column);
  if (pos_x)
    return pos_x.value();
  return -1;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer SimpleCsvOutputService::
numRows()
{
  return m_values_csv.dim1Size();
}

Integer SimpleCsvOutputService::
numColumns()
{
  return m_values_csv.dim2Size();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool SimpleCsvOutputService::
editNameRow(Integer pos, String new_name)
{
  if (pos < 0 || pos >= m_values_csv.dim1Size())
    return false;
  m_name_rows[pos] = new_name;
  return true;
}

bool SimpleCsvOutputService::
editNameRow(String name_row, String new_name)
{
  std::optional<Integer> pos_y = m_name_rows.span().findFirst(name_row);
  if (pos_y)
    return editNameRow(pos_y.value(), new_name);
  return false;
}

bool SimpleCsvOutputService::
editNameColumn(Integer pos, String new_name)
{
  if (pos < 0 || pos >= m_values_csv.dim2Size())
    return false;
  m_name_columns[pos] = new_name;
  return true;
}

bool SimpleCsvOutputService::
editNameColumn(String name_column, String new_name)
{
  std::optional<Integer> pos_x = m_name_columns.span().findFirst(name_column);
  if (pos_x)
    return editNameColumn(pos_x.value(), new_name);
  return false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer SimpleCsvOutputService::
addAverageColumn(String name_column)
{
  Integer pos = addColumn(name_column);
  for (Integer i = 0; i < m_values_csv.dim1Size(); i++) {
    Real avg = 0.0;
    ConstArrayView<Real> view = m_values_csv[i];
    for (Integer j = 0; j < view.size() - 1; j++) {
      avg += view[j];
    }
    avg /= view.size() - 1;
    addElemColumn(pos, avg);
  }
  return pos;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SimpleCsvOutputService::
setPrecision(Integer precision)
{
  if (precision < 1)
    m_precision_print = 1;
  else if (precision > (std::numeric_limits<Real>::digits10 + 1))
    m_precision_print = (std::numeric_limits<Real>::digits10 + 1);
  else
    m_precision_print = precision;
}

void SimpleCsvOutputService::
setFixed(bool fixed)
{
  m_is_fixed_print = fixed;
}

void SimpleCsvOutputService::
print(Integer only_proc)
{
  if (only_proc != -1 && mesh()->parallelMng()->commRank() != only_proc)
    return;
  pinfo() << "P" << mesh()->parallelMng()->commRank() << " - Ecriture du tableau dans la sortie standard :";
  _print(std::cout);
  pinfo() << "P" << mesh()->parallelMng()->commRank() << " - Fin écriture tableau";
}

bool SimpleCsvOutputService::
writeFile(Integer only_proc)
{
  // Si l'on n'est pas le processus demandé, on return true.
  // -1 = tout le monde écrit.
  if (only_proc != -1 && mesh()->parallelMng()->commRank() != only_proc)
    return true;

  String file_name = _computeFinal();

  // Si true, alors les noms de fichier et dossier ne permettent pas d'écrire
  // un fichier par processus (pas de @proc_id@ dans l'un des noms).
  bool only_one_proc = m_path_only_P0 && m_name_tab_only_P0;

  // Si l'on a only_proc == -1 et que only_one_proc == true, alors il n'y a que le
  // processus 0 qui doit écrire.
  if ((only_proc == -1 && only_one_proc) && mesh()->parallelMng()->commRank() != 0)
    return true;

  Directory dir(m_path);
  bool sf = false;
  if (mesh()->parallelMng()->commRank() == 0 || only_proc != -1) {
    sf = dir.createDirectory();
  }
  if (mesh()->parallelMng()->commSize() != 1 && only_proc == -1) {
    sf = mesh()->parallelMng()->reduce(Parallel::ReduceMax, sf ? 1 : 0);
  }
  if (sf)
    return false;

  std::ofstream ofile(dir.file(file_name).localstr());
  if (ofile.fail())
    return false;

  _print(ofile);

  ofile.close();
  return true;
}

bool SimpleCsvOutputService::
writeFile(String path, Integer only_proc)
{
  m_path = path;
  m_path_computed = false;
  return writeFile(only_proc);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String SimpleCsvOutputService::
path()
{
  if (!m_path_computed) {
    m_path = _computeAt(m_path, m_path_only_P0);
    m_path_computed = true;
  }
  return m_path;
}

void SimpleCsvOutputService::
setPath(String path)
{
  m_path = path;
  m_path_computed = false;
}

String SimpleCsvOutputService::
name()
{
  if (!m_name_tab_computed) {
    m_name_tab = _computeAt(m_name_tab, m_name_tab_only_P0);
    m_name_tab_computed = true;
  }
  return m_name_tab;
}

void SimpleCsvOutputService::
setName(String name)
{
  m_name_tab = name;
  m_name_tab_computed = false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/**
 * @brief Méthode permettant de finaliser les noms avant écriture.
 * 
 * @return String Le nom du fichier à sortir (avec extension).
 */
String SimpleCsvOutputService::
_computeFinal()
{
  if (!m_path_computed) {
    m_path = _computeAt(m_path, m_path_only_P0);
    m_path_computed = true;
  }
  if (!m_name_tab_computed) {
    m_name_tab = _computeAt(m_name_tab, m_name_tab_only_P0);
    m_name_tab_computed = true;
  }
  return m_name_tab + ".csv";
}

/**
 * @brief Méthode permettant de remplacer les symboles de nom par leur valeur.
 * 
 * @param name [IN] Le nom à modifier.
 * @param only_once [OUT] Si le nom contient le symbole '\@proc_id\@' permettant 
 *                de différencier les fichiers écrits par differents processus.
 * @return String Le nom avec les symboles remplacés.
 */
String SimpleCsvOutputService::
_computeAt(String name, bool& only_once)
{
  // Permet de contourner le bug avec String::split() si le nom commence par '@'.
  if (name.startsWith("@")) {
    name = "@" + name;
  }

  StringUniqueArray string_splited;
  // On découpe la string là où se trouve les @.
  name.split(string_splited, '@');

  // On traite les mots entre les "@".
  if (string_splited.size() > 1) {
    // On recherche "proc_id" dans le tableau (donc @proc_id@ dans le nom).
    std::optional<Integer> proc_id = string_splited.span().findFirst("proc_id");
    // On remplace "@proc_id@" par l'id du proc.
    if (proc_id) {
      string_splited[proc_id.value()] = String::fromNumber(mesh()->parallelMng()->commRank());
      only_once = false;
    }
    // Il n'y a que un seul proc qui write.
    else {
      only_once = true;
    }

    // On recherche "num_procs" dans le tableau (donc @num_procs@ dans le nom).
    std::optional<Integer> num_procs = string_splited.span().findFirst("num_procs");
    // On remplace "@num_procs@" par l'id du proc.
    if (num_procs) {
      string_splited[num_procs.value()] = String::fromNumber(mesh()->parallelMng()->commSize());
    }
  }

  // On recombine la chaine.
  String combined = "";
  for (String str : string_splited) {
    // Permet de contourner le bug avec String::split() s'il y a '@@@' dans le nom ou si le
    // nom commence par '@' (en complément des premières lignes de la méthode).
    if (str == "@")
      continue;
    combined = combined + str;
  }
  return combined;
}

/**
 * @brief Méthode permettant d'écrire le tableau dans un stream de sortie.
 * 
 * @param stream [IN/OUT] Le stream de sortie.
 */
void SimpleCsvOutputService::
_print(std::ostream& stream)
{
  // On enregistre les infos du stream pour les restaurer à la fin.
  std::ios_base::fmtflags save_flags = stream.flags();
  std::streamsize save_prec = stream.precision();

  if (m_is_fixed_print) {
    stream << std::setiosflags(std::ios::fixed);
  }
  stream << std::setprecision(m_precision_print);

  stream << m_name_tab << m_separator;

  for (Integer j = 0; j < m_name_columns.size(); j++) {
    stream << m_name_columns[j] << m_separator;
  }
  stream << std::endl;

  for (Integer i = 0; i < m_values_csv.dim1Size(); i++) {
    if (m_name_rows.size() > i)
      stream << m_name_rows[i];
    stream << m_separator;
    ConstArrayView<Real> view = m_values_csv[i];
    for (Integer j = 0; j < m_values_csv.dim2Size(); j++) {
      stream << view[j] << m_separator;
    }
    stream << std::endl;
  }

  stream.flags(save_flags);
  stream.precision(save_prec);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
