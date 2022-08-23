// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SimpleCsvComparatorService.cc                               (C) 2000-2022 */
/*                                                                           */
/* Service permettant de comparer un ISimpleTableOutput avec un fichier de   */
/* référence en format csv.                                                  */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/std/SimpleCsvComparatorService.h"

#include "arcane/Directory.h"
#include "arcane/IMesh.h"
#include "arcane/IParallelMng.h"
#include "arcane/utils/Iostream.h"

#include <optional>
#include <regex>
#include <string>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SimpleCsvComparatorService::
init(ISimpleTableOutput* simple_table_output_ptr)
{
  // On enregistre le pointeur qui nous est donné.
  ARCANE_CHECK_PTR(simple_table_output_ptr);
  m_simple_table_output_ptr = simple_table_output_ptr;

  m_simple_table_internal_reference = m_simple_table_output_ptr->internal();
  m_simple_table_internal_comparator.setInternalRef(m_simple_table_internal_reference);

  // On déduit l'emplacement des fichiers de réferences.
  m_output_directory = m_simple_table_output_ptr->outputDirectory();
  m_root_path = Directory(subDomain()->exportDirectory(), m_simple_table_output_ptr->fileType() + "_refs");
  m_reference_path = Directory(m_root_path, m_output_directory);
  m_table_name = m_simple_table_output_ptr->tableName();
  m_file_name = m_table_name + "." + m_simple_table_output_ptr->fileType();
}

void SimpleCsvComparatorService::
clear()
{
  m_simple_table_internal_comparator.clearComparator();

  m_simple_table_internal_to_compare->clear();

  m_simple_table_internal_reference.reset();
  m_simple_table_output_ptr = nullptr;

  m_is_file_read = false;
}

void SimpleCsvComparatorService::
editRootDirectory(const Directory& root_directory)
{
  m_root_path = root_directory;
  m_reference_path = Directory(m_root_path, m_output_directory);
}

void SimpleCsvComparatorService::
print(Integer rank)
{
  if (rank != -1 && subDomain()->parallelMng()->commRank() != rank)
    return;
  m_simple_csv_reader_writer.print();
}

bool SimpleCsvComparatorService::
writeReferenceFile(Integer rank)
{
  ARCANE_CHECK_PTR(m_simple_table_output_ptr);
  // On sauvegarde les paramètres d'origines.
  Integer save_preci = m_simple_table_output_ptr->precision();
  bool save_fixed = m_simple_table_output_ptr->isFixed();

  // On défini la précision max.
  m_simple_table_output_ptr->setPrecision(std::numeric_limits<Real>::digits10 + 1);
  m_simple_table_output_ptr->setFixed(true);

  // On écrit nos fichiers de référence.
  bool fin = m_simple_table_output_ptr->writeFile(m_root_path, rank);

  // On remet les paramètres par défault.
  m_simple_table_output_ptr->setPrecision(save_preci);
  m_simple_table_output_ptr->setFixed(save_fixed);

  return fin;
}

bool SimpleCsvComparatorService::
readReferenceFile(Integer rank)
{
  if (rank != -1 && subDomain()->parallelMng()->commRank() != rank)
    return false;

  m_is_file_read = m_simple_csv_reader_writer.readTable(m_reference_path, m_table_name);

  return m_is_file_read;
}

bool SimpleCsvComparatorService::
isReferenceExist(Integer rank)
{
  if (rank != -1 && subDomain()->parallelMng()->commRank() != rank)
    return false;

  return SimpleTableReaderWriterUtils::isFileExist(m_reference_path, m_file_name);
}

bool SimpleCsvComparatorService::
compareWithReference(Integer rank, Integer epsilon, bool compare_dimension_too)
{
  // Si le proc appelant ne doit pas lire.
  if (rank != -1 && subDomain()->parallelMng()->commRank() != rank) {
    return false;
  }
  // Si le fichier ne peut pas être lu.
  if (!m_is_file_read && !readReferenceFile(rank)) {
    return false;
  }

  m_simple_table_internal_reference->m_values.dim1Size();

  return m_simple_table_internal_comparator.compare(epsilon, compare_dimension_too);
}

bool SimpleCsvComparatorService::
addColumnForComparing(const String& column_name)
{
  return m_simple_table_internal_comparator.addColumnForComparing(column_name);
}
bool SimpleCsvComparatorService::
addRowForComparing(const String& row_name)
{
  return m_simple_table_internal_comparator.addRowForComparing(row_name);
}

void SimpleCsvComparatorService::
isAnArrayExclusiveColumns(bool is_exclusive)
{
  m_simple_table_internal_comparator.isAnArrayExclusiveColumns(is_exclusive);
}
void SimpleCsvComparatorService::
isAnArrayExclusiveRows(bool is_exclusive)
{
  m_simple_table_internal_comparator.isAnArrayExclusiveRows(is_exclusive);
}

void SimpleCsvComparatorService::
editRegexColumns(const String& regex_column)
{
  m_simple_table_internal_comparator.editRegexColumns(regex_column);
}
void SimpleCsvComparatorService::
editRegexRows(const String& regex_row)
{
  m_simple_table_internal_comparator.editRegexRows(regex_row);
}

void SimpleCsvComparatorService::
isARegexExclusiveColumns(bool is_exclusive)
{
  m_simple_table_internal_comparator.isARegexExclusiveColumns(is_exclusive);
}
void SimpleCsvComparatorService::
isARegexExclusiveRows(bool is_exclusive)
{
  m_simple_table_internal_comparator.isARegexExclusiveRows(is_exclusive);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
