﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
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

  if (!m_is_already_init) {
    m_is_already_init = true;

    m_simple_table_internal_reference = makeRef(new SimpleTableInternal(mesh()->parallelMng()));

    m_simple_table_internal_comparator.setInternalRef(m_simple_table_internal_reference);
    m_simple_csv_reader_writer.setInternal(m_simple_table_internal_reference);
  }

  m_simple_table_internal_to_compare = m_simple_table_output_ptr->internal();
  m_simple_table_internal_comparator.setInternalToCompare(m_simple_table_internal_to_compare);

  // On déduit l'emplacement des fichiers de réferences.
  m_output_directory = m_simple_table_output_ptr->outputDirectory();
  m_root_path = Directory(subDomain()->exportDirectory(), m_simple_table_output_ptr->fileType() + "_refs");
  m_reference_path = Directory(m_root_path, m_output_directory);
  m_table_name = m_simple_table_output_ptr->tableName();
  m_file_name = m_table_name + "." + m_simple_table_output_ptr->fileType();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SimpleCsvComparatorService::
clear()
{
  m_simple_table_internal_comparator.clearComparator();
  m_is_file_read = false;

  if (m_is_already_init) {
    m_simple_table_internal_reference->clear();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SimpleCsvComparatorService::
editRootDirectory(const Directory& root_directory)
{
  m_root_path = root_directory;
  m_reference_path = Directory(m_root_path, m_output_directory);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SimpleCsvComparatorService::
print(Integer rank)
{
  if (rank != -1 && subDomain()->parallelMng()->commRank() != rank)
    return;
  m_simple_csv_reader_writer.print();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/**
 * Méthode effectuant des opérations collectives.
 */
bool SimpleCsvComparatorService::
writeReferenceFile(Integer rank)
{
  ARCANE_CHECK_PTR(m_simple_table_output_ptr);
  // On sauvegarde les paramètres d'origines.
  Integer save_preci = m_simple_table_output_ptr->precision();
  bool save_scientific = m_simple_table_output_ptr->isForcedToUseScientificNotation();
  bool save_fixed = m_simple_table_output_ptr->isFixed();

  // On défini la précision max (+2 pour erreurs d'arrondis).
  m_simple_table_output_ptr->setPrecision(std::numeric_limits<Real>::max_digits10);
  m_simple_table_output_ptr->setForcedToUseScientificNotation(true);
  m_simple_table_output_ptr->setFixed(false);

  // On écrit nos fichiers de référence.
  bool fin = m_simple_table_output_ptr->writeFile(m_root_path, rank);

  // On remet les paramètres par défault.
  m_simple_table_output_ptr->setPrecision(save_preci);
  m_simple_table_output_ptr->setForcedToUseScientificNotation(save_scientific);
  m_simple_table_output_ptr->setFixed(save_fixed);

  return fin;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool SimpleCsvComparatorService::
readReferenceFile(Integer rank)
{
  if (rank != -1 && subDomain()->parallelMng()->commRank() != rank)
    return true;

  m_is_file_read = m_simple_csv_reader_writer.readTable(m_reference_path, m_table_name);

  return m_is_file_read;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool SimpleCsvComparatorService::
isReferenceExist(Integer rank)
{
  if (rank != -1 && subDomain()->parallelMng()->commRank() != rank)
    return true;

  return SimpleTableReaderWriterUtils::isFileExist(m_reference_path, m_file_name);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool SimpleCsvComparatorService::
compareWithReference(Integer rank, bool compare_dimension_too)
{
  // Si le proc appelant ne doit pas lire.
  if (rank != -1 && subDomain()->parallelMng()->commRank() != rank) {
    return true;
  }
  // Si le fichier ne peut pas être lu.
  if (!m_is_file_read && !readReferenceFile(rank)) {
    error() << "Error with the file reader: invalid file";
    return false;
  }

  return m_simple_table_internal_comparator.compare(compare_dimension_too);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool SimpleCsvComparatorService::
compareElemWithReference(const String& column_name, const String& row_name, Integer rank)
{
  // Si le proc appelant ne doit pas lire.
  if (rank != -1 && subDomain()->parallelMng()->commRank() != rank) {
    return true;
  }
  // Si le fichier ne peut pas être lu.
  if (!m_is_file_read && !readReferenceFile(rank)) {
    error() << "Error with the file reader: invalid file";
    return false;
  }

  return m_simple_table_internal_comparator.compareElem(column_name, row_name);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool SimpleCsvComparatorService::
compareElemWithReference(Real elem, const String& column_name, const String& row_name, Integer rank)
{
  // Si le proc appelant ne doit pas lire.
  if (rank != -1 && subDomain()->parallelMng()->commRank() != rank) {
    return true;
  }
  // Si le fichier ne peut pas être lu.
  if (!m_is_file_read && !readReferenceFile(rank)) {
    error() << "Error with the file reader: invalid file";
    return false;
  }

  return m_simple_table_internal_comparator.compareElem(elem, column_name, row_name);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool SimpleCsvComparatorService::
addColumnForComparing(const String& column_name)
{
  return m_simple_table_internal_comparator.addColumnForComparing(column_name);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool SimpleCsvComparatorService::
addRowForComparing(const String& row_name)
{
  return m_simple_table_internal_comparator.addRowForComparing(row_name);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SimpleCsvComparatorService::
isAnArrayExclusiveColumns(bool is_exclusive)
{
  m_simple_table_internal_comparator.isAnArrayExclusiveColumns(is_exclusive);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SimpleCsvComparatorService::
isAnArrayExclusiveRows(bool is_exclusive)
{
  m_simple_table_internal_comparator.isAnArrayExclusiveRows(is_exclusive);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SimpleCsvComparatorService::
editRegexColumns(const String& regex_column)
{
  m_simple_table_internal_comparator.editRegexColumns(regex_column);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SimpleCsvComparatorService::
editRegexRows(const String& regex_row)
{
  m_simple_table_internal_comparator.editRegexRows(regex_row);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SimpleCsvComparatorService::
isARegexExclusiveColumns(bool is_exclusive)
{
  m_simple_table_internal_comparator.isARegexExclusiveColumns(is_exclusive);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SimpleCsvComparatorService::
isARegexExclusiveRows(bool is_exclusive)
{
  m_simple_table_internal_comparator.isARegexExclusiveRows(is_exclusive);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool SimpleCsvComparatorService::
addEpsilonColumn(const String& column_name, Real epsilon)
{
  return m_simple_table_internal_comparator.addEpsilonColumn(column_name, epsilon);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool SimpleCsvComparatorService::
addEpsilonRow(const String& row_name, Real epsilon)
{
  return m_simple_table_internal_comparator.addEpsilonRow(row_name, epsilon);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE_SIMPLECSVCOMPARATOR(SimpleCsvComparator, SimpleCsvComparatorService);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
