// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SimpleCsvComparatorService.cc                                   (C) 2000-2022 */
/*                                                                           */
/* Service permettant de construire et de sortir un tableau au formet csv.   */
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
init(ISimpleTableOutput* ptr_sto)
{
  // On enregistre le pointeur qui nous est donné.
  ARCANE_CHECK_PTR(ptr_sto);
  m_iSTO = ptr_sto;

  m_sti_ref = m_iSTO->internal();
  m_stic.setInternalRef(m_sti_ref);

  // On déduit l'emplacement des fichiers de réferences.
  m_output_dir = m_iSTO->outputDir();
  m_root_path = Directory(subDomain()->exportDirectory(), m_iSTO->outputFileType() + "_refs");
  m_ref_path = Directory(m_root_path, m_output_dir);
  m_name_tab = m_iSTO->tabName();
  m_file_name = m_name_tab + "." + m_iSTO->outputFileType();
}

void SimpleCsvComparatorService::
clear()
{
  m_stic.clearComparator();

  m_sti_to_compare.clear();

  m_sti_ref = nullptr;
  m_iSTO = nullptr;

  m_is_file_read = false;
}

void SimpleCsvComparatorService::
editRootDir(const Directory& root_dir)
{
  m_root_path = root_dir;
  m_ref_path = Directory(m_root_path, m_output_dir);
}

void SimpleCsvComparatorService::
print(Integer only_proc)
{
  if (only_proc != -1 && subDomain()->parallelMng()->commRank() != only_proc)
    return;
  m_scrw.print();
}

bool SimpleCsvComparatorService::
writeRefFile(Integer only_proc)
{
  ARCANE_CHECK_PTR(m_iSTO);
  // On sauvegarde les paramètres d'origines.
  Integer save_preci = m_iSTO->precision();
  bool save_fixed = m_iSTO->fixed();

  // On défini la précision max.
  m_iSTO->setPrecision(std::numeric_limits<Real>::digits10 + 1);
  m_iSTO->setFixed(true);

  // On écrit nos fichiers de référence.
  bool fin = m_iSTO->writeFile(m_root_path, only_proc);

  // On remet les paramètres par défault.
  m_iSTO->setPrecision(save_preci);
  m_iSTO->setFixed(save_fixed);

  return fin;
}

bool SimpleCsvComparatorService::
readRefFile(Integer only_proc)
{
  if (only_proc != -1 && subDomain()->parallelMng()->commRank() != only_proc)
    return false;

  m_is_file_read = m_scrw.readTable(m_ref_path, m_name_tab);

  return m_is_file_read;
}

bool SimpleCsvComparatorService::
isRefExist(Integer only_proc)
{
  if (only_proc != -1 && subDomain()->parallelMng()->commRank() != only_proc)
    return false;

  return SimpleTableReaderWriterUtils::isFileExist(m_ref_path, m_file_name);
}

bool SimpleCsvComparatorService::
compareWithRef(Integer only_proc, Integer epsilon, bool dim_compare)
{
  ARCANE_CHECK_PTR(m_sti_ref);
  // Si le proc appelant ne doit pas lire.
  if (only_proc != -1 && subDomain()->parallelMng()->commRank() != only_proc) {
    return false;
  }
  // Si le fichier ne peut pas être lu.
  if (!m_is_file_read && !readRefFile(only_proc)) {
    return false;
  }

  m_sti_ref->m_values_csv.dim1Size();

  return m_stic.compare(epsilon, dim_compare);
}

bool SimpleCsvComparatorService::
addColumnForComparing(const String& name_column)
{
  return m_stic.addColumnForComparing(name_column);
}
bool SimpleCsvComparatorService::
addRowForComparing(const String& name_row)
{
  return m_stic.addRowForComparing(name_row);
}

void SimpleCsvComparatorService::
isAnArrayExclusiveColumns(bool is_exclusive)
{
  m_stic.isAnArrayExclusiveColumns(is_exclusive);
}
void SimpleCsvComparatorService::
isAnArrayExclusiveRows(bool is_exclusive)
{
  m_stic.isAnArrayExclusiveRows(is_exclusive);
}

void SimpleCsvComparatorService::
editRegexColumns(const String& regex_column)
{
  m_stic.editRegexColumns(regex_column);
}
void SimpleCsvComparatorService::
editRegexRows(const String& regex_row)
{
  m_stic.editRegexRows(regex_row);
}

void SimpleCsvComparatorService::
isARegexExclusiveColumns(bool is_exclusive)
{
  m_stic.isARegexExclusiveColumns(is_exclusive);
}
void SimpleCsvComparatorService::
isARegexExclusiveRows(bool is_exclusive)
{
  m_stic.isARegexExclusiveRows(is_exclusive);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
