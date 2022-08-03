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

#include <arcane/Directory.h>
#include <arcane/IMesh.h>
#include <arcane/IParallelMng.h>

#include <optional>
#include <string>
#include <regex>

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
  m_root_path = Directory(subDomain()->exportDirectory(), m_iSTO->outputFileType()+"_refs");
  m_ref_path = Directory(m_root_path, m_output_dir);
  m_name_tab = m_iSTO->tabName();
  m_file_name = m_name_tab+"."+m_iSTO->outputFileType();
}

void SimpleCsvComparatorService::
clear()
{
  m_stic.clear();

  m_sti_to_compare.clear();

  m_sti_ref = nullptr;
  m_iSTO = nullptr;
  
  m_is_file_read = false;
}

void SimpleCsvComparatorService::
editRootDir(Directory root_dir)
{
  m_root_path = root_dir;
  m_ref_path = Directory(m_root_path, m_output_dir);
}

void SimpleCsvComparatorService::
print(Integer only_proc)
{
  m_scrw.print(only_proc);
}

bool SimpleCsvComparatorService::
writeRefFile(Integer only_proc)
{
  
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
  if (only_proc != -1 && mesh()->parallelMng()->commRank() != only_proc)
    return false;

  m_is_file_read = m_scrw.read(m_ref_path, m_name_tab);

  return m_is_file_read;
}

bool SimpleCsvComparatorService::
isRefExist(Integer only_proc)
{
  if (only_proc != -1 && mesh()->parallelMng()->commRank() != only_proc)
    return false;

  return SimpleTableReaderWriterUtils::isFileExist(m_ref_path, m_file_name);
}

bool SimpleCsvComparatorService::
compareWithRef(Integer only_proc, Integer epsilon)
{
  // Si le proc appelant ne doit pas lire.
  if (only_proc != -1 && mesh()->parallelMng()->commRank() != only_proc){
    return false;
  }
  // Si le fichier ne peut pas être lu.
  if (!m_is_file_read && !readRefFile(only_proc)){
    return false;
  }

  return m_stic.compare(epsilon);
}

bool SimpleCsvComparatorService::
addColumnToCompare(String name_column)
{
  return m_stic.addColumnToCompare(name_column);
}
bool SimpleCsvComparatorService::
addRowToCompare(String name_row)
{
  return m_stic.addRowToCompare(name_row);
}

bool SimpleCsvComparatorService::
removeColumnToCompare(String name_column)
{
  return m_stic.removeColumnToCompare(name_column);
}
bool SimpleCsvComparatorService::
removeRowToCompare(String name_row)
{
  return m_stic.removeRowToCompare(name_row);
}

void SimpleCsvComparatorService::
editRegexColumns(String regex_column)
{
  m_stic.editRegexColumns(regex_column);
}
void SimpleCsvComparatorService::
editRegexRows(String regex_row)
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
