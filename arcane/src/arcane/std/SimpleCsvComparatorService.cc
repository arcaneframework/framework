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

  // On déduit l'emplacement des fichiers de réferences.
  m_output_dir = m_iSTO->outputDir();
  m_root_path = Directory(subDomain()->exportDirectory(), "csv_refs");
  m_ref_path = Directory(m_root_path, m_output_dir);
  m_name_tab = m_iSTO->tabName();
  if(m_iSTO->outputFileType() == "csv")
  {
    m_file_name = m_iSTO->fileName();
    m_is_csv_implem = true;
  }
  else
  {
    m_file_name = m_name_tab + ".csv";
    m_is_csv_implem = false;
    warning() << "L'implémentation de ISimpleTableOutput n'utilise pas de fichier de type 'csv'";
  }

  // On défini la précision max.
  m_precision_print = std::numeric_limits<Real>::digits10 + 1;
  m_is_fixed_print = true;
}

void SimpleCsvComparatorService::
clear()
{
  clearCsv();

  m_iSTO = nullptr;
  
  m_is_file_read = false;

  m_regex_rows = "";
  m_is_excluding_regex_rows = false;

  m_regex_columns = "";
  m_is_excluding_regex_columns = false;

  m_compared_rows.clear();
  m_compared_columns.clear();
  m_is_csv_implem = false;
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
  printCsv(only_proc);
}

bool SimpleCsvComparatorService::
writeRefFile(Integer only_proc)
{
  if(!m_is_csv_implem) {
    warning() << "Pour l'instant, ce service utilise la méthode 'writeFile()' \
                  de l'objet implémentant ISimpleTableOutput \
                  passé en paramètre lors de l'init, or cet objet \
                  n'utilise pas le type 'csv', donc impossible \
                  d'utiliser cette méthode.";
    return false;
  }

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

  m_is_file_read = readCsv(m_ref_path, m_file_name);
  return m_is_file_read;
}

bool SimpleCsvComparatorService::
isRefExist(Integer only_proc)
{
  if (only_proc != -1 && mesh()->parallelMng()->commRank() != only_proc)
    return false;
  return isFileExist(m_ref_path, m_file_name);
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

  bool is_ok = true;

  const Integer dim1 = m_values_csv.dim1Size();
  const Integer dim2 = m_values_csv.dim2Size();

  for (Integer i = 0; i < dim1; i++) {
    // On regarde si l'on doit comparer la ligne actuelle.
    if(!_exploreRows(i)) continue;

    ConstArrayView<Real> view = m_values_csv[i];

    for (Integer j = 0; j < dim2; j++) {
    // On regarde si l'on doit comparer la colonne actuelle.
      if(!_exploreColumn(j)) continue;

      const Real val1 = m_iSTO->elem(m_name_columns[j], m_name_rows[i]);
      const Real val2 = view[j];

      if(!math::isNearlyEqualWithEpsilon(val1, val2, epsilon)) {
        warning() << "Values not equals -- Column name: \"" << m_name_columns[j] << "\" -- Row name: \"" << m_name_rows[i] << "\"";
        is_ok = false;
      }
    }
  }
  return is_ok;
}

bool SimpleCsvComparatorService::
addColumnToCompare(String name_column)
{
  m_compared_columns.add(name_column);
  return true;
}
bool SimpleCsvComparatorService::
addRowToCompare(String name_row)
{
  m_compared_rows.add(name_row);
  return true;
}

bool SimpleCsvComparatorService::
removeColumnToCompare(String name_column)
{
  std::optional index = m_compared_columns.span().findFirst(name_column);
  if(index) {
    m_compared_columns.remove(index.value());
    return true;
  }
  return false;
}
bool SimpleCsvComparatorService::
removeRowToCompare(String name_row)
{
  std::optional index = m_compared_rows.span().findFirst(name_row);
  if(index) {
    m_compared_rows.remove(index.value());
    return true;
  }
  return false;
}

void SimpleCsvComparatorService::
editRegexColumns(String regex_column)
{
  m_regex_columns = regex_column;
}
void SimpleCsvComparatorService::
editRegexRows(String regex_row)
{
  m_regex_rows = regex_row;
}

void SimpleCsvComparatorService::
isARegexExclusiveColumns(bool is_exclusive)
{
  m_is_excluding_regex_columns = is_exclusive;
}
void SimpleCsvComparatorService::
isARegexExclusiveRows(bool is_exclusive)
{
  m_is_excluding_regex_rows = is_exclusive;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool SimpleCsvComparatorService::
_exploreColumn(Integer pos)
{
  // S'il n'y a pas de précisions, on compare toutes les colonnes.
  if(m_compared_columns.empty() && m_regex_columns.empty()) {
    return true;
  }

  const String name_column = m_name_columns[pos];

  // D'abord, on regarde si le nom de la colonne est dans le tableau. 
  if(m_compared_columns.contains(name_column))
  {
    return true;
  }

  // S'il n'est pas dans le tableau et qu'il n'a a pas de regex, on return false.
  else if(m_regex_columns.empty())
  {
    return false;
  }

  // Sinon, on regarde aussi la regex.
  // TODO : Voir s'il y a un interet de faire des regex en mode JS.
  std::regex self_regex(m_regex_columns.localstr(), std::regex_constants::ECMAScript | std::regex_constants::icase);

  // Si quelque chose dans le nom correspond à la regex.
  if (std::regex_search(name_column.localstr(), self_regex))
  {
    return !m_is_excluding_regex_columns;
  }

  // Sinon.
  return m_is_excluding_regex_columns;
}

bool SimpleCsvComparatorService::
_exploreRows(Integer pos)
{
  // S'il n'y a pas de précisions, on compare toutes les colonnes.
  if(m_compared_rows.empty() && m_regex_rows.empty()) {
    return true;
  }

  const String name_rows = m_name_rows[pos];

  // D'abord, on regarde si le nom de la colonne est dans le tableau. 
  if(m_compared_rows.contains(name_rows))
  {
    return true;
  }
  // S'il n'est pas dans le tableau et qu'il n'a a pas de regex, on return false.
  else if(m_regex_rows.empty())
  {
    return false;
  }

  // Sinon, on regarde aussi la regex.
  // TODO : Voir s'il y a un interet de faire des regex en mode JS.
  std::regex self_regex(m_regex_rows.localstr(), std::regex_constants::ECMAScript | std::regex_constants::icase);
  if (std::regex_search(name_rows.localstr(), self_regex))
  {
    return !m_is_excluding_regex_rows;
  }

  return m_is_excluding_regex_rows;
}



/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
