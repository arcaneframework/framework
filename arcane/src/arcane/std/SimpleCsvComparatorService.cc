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
  ARCANE_CHECK_PTR(ptr_sto);
  m_iSTO = ptr_sto;
  // On déduit la position des tableaux de références avec le STO qu'on 
  // nous a donné.
  m_path_ref_str = m_iSTO->dir()+"_ref";
  m_path_ref = Directory(m_iSTO->rootPathOutput(), m_path_ref_str);
  m_name_ref = m_iSTO->nameFile();
}

void SimpleCsvComparatorService::
editRefFileEntry(String path, String name)
{
  m_path_ref = Directory(path);
  m_name_ref = name;
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
  bool fin = m_iSTO->writeFile(m_path_ref_str, only_proc);

  // On remet les paramètres par défault.
  m_iSTO->setPrecision(save_preci);
  m_iSTO->setFixed(save_fixed);

  return fin;
}

bool SimpleCsvComparatorService::
isRefExist(Integer only_proc)
{
  if (only_proc != -1 && mesh()->parallelMng()->commRank() != only_proc)
    return false;
  _openFile(m_name_ref);
  info() << "file : " << m_name_ref << " " << m_ifstream.good();
  return m_ifstream.good();
}

bool SimpleCsvComparatorService::
readRefFile(Integer only_proc)
{
  // Pas de fichier, pas de chocolats.
  if(!isRefExist(only_proc)) return false;
  info() << "Pass 1";

  std::string line;

  // S'il n'y a pas de première ligne, on arrete là.
  // Un fichier écrit par SimpleCsvOutput possède toujours au
  // moins une ligne.
  if(!std::getline(m_ifstream, line)) return false;
  info() << "Pass 2";

  // Sinon, on a la ligne d'entête, contenant les noms
  // des colonnes (et le nom du tableau).
  String ligne(line);
  ligne.split(m_name_columns_with_name_of_tab, ';');

  // S'il n'y a pas d'autres lignes, c'est qu'il n'y a que des 
  // colonnes vides (ou aucunes colonnes) et aucunes lignes.
  if(!std::getline(m_ifstream, line)) return true;
  info() << "Pass 3";

  Integer num_columns = m_name_columns_with_name_of_tab.size()-1;

  m_values_csv.resize(1, num_columns);


  Integer compt_line = 0;

  do{
    StringUniqueArray splitted_line;
    m_values_csv.resize(compt_line+1);
    String ligne(line);
    ligne.split(splitted_line, ';');
    m_name_rows.add(splitted_line[0]);
    for(Integer i = 1; i < splitted_line.size(); i++){
      m_values_csv[compt_line][i-1] = std::stod(splitted_line[i].localstr());
    }
    compt_line++;
  } while(std::getline(m_ifstream, line));
  info() << "Pass 4";

  return true;
}

bool SimpleCsvComparatorService::
compareWithRef(Integer epsilon)
{
  bool isOk = true;
  Integer dim1 = m_values_csv.dim1Size();
  Integer dim2 = m_values_csv.dim2Size();
  for (Integer i = 0; i < dim1; i++) {
    if(!_exploreRows(i)) continue;
    ConstArrayView<Real> view = m_values_csv[i];
    for (Integer j = 0; j < dim2; j++) {
      if(!_exploreColumn(j)) continue;
      Real val1 = m_iSTO->elem(m_name_columns_with_name_of_tab[j+1], m_name_rows[i]);
      Real val2 = view[j];
      if(math::isNearlyEqualWithEpsilon(val1, val2, epsilon))
      {
        info() << "Compare value -- Column name: " << m_name_columns_with_name_of_tab[j+1] << " -- Row name: " << m_name_rows[i];
        info() << "It's equals";
      }
      else
      {
        info() << "Compare value -- Column name: " << m_name_columns_with_name_of_tab[j+1] << " -- Row name: " << m_name_rows[i];
        warning() << "It's not equals";
        isOk = false;
      }
    }
  }
  return isOk;
}

// TODO DEBUG A suppr.
void SimpleCsvComparatorService::
print()
{
  String m_separator = ";";


  std::cout << std::setiosflags(std::ios::fixed);

  std::cout << std::setprecision(std::numeric_limits<Real>::digits10 + 1);

  for (Integer j = 0; j < m_name_columns_with_name_of_tab.size(); j++) {
    std::cout << m_name_columns_with_name_of_tab[j] << m_separator;
  }
  std::cout << std::endl;

  for (Integer i = 0; i < m_values_csv.dim1Size(); i++) {
    std::cout << m_name_rows[i];
    std::cout << m_separator;
    ConstArrayView<Real> view = m_values_csv[i];
    for (Integer j = 0; j < m_values_csv.dim2Size(); j++) {
      std::cout << view[j] << m_separator;
    }
    std::cout << std::endl;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SimpleCsvComparatorService::
_openFile(String name_file)
{
  if(m_is_file_open) return;
  m_ifstream.open(m_path_ref.file(name_file).localstr(), std::ifstream::in);
  m_is_file_open = true;
}

bool SimpleCsvComparatorService::
_exploreColumn(Integer pos)
{
  // S'il n'y a pas de précisions, on compare toutes les colonnes.
  if(m_compared_columns.empty() && m_regex_columns.empty()) {
    return true;
  }

  String name_column = m_name_columns_with_name_of_tab[pos+1];

  // D'abord, on regarde si le nom de la colonne est dans 
  // le tableau. 
  if(m_compared_columns.contains(name_column))
  {
    return true;
  }
  // S'il n'est pas dans le tableau et qu'il n'a a pas de regex
  // on return false.
  else if(m_regex_columns.empty())
  {
    return false;
  }

  // Sinon, on regarde aussi la regex.
  std::regex self_regex(m_regex_columns.localstr(), std::regex_constants::ECMAScript | std::regex_constants::icase);
  if (std::regex_search(name_column.localstr(), self_regex))
  {
    return !m_is_excluding_regex_columns;
  }

  return m_is_excluding_regex_columns;
}

bool SimpleCsvComparatorService::
_exploreRows(Integer pos)
{
  // S'il n'y a pas de précisions, on compare toutes les colonnes.
  if(m_compared_rows.empty() && m_regex_rows.empty()) {
    return true;
  }

  String name_rows = m_name_rows[pos];

  // D'abord, on regarde si le nom de la colonne est dans 
  // le tableau. 
  if(m_compared_rows.contains(name_rows))
  {
    return true;
  }
  // S'il n'est pas dans le tableau et qu'il n'a a pas de regex
  // on return false.
  else if(m_regex_rows.empty())
  {
    return false;
  }

  // Sinon, on regarde aussi la regex.
  std::regex self_regex(m_regex_rows.localstr(), std::regex_constants::ECMAScript | std::regex_constants::icase);
  if (std::regex_search(name_rows.localstr(), self_regex))
  {
    return !m_is_excluding_regex_rows;
  }

  return m_is_excluding_regex_rows;
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

bool SimpleCsvComparatorService::
editRegexColumns(String regex_column)
{
  m_regex_columns = regex_column;
  return true;
}
bool SimpleCsvComparatorService::
editRegexRows(String regex_row)
{
  m_regex_rows = regex_row;
  return true;
}

bool SimpleCsvComparatorService::
isARegexExclusiveColumns(bool is_exclusive)
{
  m_is_excluding_regex_columns = is_exclusive;
}
bool SimpleCsvComparatorService::
isARegexExclusiveRows(bool is_exclusive)
{
  m_is_excluding_regex_rows = is_exclusive;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
