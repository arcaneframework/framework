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
  m_tab_name = m_iSTO->tabName();
  if(m_iSTO->outputFileType() == "csv")
  {
    m_file_name = m_iSTO->fileName();
    m_is_csv_implem = true;
  }
  else
  {
    m_file_name = m_tab_name + ".csv";
    m_is_csv_implem = false;
    warning() << "L'implémentation de ISimpleTableOutput n'utilise pas de fichier de type 'csv'";
  }

}

void SimpleCsvComparatorService::
clear()
{
  m_values_csv.clear();

  m_name_rows.clear();
  m_name_columns_with_name_of_tab.clear();

  m_iSTO = nullptr;
  _closeFile();
  
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
isRefExist(Integer only_proc)
{
  if (only_proc != -1 && mesh()->parallelMng()->commRank() != only_proc)
    return false;

  // On ouvre en lecture le fichier de ref de notre proc.
  _openFile(m_file_name);
  return m_ifstream.good();
}

bool SimpleCsvComparatorService::
readRefFile(Integer only_proc)
{
  if (only_proc != -1 && mesh()->parallelMng()->commRank() != only_proc)
    return false;

  // Pas de fichier, pas de chocolats.
  if(!isRefExist(only_proc)) {
    m_is_file_read = false;
    return false;
  }

  std::string line;

  // S'il n'y a pas de première ligne, on arrete là.
  // Un fichier écrit par SimpleCsvOutput possède toujours au
  // moins une ligne.
  if(!std::getline(m_ifstream, line)) {
    m_is_file_read = false;
    _closeFile();
    return false;
  }

  // Sinon, on a la ligne d'entête, contenant les noms
  // des colonnes (et le nom du tableau).
  String ligne(line);
  ligne.split(m_name_columns_with_name_of_tab, ';');

  // S'il n'y a pas d'autres lignes, c'est qu'il n'y a que des 
  // colonnes vides (ou aucunes colonnes) et aucunes lignes.
  if(!std::getline(m_ifstream, line)) {
    m_is_file_read = true;
    _closeFile();
    return true;
  }

  // Maintenant que l'on a le nombre de colonnes, on peut définir
  // la dimension 2 du tableau de valeurs.
  m_values_csv.resize(1, m_name_columns_with_name_of_tab.size()-1);

  Integer compt_line = 0;

  do{
    // On n'a pas le nombre de lignes en avance,
    // donc on doit resize à chaque tour.
    m_values_csv.resize(compt_line+1);

    // On split la ligne récupéré.
    StringUniqueArray splitted_line;
    String ligne(line);
    ligne.split(splitted_line, ';');

    // Le premier élement est le nom de ligne.
    m_name_rows.add(splitted_line[0]);

    // Les autres élements sont des Reals.
    for(Integer i = 1; i < splitted_line.size(); i++){
      m_values_csv[compt_line][i-1] = std::stod(splitted_line[i].localstr());
    }

    compt_line++;
  } while(std::getline(m_ifstream, line));

  m_is_file_read = true;
  _closeFile();
  return true;
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

      const Real val1 = m_iSTO->elem(m_name_columns_with_name_of_tab[j+1], m_name_rows[i]);
      const Real val2 = view[j];

      if(!math::isNearlyEqualWithEpsilon(val1, val2, epsilon)) {
        warning() << "Values not equals -- Column name: \"" << m_name_columns_with_name_of_tab[j+1] << "\" -- Row name: \"" << m_name_rows[i] << "\"";
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

void SimpleCsvComparatorService::
_openFile(String name_file)
{
  if(m_is_file_open) return;
  m_ifstream.open(m_ref_path.file(name_file).localstr(), std::ifstream::in);
  m_is_file_open = true;
}

void SimpleCsvComparatorService::
_closeFile()
{
  if(!m_is_file_open) return;
  m_ifstream.close();
  m_is_file_open = false;
}

bool SimpleCsvComparatorService::
_exploreColumn(Integer pos)
{
  // S'il n'y a pas de précisions, on compare toutes les colonnes.
  if(m_compared_columns.empty() && m_regex_columns.empty()) {
    return true;
  }

  const String name_column = m_name_columns_with_name_of_tab[pos+1];

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
