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

#include "arcane/std/SimpleTableOutputMng.h"

#include <arcane/Directory.h>
#include <arcane/IMesh.h>
#include <arcane/IParallelMng.h>
#include <arcane/ISubDomain.h>
#include "arcane/utils/StringBuilder.h"

#include <optional>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool SimpleTableOutputMng::
init()
{
  return init("Table_@proc_id@");
}

bool SimpleTableOutputMng::
init(String name_table)
{
  return init(name_table, "");
}

bool SimpleTableOutputMng::
init(String name_table, String name_dir)
{
  m_sti->m_name_tab = name_table;
  _computeName();

  m_name_output_dir = name_dir;

  m_root = Directory(m_sti->m_mesh->subDomain()->exportDirectory(), m_strw->typeFile());
  return true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SimpleTableOutputMng::
print(Integer only_proc)
{
  m_strw->print(only_proc);
}

bool SimpleTableOutputMng::
writeFile(Directory root_dir, Integer only_proc)
{
  // Finalisation du nom du csv (si ce n'est pas déjà fait).
  _computeName();

  // Création du répertoire.
  bool result = SimpleTableReaderWriterUtils::createDirectoryOnlyP0(m_sti->m_mesh, root_dir);
  if(!result) {
    return false;
  }

  // Si l'on n'est pas le processus demandé, on return true.
  // -1 = tout le monde écrit.
  if (only_proc != -1 && m_sti->m_mesh->parallelMng()->commRank() != only_proc)
    return true;

  // Si l'on a only_proc == -1 et que m_sti->m_name_tab_only_once == true, alors il n'y a que le
  // processus 0 qui doit écrire.
  if ((only_proc == -1 && m_name_tab_only_once) && m_sti->m_mesh->parallelMng()->commRank() != 0)
    return true;

  return m_strw->write(Directory(root_dir, m_name_output_dir), m_sti->m_name_tab);
}

bool SimpleTableOutputMng::
writeFile(Integer only_proc)
{
  return writeFile(m_root, only_proc);
}

bool SimpleTableOutputMng::
writeFile(String dir, Integer only_proc)
{
  setOutputDir(dir);
  return writeFile(m_root, only_proc);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer SimpleTableOutputMng::
precision()
{
  return m_strw->precision();
}

void SimpleTableOutputMng::
setPrecision(Integer precision)
{
  m_strw->setPrecision(precision);

}

bool SimpleTableOutputMng::
fixed()
{
  return m_strw->fixed();
}

void SimpleTableOutputMng::
setFixed(bool fixed)
{
  m_strw->setFixed(fixed);
}


String SimpleTableOutputMng::
outputDir()
{
  return m_name_output_dir;
}

void SimpleTableOutputMng::
setOutputDir(String dir)
{
  m_name_output_dir = dir;
}

String SimpleTableOutputMng::
tabName()
{
  _computeName();
  return m_sti->m_name_tab;
}

void SimpleTableOutputMng::
setTabName(String name)
{
  m_sti->m_name_tab = name;
  m_name_tab_computed = false;
}

String SimpleTableOutputMng::
fileName()
{
  _computeName();
  return m_name_file;
}

Directory SimpleTableOutputMng::
outputPath()
{
  return Directory(m_root, m_name_output_dir);
}

Directory SimpleTableOutputMng::
rootPath()
{
  return m_root;
}

bool SimpleTableOutputMng::
isOneFileByProcsPermited()
{
  _computeName();
  return !m_name_tab_only_once;
}

String SimpleTableOutputMng::
outputFileType()
{
  return m_strw->typeFile();
}


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/**
 * @brief Méthode permettant de remplacer les symboles de nom par leur valeur.
 * 
 * @param name [IN] Le nom à modifier.
 * @param only_once [OUT] Si le nom contient le symbole '\@proc_id\@' permettant 
 *                de différencier les fichiers écrits par differents processus.
 * @return String Le nom avec les symboles remplacés.
 */
void SimpleTableOutputMng::
_computeName()
{
  if(m_name_tab_computed){
    return;
  }

  ARCANE_CHECK_PTR(m_sti);
  ARCANE_CHECK_PTR(m_strw);

  std::cout << "-------------------------1" << std::endl;

  // Permet de contourner le bug avec String::split() si le nom commence par '@'.
  if (m_sti->m_name_tab.startsWith("@")) {
    m_sti->m_name_tab = "@" + m_sti->m_name_tab;
  }
  std::cout << "-------------------------2" << std::endl;

  StringUniqueArray string_splited;
  // On découpe la string là où se trouve les @.
  m_sti->m_name_tab.split(string_splited, '@');
  std::cout << "-------------------------3" << std::endl;

  // On traite les mots entre les "@".
  if (string_splited.size() > 1) {
    // On recherche "proc_id" dans le tableau (donc @proc_id@ dans le nom).
    std::optional<Integer> proc_id = string_splited.span().findFirst("proc_id");
    // On remplace "@proc_id@" par l'id du proc.
    if (proc_id) {
      string_splited[proc_id.value()] = String::fromNumber(m_sti->m_mesh->parallelMng()->commRank());
      m_name_tab_only_once = false;
    }
    // Il n'y a que un seul proc qui write.
    else {
      m_name_tab_only_once = true;
    }

    // On recherche "num_procs" dans le tableau (donc @num_procs@ dans le nom).
    std::optional<Integer> num_procs = string_splited.span().findFirst("num_procs");
    // On remplace "@num_procs@" par l'id du proc.
    if (num_procs) {
      string_splited[num_procs.value()] = String::fromNumber(m_sti->m_mesh->parallelMng()->commSize());
    }
  }
  std::cout << "-------------------------4" << std::endl;

  // On recombine la chaine.
  StringBuilder combined = "";
  for (String str : string_splited) {
    // Permet de contourner le bug avec String::split() s'il y a '@@@' dans le nom ou si le
    // nom commence par '@' (en complément des premières lignes de la méthode).
    if (str == "@")
      continue;
    combined.append(str);
  }
  std::cout << "-------------------------5" << std::endl;

  m_sti->m_name_tab = combined.toString();
  combined.append(".");
  combined.append(m_strw->typeFile());
  std::cout << "-------------------------6" << std::endl;
  m_name_file = combined.toString();

  m_name_tab_computed = true;
  return;
}

SimpleTableInternal* SimpleTableOutputMng::
internal() 
{
  return m_sti;
}

void SimpleTableOutputMng::
setInternal(SimpleTableInternal* sti) 
{
  ARCANE_CHECK_PTR(sti);
  m_sti = sti;
}

void SimpleTableOutputMng::
setInternal(SimpleTableInternal& sti) 
{
  m_sti = &sti;
}

ISimpleTableReaderWriter* SimpleTableOutputMng::
readerWriter() 
{
  return m_strw;
}

void SimpleTableOutputMng::
setReaderWriter(ISimpleTableReaderWriter* strw) 
{
  ARCANE_CHECK_PTR(strw);
  m_strw = strw;
}

void SimpleTableOutputMng::
setReaderWriter(ISimpleTableReaderWriter& strw) 
{
  m_strw = &strw;
}


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
