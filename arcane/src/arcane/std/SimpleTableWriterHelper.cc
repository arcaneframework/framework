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

#include "arcane/std/SimpleTableWriterHelper.h"

#include "arcane/Directory.h"
#include "arcane/IParallelMng.h"
#include "arcane/utils/StringBuilder.h"

#include <optional>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool SimpleTableWriterHelper::
init()
{
  return init("Table_@proc_id@");
}

bool SimpleTableWriterHelper::
init(const String& table_name)
{
  return init(table_name, "");
}

bool SimpleTableWriterHelper::
init(const String& table_name, const String& directory_name)
{
  ARCANE_CHECK_PTR(m_simple_table_internal);

  setTableName(table_name);
  _computeName();

  m_name_output_directory = directory_name;

  m_root = Directory(m_simple_table_internal->m_sub_domain->exportDirectory(), m_simple_table_reader_writer->fileType());
  return true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SimpleTableWriterHelper::
print(Integer process_id)
{
  ARCANE_CHECK_PTR(m_simple_table_reader_writer);
  if (process_id != -1 && m_simple_table_internal->m_sub_domain->parallelMng()->commRank() != process_id)
    return;
  m_simple_table_reader_writer->print();
}

bool SimpleTableWriterHelper::
writeFile(const Directory& root_directory, Integer process_id)
{
  ARCANE_CHECK_PTR(m_simple_table_internal);
  ARCANE_CHECK_PTR(m_simple_table_reader_writer);
  // Finalisation du nom du csv (si ce n'est pas déjà fait).
  _computeName();

  // Création du répertoire.
  bool result = SimpleTableReaderWriterUtils::createDirectoryOnlyProcess0(m_simple_table_internal->m_sub_domain, root_directory);
  if (!result) {
    return false;
  }

  // Si l'on n'est pas le processus demandé, on return true.
  // -1 = tout le monde écrit.
  if (process_id != -1 && m_simple_table_internal->m_sub_domain->parallelMng()->commRank() != process_id)
    return true;

  // Si l'on a process_id == -1 et que m_simple_table_internal->m_name_table_once_process == true, alors il n'y a que le
  // processus 0 qui doit écrire.
  if ((process_id == -1 && m_name_table_once_process) && m_simple_table_internal->m_sub_domain->parallelMng()->commRank() != 0)
    return true;

  return m_simple_table_reader_writer->writeTable(Directory(root_directory, m_name_output_directory), m_simple_table_internal->m_table_name);
}

bool SimpleTableWriterHelper::
writeFile(Integer process_id)
{
  return writeFile(m_root, process_id);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer SimpleTableWriterHelper::
precision()
{
  ARCANE_CHECK_PTR(m_simple_table_reader_writer);
  return m_simple_table_reader_writer->precision();
}

void SimpleTableWriterHelper::
setPrecision(Integer precision)
{
  ARCANE_CHECK_PTR(m_simple_table_reader_writer);
  m_simple_table_reader_writer->setPrecision(precision);
}

bool SimpleTableWriterHelper::
isFixed()
{
  ARCANE_CHECK_PTR(m_simple_table_reader_writer);
  return m_simple_table_reader_writer->isFixed();
}

void SimpleTableWriterHelper::
setFixed(bool fixed)
{
  ARCANE_CHECK_PTR(m_simple_table_reader_writer);
  m_simple_table_reader_writer->setFixed(fixed);
}

String SimpleTableWriterHelper::
outputDirectory()
{
  return m_name_output_directory;
}

String SimpleTableWriterHelper::
outputDirectoryWithoutComputation()
{
  return m_name_output_directory;
}

void SimpleTableWriterHelper::
setOutputDirectory(const String& directory)
{
  m_name_output_directory = directory;
}

String SimpleTableWriterHelper::
tableName()
{
  ARCANE_CHECK_PTR(m_simple_table_internal);
  _computeName();
  return m_simple_table_internal->m_table_name;
}

String SimpleTableWriterHelper::
tableNameWithoutComputation()
{
  return m_name_table_without_computation;
}

void SimpleTableWriterHelper::
setTableName(const String& name)
{
  ARCANE_CHECK_PTR(m_simple_table_internal);
  m_simple_table_internal->m_table_name = name;
  m_name_table_without_computation = name;
  m_name_table_computed = false;
}

String SimpleTableWriterHelper::
fileName()
{
  ARCANE_CHECK_PTR(m_simple_table_internal);
  _computeName();
  return m_simple_table_internal->m_table_name + "." + m_simple_table_reader_writer->fileType();
}

Directory SimpleTableWriterHelper::
outputPath()
{
  return Directory(m_root, m_name_output_directory);
}

Directory SimpleTableWriterHelper::
rootPath()
{
  return m_root;
}

bool SimpleTableWriterHelper::
isOneFileByProcsPermited()
{
  _computeName();
  return !m_name_table_once_process;
}

String SimpleTableWriterHelper::
fileType()
{
  ARCANE_CHECK_PTR(m_simple_table_reader_writer);
  return m_simple_table_reader_writer->fileType();
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
void SimpleTableWriterHelper::
_computeName()
{
  if (m_name_table_computed) {
    return;
  }

  ARCANE_CHECK_PTR(m_simple_table_internal);
  ARCANE_CHECK_PTR(m_simple_table_reader_writer);

  // Permet de contourner le bug avec String::split() si le nom commence par '@'.
  if (m_simple_table_internal->m_table_name.startsWith("@")) {
    m_simple_table_internal->m_table_name = "@" + m_simple_table_internal->m_table_name;
  }

  StringUniqueArray string_splited;
  // On découpe la string là où se trouve les @.
  m_simple_table_internal->m_table_name.split(string_splited, '@');

  // On traite les mots entre les "@".
  if (string_splited.size() > 1) {
    // On recherche "proc_id" dans le tableau (donc @proc_id@ dans le nom).
    std::optional<Integer> proc_id = string_splited.span().findFirst("proc_id");
    // On remplace "@proc_id@" par l'id du proc.
    if (proc_id) {
      string_splited[proc_id.value()] = String::fromNumber(m_simple_table_internal->m_sub_domain->parallelMng()->commRank());
      m_name_table_once_process = false;
    }
    // Il n'y a que un seul proc qui write.
    else {
      m_name_table_once_process = true;
    }

    // On recherche "num_procs" dans le tableau (donc @num_procs@ dans le nom).
    std::optional<Integer> num_procs = string_splited.span().findFirst("num_procs");
    // On remplace "@num_procs@" par l'id du proc.
    if (num_procs) {
      string_splited[num_procs.value()] = String::fromNumber(m_simple_table_internal->m_sub_domain->parallelMng()->commSize());
    }
  }

  // On recombine la chaine.
  StringBuilder combined = "";
  for (String str : string_splited) {
    // Permet de contourner le bug avec String::split() s'il y a '@@@' dans le nom ou si le
    // nom commence par '@' (en complément des premières lignes de la méthode).
    if (str == "@")
      continue;
    combined.append(str);
  }

  m_simple_table_internal->m_table_name = combined.toString();

  m_name_table_computed = true;
  return;
}

SimpleTableInternal* SimpleTableWriterHelper::
internal()
{
  return m_simple_table_internal;
}

ISimpleTableReaderWriter* SimpleTableWriterHelper::
readerWriter()
{
  return m_simple_table_reader_writer;
}

void SimpleTableWriterHelper::
setReaderWriter(ISimpleTableReaderWriter* simple_table_reader_writer)
{
  ARCANE_CHECK_PTR(m_simple_table_reader_writer);
  m_simple_table_reader_writer = simple_table_reader_writer;
  m_simple_table_internal = m_simple_table_reader_writer->internal();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
