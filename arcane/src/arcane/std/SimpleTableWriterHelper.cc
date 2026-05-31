// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SimpleTableWriterHelper.cc                                  (C) 2000-2023 */
/*                                                                           */
/* Class allowing writing a SimpleTableInternal to a file.                   */
/* Simplifies the use of the writer by managing multiprocess and the names   */
/* of files/directories.                                                     */
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
init(const Directory& root_directory, const String& table_name, const String& directory_name)
{
  setTableName(table_name);
  _computeTableName();

  setOutputDirectory(directory_name);
  _computeOutputDirectory();

  m_root = Directory(root_directory, m_simple_table_reader_writer->fileType());
  return true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SimpleTableWriterHelper::
print(Integer rank)
{
  if (rank != -1 && m_simple_table_internal->m_parallel_mng->commRank() != rank)
    return;
  m_simple_table_reader_writer->print();
}

/**
 * Method performing collective operations.
 */
bool SimpleTableWriterHelper::
writeFile(const Directory& root_directory, Integer rank)
{
  // Finalizing the name and directory of the csv (if not already done).
  _computeTableName();
  _computeOutputDirectory();

  // Creating the directory.
  bool result = SimpleTableReaderWriterUtils::createDirectoryOnlyProcess0(m_simple_table_internal->m_parallel_mng, root_directory);
  if (!result) {
    return false;
  }

  Directory output_directory(root_directory, m_name_output_directory);

  result = SimpleTableReaderWriterUtils::createDirectoryOnlyProcess0(m_simple_table_internal->m_parallel_mng, output_directory);
  if (!result) {
    return false;
  }

  // If we are not the requested process, we return true.
  // -1 = everyone writes.
  if (rank != -1 && m_simple_table_internal->m_parallel_mng->commRank() != rank)
    return true;

  // If rank == -1 and isOneFileByRanksPermited() == false, then only process 0
  // must write.
  if ((rank == -1 && !isOneFileByRanksPermited()) && m_simple_table_internal->m_parallel_mng->commRank() != 0)
    return true;

  return m_simple_table_reader_writer->writeTable(output_directory, m_simple_table_internal->m_table_name);
}

/**
 * Method performing collective operations.
 */
bool SimpleTableWriterHelper::
writeFile(Integer rank)
{
  return writeFile(m_root, rank);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer SimpleTableWriterHelper::
precision()
{
  return m_simple_table_reader_writer->precision();
}

void SimpleTableWriterHelper::
setPrecision(Integer precision)
{
  m_simple_table_reader_writer->setPrecision(precision);
}

bool SimpleTableWriterHelper::
isFixed()
{
  return m_simple_table_reader_writer->isFixed();
}

void SimpleTableWriterHelper::
setFixed(bool fixed)
{
  m_simple_table_reader_writer->setFixed(fixed);
}

bool SimpleTableWriterHelper::
isForcedToUseScientificNotation()
{
  return m_simple_table_reader_writer->isForcedToUseScientificNotation();
}

void SimpleTableWriterHelper::
setForcedToUseScientificNotation(bool use_scientific)
{
  m_simple_table_reader_writer->setForcedToUseScientificNotation(use_scientific);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String SimpleTableWriterHelper::
outputDirectory()
{
  _computeOutputDirectory();
  return m_name_output_directory;
}

String SimpleTableWriterHelper::
outputDirectoryWithoutComputation()
{
  return m_name_output_directory_without_computation;
}

void SimpleTableWriterHelper::
setOutputDirectory(const String& directory)
{
  m_name_output_directory = directory;
  m_name_output_directory_without_computation = directory;
  m_name_output_directory_computed = false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String SimpleTableWriterHelper::
tableName()
{
  _computeTableName();
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
  m_simple_table_internal->m_table_name = name;
  m_name_table_without_computation = name;
  m_name_table_computed = false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String SimpleTableWriterHelper::
fileName()
{
  _computeTableName();
  return m_simple_table_internal->m_table_name + "." + m_simple_table_reader_writer->fileType();
}

Directory SimpleTableWriterHelper::
outputPath()
{
  _computeOutputDirectory();
  return Directory(m_root, m_name_output_directory);
}

Directory SimpleTableWriterHelper::
rootPath()
{
  return m_root;
}

bool SimpleTableWriterHelper::
isOneFileByRanksPermited()
{
  _computeTableName();
  _computeOutputDirectory();

  return m_name_table_one_file_by_ranks_permited || m_name_output_directory_one_file_by_ranks_permited;
}

String SimpleTableWriterHelper::
fileType()
{
  return m_simple_table_reader_writer->fileType();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<SimpleTableInternal> SimpleTableWriterHelper::
internal()
{
  return m_simple_table_internal;
}

Ref<ISimpleTableReaderWriter> SimpleTableWriterHelper::
readerWriter()
{
  return m_simple_table_reader_writer;
}

void SimpleTableWriterHelper::
setReaderWriter(const Ref<ISimpleTableReaderWriter>& simple_table_reader_writer)
{
  if (simple_table_reader_writer.isNull())
    ARCANE_FATAL("The reference passed as parameter is Null.");
  m_simple_table_reader_writer = simple_table_reader_writer;
  m_simple_table_internal = m_simple_table_reader_writer->internal();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SimpleTableWriterHelper::
_computeTableName()
{
  if (!m_name_table_computed) {
    m_simple_table_internal->m_table_name = _computeName(m_name_table_without_computation, m_name_table_one_file_by_ranks_permited);
    m_name_table_computed = true;
  }
}

void SimpleTableWriterHelper::
_computeOutputDirectory()
{
  if (!m_name_output_directory_computed) {
    m_name_output_directory = _computeName(m_name_output_directory_without_computation, m_name_output_directory_one_file_by_ranks_permited);
    m_name_output_directory_computed = true;
  }
}

/**
 * @brief Method allowing replacement of name symbols by their value.
 * 
 * @param name [IN] The name to modify.
 * @param one_file_by_ranks_permited [OUT] True if the name contains the symbol '\@proc_id\@'
 *                                   allowing differentiation of files written by
 *                                   different processes.
 * @return String The name with the symbols replaced.
 */
String SimpleTableWriterHelper::
_computeName(String name, bool& one_file_by_ranks_permited)
{
  one_file_by_ranks_permited = false;

  // Allows bypassing the bug with String::split() if the name starts with '@'.
  if (name.startsWith("@")) {
    name = "@" + name;
  }

  StringUniqueArray string_splited;
  // We split the string where the @ symbols are located.
  name.split(string_splited, '@');

  // We process the words between the "@" symbols.
  if (string_splited.size() > 1) {
    // We search for "proc_id" in the array (i.e., @proc_id@ in the name).
    std::optional<Integer> proc_id = string_splited.span().findFirst("proc_id");
    // We replace "@proc_id@" with the process ID.
    if (proc_id) {
      string_splited[proc_id.value()] = String::fromNumber(m_simple_table_internal->m_parallel_mng->commRank());
      one_file_by_ranks_permited = true;
    }
    // Only one process writes.
    else {
      one_file_by_ranks_permited = false;
    }

    // We search for "num_procs" in the array (i.e., @num_procs@ in the name).
    std::optional<Integer> num_procs = string_splited.span().findFirst("num_procs");
    // We replace "@num_procs@" with the process ID.
    if (num_procs) {
      string_splited[num_procs.value()] = String::fromNumber(m_simple_table_internal->m_parallel_mng->commSize());
    }
  }

  // We recombine the chain.
  StringBuilder combined = "";
  for (String str : string_splited) {
    // Allows bypassing the bug with String::split() if there is '@@@' in the name or if the
    // name starts with '@' (in addition to the first lines of the method).
    if (str == "@")
      continue;
    combined.append(str);
  }

  return combined.toString();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
