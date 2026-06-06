// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ISimpleTableWriterHelper.h                                  (C) 2000-2025 */
/*                                                                           */
/* Interface representing a simple writer using an                           */
/* ISimpleTableReaderWriter.                                                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ISIMPLETABLEWRITERHELPER_H
#define ARCANE_CORE_ISIMPLETABLEWRITERHELPER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ISimpleTableInternalMng.h"
#include "arcane/core/ISimpleTableReaderWriter.h"

#include "arcane/core/Directory.h"
#include "arcane/core/ItemTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/**
 * @brief Class interface for writing a file
 * using ISimpleTableReaderWriter.
 * Provides methods for managing parallel writing
 * and name symbols.
 * 
 * This class is, in a way, a wrapper around 
 * ISimpleTableReaderWriter, which is quite basic.
 * ISimpleTableWriterHelper is here to simplify
 * the use of ISimpleTableReaderWriter.
 * 
 * In the SimpleTable part, name symbols are keywords surrounded by 
 * at signs (@) that will be replaced by their meaning during execution.
 * In the SimpleTableWriterHelper implementation, there are
 * currently two supported name symbols:
 * - \@proc_id\@: Will be replaced by the process ID.
 * - \@num_procs\@: Will be replaced by the number of processes.
 * And in SimpleTableWriterHelper, these symbols are only replaced
 * in the table name.
 */
class ARCANE_CORE_EXPORT ISimpleTableWriterHelper
{
 public:
  virtual ~ISimpleTableWriterHelper() = default;

 public:
  /**
   * @brief Method to initialize the object.
   * Specifically, the table name and the directory name that will contain
   * the files (the tables directory/directory_name).
   * 
   * @param table_name The table name (and the output file name).
   * @param directory_name The folder name where the tables will be saved.
   */
  virtual bool init(const Directory& root_directory, const String& table_name, const String& directory_name) = 0;

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /**
   * @brief Method to display the table.
   * 
   * @param rank The process ID that should display the table (-1 to 
   *                  signify "all processes").
   */
  virtual void print(Integer rank = 0) = 0;

  /**
   * @brief Method to write the table to a file.
   * If rank != -1, processes other than rank return true.
   * 
   * For example, in the SimpleTableWriterHelper implementation,
   * the file(s) will be written in the directory:
   * root_directory/[directory_name]/[table_name].[ISimpleTableReaderWriter.fileType()]
   * 
   * @param root_directory The root directory where the tables directory should be created.
   * @param rank The process ID that should write the table to a file 
   *                  (-1 to signify "all processes").
   * @return true If the file was written correctly.
   * @return false If the file was not written correctly.
   */
  virtual bool writeFile(const Directory& root_directory, Integer rank) = 0;

  /**
   * @brief Method to write the table to a file.
   * If rank != -1, processes other than rank return true.
   * 
   * For example, in the SimpleTableWriterHelper implementation,
   * the file(s) will be written in the directory:
   * ./[output]/[directory_name]/[table_name].[ISimpleTableReaderWriter.fileType()]
   * 
   * @param rank The process ID that should write the table to a file 
   *                  (-1 to signify "all processes").
   * @return true If the file was written correctly.
   * @return false If the file was not written correctly.
   */
  virtual bool writeFile(Integer rank = -1) = 0;

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /**
   * @brief Method to retrieve the precision currently
   * used for writing values.
   * 
   * @return Integer The precision.
   */
  virtual Integer precision() = 0;
  /**
   * @brief Method to modify the print precision.
   * 
   * Applicable to both the 'print()' method and the 'writeFile()' methods.
   * 
   * @warning The "std::fixed" flag modifies the behavior of "setPrecision()";
   *          if the "std::fixed" flag is disabled, the precision defines the
   *          total number of digits (before and after the comma);
   *          if the "std::fixed" flag is enabled, the precision defines the
   *          number of digits after the comma. Be careful when using
   *          "std::numeric_limits<Real>::max_digits10"
   *          (for writing) or "std::numeric_limits<Real>::digits10"
   *          (for reading), which should be used without the "std::fixed" flag.
   * 
   * @param precision The new precision.
   */
  virtual void setPrecision(Integer precision) = 0;

  /**
   * @brief Method to check if the 'std::fixed' flag is
   * active for writing values.
   * 
   * @return true If yes.
   * @return false If no.
   */
  virtual bool isFixed() = 0;
  /**
   * @brief Method to set or unset the 'std::fixed' flag.
   * 
   * Applicable to both the 'print()' method and the 'writeFile()' methods.
   * 
   * This flag allows 'forcing' the number of digits after the comma to
   * the desired precision. For example, if 'setPrecision(4)' was called,
   * and 'setFixed(true)' is called, the print of '6.1' will output '6.1000'.
   * 
   * @warning The "std::fixed" flag modifies the behavior of "setPrecision()";
   *          if the "std::fixed" flag is disabled, the precision defines the
   *          total number of digits (before and after the comma);
   *          if the "std::fixed" flag is enabled, the precision defines the
   *          number of digits after the comma. Be careful when using
   *          "std::numeric_limits<Real>::max_digits10"
   *          (for writing) or "std::numeric_limits<Real>::digits10"
   *          (for reading), which should be used without the "std::fixed" flag.
   * 
   * @param fixed Whether the 'std::fixed' flag should be set or not.
   */
  virtual void setFixed(bool fixed) = 0;

  /**
   * @brief Method to check if the 'std::scientific' flag is
   * active for writing values.
   * 
   * @return true If yes.
   * @return false If no.
   */
  virtual bool isForcedToUseScientificNotation() = 0;
  /**
   * @brief Method to set or unset the 'std::scientific' flag.
   * 
   * Applicable to both the 'print()' method and the 'writetable()' method.
   * 
   * This flag allows 'forcing' the display of values in scientific
   * notation.
   * 
   * @param use_scientific Whether the 'std::scientific' flag should be set or not.
   */
  virtual void setForcedToUseScientificNotation(bool use_scientific) = 0;

  /**
   * @brief Method to retrieve the directory name as it
   * was previously provided.
   * 
   * Name symbols are still present here.
   * 
   * @return String The directory.
   */
  virtual String outputDirectoryWithoutComputation() = 0;

  /**
   * @brief Method to retrieve the directory name where the tables will be
   * placed.
   * 
   * May be different for each process (depending on the implementation).
   * 
   * Name symbols have been resolved here.
   * 
   * @return String The directory.
   */
  virtual String outputDirectory() = 0;
  /**
   * @brief Method to set the directory
   * where the tables should be saved.
   * 
   * May be different for each process (depending on the implementation).
   * 
   * @param directory The directory.
   */
  virtual void setOutputDirectory(const String& directory) = 0;

  /**
   * @brief Method to retrieve the table name as it
   * was previously provided.
   * 
   * Name symbols are still present here.
   * 
   * @return String The name.
   */
  virtual String tableNameWithoutComputation() = 0;

  /**
   * @brief Method to retrieve the table name.
   * 
   * May be different for each process (depending on the implementation).
   * 
   * Name symbols have been resolved here.
   * 
   * @return String The name.
   */
  virtual String tableName() = 0;
  /**
   * @brief Method to set the table name.
   * 
   * @param name The name.
   */
  virtual void setTableName(const String& name) = 0;

  /**
   * @brief Method to retrieve the file name.
   * 
   * May be different for each process (depending on the implementation).
   * 
   * Name symbols have been resolved and the extension is added here.
   * 
   * @return String The name.
   */
  virtual String fileName() = 0;

  /**
   * @brief Method to retrieve the path where
   * the tables will be saved. 
   * 
   * Example (relative):
   * ./output/csv/[directory_name]/
   * 
   * @return String The path.
   */
  virtual Directory outputPath() = 0;

  /**
   * @brief Method to retrieve the path where the implementation
   * saves these tables. 
   * 
   * Example (relative):
   * ./output/csv/
   * 
   * @return String The path.
   */
  virtual Directory rootPath() = 0;

  /**
   * @brief Method to check if the parameters currently held
   * by the implementation allow it to write a file per process, especially 
   * thanks to name symbols.
   * 
   * @return true If yes, the implementation can write a file per process.
   * @return false Otherwise, only one file can be written.
   */
  virtual bool isOneFileByRanksPermited() = 0;

  /**
   * @brief Method to know the file type that will be used.
   * 
   * @return String The output file type (= the extension).
   */
  virtual String fileType() = 0;

  /**
   * @brief Method to retrieve a reference to the
   * SimpleTableInternal object used.
   * 
   * @return Ref<SimpleTableInternal> A copy of the reference. 
   */
  virtual Ref<SimpleTableInternal> internal() = 0;

  /**
   * @brief Method to retrieve a reference to the
   * ISimpleTableReaderWriter object used.
   * 
   * @return Ref<ISimpleTableReaderWriter> A copy of the reference. 
   */
  virtual Ref<ISimpleTableReaderWriter> readerWriter() = 0;

  /**
   * @brief Method to set a reference to an
   * ISimpleTableReaderWriter.
   * 
   * @param simple_table_reader_writer The reference to an ISimpleTableReaderWriter.
   */
  virtual void setReaderWriter(const Ref<ISimpleTableReaderWriter>& simple_table_reader_writer) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
