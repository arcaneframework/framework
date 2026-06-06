// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ISimpleTableReaderWriter.h                                  (C) 2000-2025 */
/*                                                                           */
/* Interface representing a simple table reader/writer.                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ISIMPLETABLEREADERWRITER_H
#define ARCANE_CORE_ISIMPLETABLEREADERWRITER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ISimpleTableInternalMng.h"

#include "arcane/core/Directory.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/ItemTypes.h"
#include "arcane/utils/Iostream.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/**
 * @brief Class containing two static methods
 * useful for implementations.
 * 
 */
class ARCANE_CORE_EXPORT SimpleTableReaderWriterUtils
{
 public:
  /**
   * @brief Static method allowing the creation of a directory with multiple
   * processes.
   * @note This is a collective method that must be called by all processes.
   * 
   * @param parallel_mng The parallel manager of the current context.
   * @param directory The directory to create.
   * @return true If the directory was successfully created.
   * @return false If the directory could not be created.
   */
  static bool createDirectoryOnlyProcess0(IParallelMng* parallel_mng, const Directory& directory)
  {
    int sf = 0;
    if (parallel_mng->commRank() == 0) {
      sf = directory.createDirectory();
    }
    if (parallel_mng->commSize() > 1) {
      sf = parallel_mng->reduce(Parallel::ReduceMax, sf);
    }
    return sf == 0;
  };

  /**
   * @brief Static method allowing verification of file existence.
   * 
   * @param directory The directory where the file is located.
   * @param file The file name (with extension).
   * @return true If the file already exists.
   * @return false If the file does not exist.
   */
  static bool isFileExist(const Directory& directory, const String& file)
  {
    std::ifstream stream;
    stream.open(directory.file(file).localstr(), std::ifstream::in);
    bool fin = stream.good();
    stream.close();
    return fin;
  };
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/**
 * @brief Class interface allowing reading a file and writing
 * a file with or from a SimpleTableInternal.
 * 
 * The file read must preferably have been written by an implementation 
 * of this same interface.
 * 
 * Imperatively, a file written by an implementation of this
 * interface must be readable by this same implementation.
 * 
 * The implementation must not destroy the SimpleTableInternal object
 * pointed to by the pointer used. The caller is responsible for managing this.
 */
class ARCANE_CORE_EXPORT ISimpleTableReaderWriter
{
 public:
  virtual ~ISimpleTableReaderWriter() = default;

 public:
  /**
   * @brief Method allowing writing a simple table to a file.
   * 
   * The extension will be added by the implementation.
   * 
   * The destination directory will be created by the implementation if
   * it does not exist.
   * 
   * The SimpleTableInternal elements that must be written are:
   * - row names (m_row_names),
   * - column names (m_column_names),
   * - table name (m_table_name),
   * - table values (m_values).
   * 
   * Other SimpleTableInternal elements are not mandatory.
   * 
   * @param dst The destination directory.
   * @param file_name The file name (without extension).
   * @return true If the file was successfully written.
   * @return false If the file could not be written.
   */
  virtual bool writeTable(const Directory& dst, const String& file_name) = 0;

  /**
   * @brief Method allowing reading a file containing a simple table.
   * 
   * The extension will be added by the implementation.
   * 
   * A call to SimpleTableInternal::clear() must be performed before reading.
   * 
   * The elements that must be retrieved are:
   * - row names (m_row_names),
   * - column names (m_column_names),
   * - table name (m_table_name),
   * - table values (m_values).
   * 
   * The elements that must be deduced if not retrieved are:
   * - row sizes (m_row_sizes),
   * - column sizes (m_column_sizes).
   * 
   * Default deduction for m_row_sizes:
   * - len(m_row_sizes) = len(m_row_names)
   * - m_row_sizes[*]   = m_values.dim2Size()
   * 
   * Default deduction for m_column_sizes:
   * - len(m_column_sizes) = len(m_column_names)
   * - m_column_sizes[*]   = m_values.dim1Size()
   * 
   * 
   * @param src The source directory.
   * @param file_name The file name (without extension).
   * @return true If the file was successfully read.
   * @return false If the file could not be read.
   */
  virtual bool readTable(const Directory& src, const String& file_name) = 0;

  /**
   * @brief Method allowing clearing the content of the
   * SimpleTableInternal object.
   */
  virtual void clearInternal() = 0;

  /**
   * @brief Method allowing writing the table to the
   * standard output.
   * 
   * The writing format is free (for the csv implementation, the
   * writing is done the same way as in a csv file).
   */
  virtual void print() = 0;

  /**
   * @brief Method allowing retrieval of the precision currently
   * used for writing values.
   * 
   * @return Integer The precision.
   */
  virtual Integer precision() = 0;

  /**
   * @brief Method allowing modification of the print precision.
   * 
   * For both the 'print()' method and the 'writetable()' method.
   * 
   * @warning The "std::fixed" flag modifies the behavior of "setPrecision()";
   *          if the "std::fixed" flag is disabled, the precision defines the
   *          total number of digits (before and after the comma);
   *          if the "std::fixed" flag is enabled, the precision defines the
   *          number of digits after the comma. Therefore, attention must be paid when
   *          using "std::numeric_limits<Real>::max_digits10"
   *          (for writing) or "std::numeric_limits<Real>::digits10"
   *          (for reading), which should be used without the "std::fixed" flag.
   * 
   * @param precision The new precision.
   */
  virtual void setPrecision(Integer precision) = 0;

  /**
   * @brief Method allowing checking if the 'std::fixed' flag is
   * active or not for writing values.
   * 
   * @return true If yes.
   * @return false If no.
   */
  virtual bool isFixed() = 0;
  /**
   * @brief Method allowing setting or unsetting the 'std::fixed' flag.
   * 
   * For both the 'print()' method and the 'writetable()' method.
   * 
   * This flag allows 'forcing' the number of digits after the comma to
   * the desired precision. For example, if 'setPrecision(4)' was called,
   * and 'setFixed(true)' is called, the print of '6.1' will yield '6.1000'.
   * 
   * @warning The "std::fixed" flag modifies the behavior of "setPrecision()";
   *          if the "std::fixed" flag is disabled, the precision defines the
   *          total number of digits (before and after the comma);
   *          if the "std::fixed" flag is enabled, the precision defines the
   *          number of digits after the comma. Therefore, attention must be paid when
   *          using "std::numeric_limits<Real>::max_digits10"
   *          (for writing) or "std::numeric_limits<Real>::digits10"
   *          (for reading), which should be used without the "std::fixed" flag.
   * 
   * @param fixed Whether the 'std::fixed' flag should be set or not.
   */
  virtual void setFixed(bool fixed) = 0;

  /**
   * @brief Method allowing checking if the 'std::scientific' flag is
   * active or not for writing values.
   * 
   * @return true If yes.
   * @return false If no.
   */
  virtual bool isForcedToUseScientificNotation() = 0;
  /**
   * @brief Method allowing setting or unsetting the 'std::scientific' flag.
   * 
   * For both the 'print()' method and the 'writetable()' method.
   * 
   * This flag allows 'forcing' the display of values in scientific
   * notation during writing.
   * 
   * @param use_scientific Whether the 'std::scientific' flag should be set or not.
   */
  virtual void setForcedToUseScientificNotation(bool use_scientific) = 0;

  /**
   * @brief Method allowing retrieval of the file type
   * that will be written by the implementation. ("csv" will be returned
   * for the csv implementation).
   * 
   * @return String The file type/extension used.
   */
  virtual String fileType() = 0;

  /**
   * @brief Method allowing retrieval of a reference to the
   * SimpleTableInternal object used.
   * 
   * @return Ref<SimpleTableInternal> A copy of the reference. 
   */
  virtual Ref<SimpleTableInternal> internal() = 0;

  /**
   * @brief Method allowing setting a reference to a
   * SimpleTableInternal.
   * 
   * @param simple_table_internal The reference to a SimpleTableInternal.
   */
  virtual void setInternal(const Ref<SimpleTableInternal>& simple_table_internal) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
