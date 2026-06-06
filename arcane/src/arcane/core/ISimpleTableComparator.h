// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ISimpleTableComparator.h                                    (C) 2000-2025 */
/*                                                                           */
/* Interface for services allowing the comparison of an ISimpleTableOutput   */
/* and a reference file.                                                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ISIMPLETABLECOMPARATOR_H
#define ARCANE_CORE_ISIMPLETABLECOMPARATOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ISimpleTableOutput.h"

#include "arcane/core/ItemTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/**
 * @ingroup StandardService
 * @brief Class interface representing a table comparator. To be used with
 * a service implementing ISimpleTableOutput.
 * 
 * The difference with ISimpleTableInternalComparator is
 * that we compare a SimpleTableInternal contained in
 * an ISimpleTableOutput with a SimpleTableInternal
 * generated from a reference file.
 * 
 * This interface also allows generating the reference files
 * using the directory name and
 * the table name of the ISimpleTableOutput, facilitating
 * the process.
 */
class ARCANE_CORE_EXPORT ISimpleTableComparator
{
 public:
  virtual ~ISimpleTableComparator() = default;

 public:
  /**
   * @brief Method allowing the service to be initialized.
   * 
   * The pointer to an ISimpleTableOutput implementation
   * must contain the values to be compared or written as
   * reference values and the destination location of the
   * output files, so that the location of the reference files
   * is automatically determined.
   * 
   * @param simple_table_output_ptr An implementation of ISimpleTableOutput.
   */
  virtual void init(ISimpleTableOutput* simple_table_output_ptr) = 0;

  /**
   * @brief Method allowing the data read by readReferenceFile() to be cleared.
   * @note Clears the comparator's SimpleTableInternal without affecting that of the
   * SimpleTableOutput.
   */
  virtual void clear() = 0;

  /**
   * @brief Method allowing the read table to be displayed.
   * 
   * @param rank The process that must display its table (-1 for all processes).
   */
  virtual void print(Integer rank = 0) = 0;

  /**
   * @brief Method allowing the root directory to be modified.
   * This allows writing or searching for reference files
   * elsewhere than in the directory determined by the implementation.
   * 
   * By default, for the csv implementation, the root directory is:
   * ./output/csv_ref/
   * 
   * @param root_directory The new root directory.
   */
  virtual void editRootDirectory(const Directory& root_directory) = 0;

  /**
   * @brief Method allowing reference files to be written.
   * 
   * @warning (For now), this method uses the object pointed to by 
   *          the pointer given during init(), so the writing will occur 
   *          in the format desired by the ISimpleTableOutput implementation.
   *          If the reading and writing formats do not match,
   *          a call to "compareWithReference()" will necessarily return
   *          false.
   * 
   * @param rank The process that must write its file (-1 for all processes).
   * @return true If the writing was successful (and if calling process != rank).
   * @return false If the writing did not occur.
   */
  virtual bool writeReferenceFile(Integer rank = -1) = 0;

  /**
   * @brief Method allowing reference files to be read.
   * 
   * The type of the reference files must correspond to the implementation
   * of this chosen interface (example: .csv file -> SimpleCsvComparatorService).
   * 
   * @param rank The process that must read its file (-1 for all processes).
   * @return true If the file was read (and if calling process != rank).
   * @return false If the file was not read.
   */
  virtual bool readReferenceFile(Integer rank = -1) = 0;

  /**
   * @brief Method allowing to check if the reference files exist.
   * 
   * @param rank The process that must look for its file (-1 for all processes). 
   * @return true If the file was found (and if calling process != rank).
   * @return false If the file was not found.
   */
  virtual bool isReferenceExist(Integer rank = -1) = 0;

  /**
   * @brief Method allowing the ISimpleTableOutput object to be compared
   * to the reference files.
   * 
   * @param rank The process that must compare its results (-1 for all processes). 
   * @param compare_dimension_too Whether the dimensions of the value tables should also be compared.
   * @return true If there are no differences (and if calling process != rank).
   * @return false If there is at least one difference.
   */
  virtual bool compareWithReference(Integer rank = -1, bool compare_dimension_too = false) = 0;

  /**
   * @brief Method allowing only an element to be compared.
   * Both SimpleTableInternals are represented by Refs,
   * so they are always up to date.
   * This method can be used during calculation, allowing
   * to compare values as the calculation progresses,
   * instead of performing a final comparison at the end (it is
   * still possible to do both).
   * 
   * @param column_name The name of the column where the element is located.
   * @param row_name The name of the row where the element is located.
   * @param rank The process that must compare its results (-1 for all processes). 
   * @return true If the two values are equal.
   * @return false If the two values are different.
   */
  virtual bool compareElemWithReference(const String& column_name, const String& row_name, Integer rank = -1) = 0;

  /**
   * @brief Method allowing a value to be compared with
   * a value from the reference table.
   * This method does not need an internal 'toCompare' 
   * (setInternalToCompare() unnecessary).
   * 
   * @param elem The value to be compared.
   * @param column_name The name of the column where the reference element is located.
   * @param row_name The name of the row where the reference element is located.
   * @param rank The process that must compare its results (-1 for all processes). 
   * @return true If the two values are equal.
   * @return false If the two values are different.
   */
  virtual bool compareElemWithReference(Real elem, const String& column_name, const String& row_name, Integer rank = -1) = 0;

  /**
   * @brief Method allowing a column to be added to the list of columns
   * to be compared.
   * 
   * @param column_name The name of the column to compare.
   * @return true If the name was successfully added.
   * @return false Otherwise.
   */
  virtual bool addColumnForComparing(const String& column_name) = 0;
  /**
   * @brief Method allowing a row to be added to the list of rows
   * to be compared.
   * 
   * @param row_name The name of the row to compare.
   * @return true If the name was successfully added.
   * @return false Otherwise.
   */
  virtual bool addRowForComparing(const String& row_name) = 0;

  /**
   * @brief Method allowing definition whether the array of
   * columns represents the columns to include in the
   * comparison (false/default) or represents the columns
   * to exclude from the comparison (true).
   * 
   * @param is_exclusive true if the columns must be
   *                     excluded.
   */
  virtual void isAnArrayExclusiveColumns(bool is_exclusive) = 0;

  /**
   * @brief Method allowing definition whether the array of
   * rows represents the rows to include in the
   * comparison (false/default) or represents the rows
   * to exclude from the comparison (true).
   * 
   * @param is_exclusive true if the rows must be
   *                     excluded.
   */
  virtual void isAnArrayExclusiveRows(bool is_exclusive) = 0;

  /**
   * @brief Method allowing a regular expression to be added
   * to determine the columns to compare.
   * 
   * @param regex_column The regular expression (ECMAScript format).
   */
  virtual void editRegexColumns(const String& regex_column) = 0;
  /**
   * @brief Method allowing a regular expression to be added
   * to determine the rows to compare.
   * 
   * @param regex_row The regular expression (ECMAScript format).
   */
  virtual void editRegexRows(const String& regex_row) = 0;

  /**
   * @brief Method allowing to request that the regular expression
   * excludes columns instead of including them.
   * 
   * @param is_exclusive If the regular expression is exclusive.
   */
  virtual void isARegexExclusiveColumns(bool is_exclusive) = 0;
  /**
   * @brief Method allowing to request that the regular expression
   * excludes rows instead of including them.
   * 
   * @param is_exclusive If the regular expression is exclusive.
   */
  virtual void isARegexExclusiveRows(bool is_exclusive) = 0;

  /**
   * @brief Method allowing an epsilon to be defined for a given column.
   * This epsilon must be positive to be taken into account.
   * If there is a conflict with a row epsilon (defined with addEpsilonRow()),
   * the largest epsilon is taken into account.
   * @note If an epsilon has already been defined on this column, then the old
   * epsilon will be replaced.
   * 
   * @param column_name The name of the column where the epsilon will be taken into account.
   * @param epsilon The epsilon error margin.
   * @return true If the epsilon could be defined.
   * @return false If the epsilon could not be defined.
   */
  virtual bool addEpsilonColumn(const String& column_name, Real epsilon) = 0;

  /**
   * @brief Method allowing an epsilon to be defined for a given row.
   * This epsilon must be positive to be taken into account.
   * If there is a conflict with a column epsilon (defined with addEpsilonColumn()),
   * the largest epsilon is taken into account.
   * @note If an epsilon has already been defined on this row, then the old
   * epsilon will be replaced.
   * 
   * @param column_name The name of the row where the epsilon will be taken into account.
   * @param epsilon The epsilon error margin.
   * @return true If the epsilon could be defined.
   * @return false If the epsilon could not be defined.
   */
  virtual bool addEpsilonRow(const String& row_name, Real epsilon) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
