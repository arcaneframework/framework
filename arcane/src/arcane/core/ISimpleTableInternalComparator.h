// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ISimpleTableInternalComparator.h                            (C) 2000-2025 */
/*                                                                           */
/* Interface representing a SimpleTableInternal comparator.                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ISIMPLETABLEINTERNALCOMPARATOR_H
#define ARCANE_CORE_ISIMPLETABLEINTERNALCOMPARATOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ISimpleTableInternalMng.h"

#include "arcane/core/ItemTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/**
 * @brief Class interface representing a SimpleTableInternal comparator
 * (aka STI).
 * 
 * The principle is to compare the values of one STI with the values of a
 * reference STI, using an epsilon representing the acceptable error margin.
 * 
 * There are two ways to configure this comparator:
 * - two arrays of Strings (row/column),
 * - two regular expressions (row/column).
 * 
 * You can add row/column names to these arrays,
 * specify whether these rows/columns should be included in the comparison, or,
 * conversely, if they should be excluded from the comparison.
 * 
 * The same applies to regular expressions: you add a row/column regular
 * expression and specify whether these expressions include or exclude
 * rows/columns.
 * 
 * 
 * If both types of methods are defined, the arrays take precedence over the
 * regular expressions: first, we check for the presence of the row/column
 * name in the corresponding array.
 * 
 * If the name is present, we include/exclude this row/column from the
 * comparison.
 * If the name is absent but a regular expression is defined, we search for
 * a match within it.
 * 
 * If neither type is defined (empty array and empty regular expression),
 * all rows/columns are included in the comparison.
 */
class ARCANE_CORE_EXPORT ISimpleTableInternalComparator
{
 public:
  virtual ~ISimpleTableInternalComparator() = default;

 public:
  /**
   * @brief Method allowing comparison of the values of the two STIs.
   * 
   * @param compare_dimension_too If the STI dimensions must be compared.
   * @return true If there are no differences.
   * @return false If there is at least one difference.
   */
  virtual bool compare(bool compare_dimension_too = false) = 0;

  /**
   * @brief Method allowing comparison of a single element.
   * Both SimpleTableInternals are represented by Refs, so they are always up to date.
   * This method can be used during calculation, allowing values to be compared as
   * the calculation progresses, instead of performing a final comparison at the
   * end (it is still possible to do both).
   * 
   * @param column_name The name of the column where the element is located.
   * @param row_name The name of the row where the element is located.
   * @return true If both values are equal.
   * @return false If both values are different.
   */
  virtual bool compareElem(const String& column_name, const String& row_name) = 0;

  /**
   * @brief Method allowing comparison of a value with a value from the reference table.
   * This method does not use the internal 'toCompare'.
   * 
   * @param elem The value to compare.
   * @param column_name The name of the column where the reference element is located.
   * @param row_name The name of the row where the reference element is located.
   * @return true If both values are equal.
   * @return false If both values are different.
   */
  virtual bool compareElem(Real elem, const String& column_name, const String& row_name) = 0;

  /**
   * @brief Method allowing the clearing of comparison arrays and regular
   * expressions. Does not affect the STIs.
   */
  virtual void clearComparator() = 0;

  /**
   * @brief Method allowing the addition of a column to the list of columns to compare.
   * 
   * @param column_name The name of the column to compare.
   * @return true If the name was successfully added.
   * @return false Otherwise.
   */
  virtual bool addColumnForComparing(const String& column_name) = 0;
  /**
   * @brief Method allowing the addition of a row to the list of rows to compare.
   * 
   * @param row_name The name of the row to compare.
   * @return true If the name was successfully added.
   * @return false Otherwise.
   */
  virtual bool addRowForComparing(const String& row_name) = 0;

  /**
   * @brief Method allowing definition of whether the column array represents
   * columns to include in the comparison (false/default) or columns to exclude
   * from the comparison (true).
   * 
   * @param is_exclusive true if the columns must be excluded.
   */
  virtual void isAnArrayExclusiveColumns(bool is_exclusive) = 0;

  /**
   * @brief Method allowing definition of whether the row array represents
   * rows to include in the comparison (false/default) or rows to exclude
   * from the comparison (true).
   * 
   * @param is_exclusive true if the rows must be excluded.
   */
  virtual void isAnArrayExclusiveRows(bool is_exclusive) = 0;

  /**
   * @brief Method allowing the addition of a regular expression to
   * determine the columns to compare.
   * 
   * @param regex_column The regular expression (ECMAScript format).
   */
  virtual void editRegexColumns(const String& regex_column) = 0;
  /**
   * @brief Method allowing the addition of a regular expression to
   * determine the rows to compare.
   * 
   * @param regex_row The regular expression (ECMAScript format).
   */
  virtual void editRegexRows(const String& regex_row) = 0;

  /**
   * @brief Method allowing specification that the regular expression
   * excludes columns instead of including them.
   * 
   * @param is_exclusive If the regular expression is exclusionary.
   */
  virtual void isARegexExclusiveColumns(bool is_exclusive) = 0;
  /**
   * @brief Method allowing specification that the regular expression
   * excludes rows instead of including them.
   * 
   * @param is_exclusive If the regular expression is exclusionary.
   */
  virtual void isARegexExclusiveRows(bool is_exclusive) = 0;

  /**
   * @brief Method allowing the definition of an epsilon for a given column.
   * This epsilon must be positive to be considered.
   * If there is a conflict with a row epsilon (defined with addEpsilonRow()),
   * the largest epsilon is taken into account.
   * @note If an epsilon has already been defined for this column, the old
   * epsilon will be replaced.
   * 
   * @param column_name The name of the column where the epsilon will be
   *                    taken into account.
   * @param epsilon The epsilon error margin.
   * @return true If the epsilon could be successfully defined.
   * @return false If the epsilon could not be defined.
   */
  virtual bool addEpsilonColumn(const String& column_name, Real epsilon) = 0;

  /**
   * @brief Method allowing the definition of an epsilon for a given row.
   * This epsilon must be positive to be considered.
   * If there is a conflict with a column epsilon (defined with addEpsilonColumn()),
   * the largest epsilon is taken into account.
   * @note If an epsilon has already been defined for this row, the old epsilon
   * will be replaced.
   * 
   * @param row_name The name of the row where the epsilon will be taken into account.
   * @param epsilon The epsilon error margin.
   * @return true If the epsilon could be successfully defined.
   * @return false If the epsilon could not be defined.
   */
  virtual bool addEpsilonRow(const String& row_name, Real epsilon) = 0;


  /**
   * @brief Method allowing retrieval of a reference to the used "reference"
   * SimpleTableInternal object.
   * 
   * @return Ref<SimpleTableInternal> A copy of the reference. 
   */
  virtual Ref<SimpleTableInternal> internalRef() = 0;

  /**
   * @brief Method allowing definition of a reference to a "reference"
   * SimpleTableInternal.
   * 
   * @param simple_table_internal The reference to a SimpleTableInternal.
   */
  virtual void setInternalRef(const Ref<SimpleTableInternal>& simple_table_internal) = 0;

  /**
   * @brief Method allowing retrieval of a reference to the used "to compare"
   * SimpleTableInternal object.
   * 
   * @return Ref<SimpleTableInternal> A copy of the reference. 
   */
  virtual Ref<SimpleTableInternal> internalToCompare() = 0;

  /**
   * @brief Method allowing definition of a reference to the "to compare"
   * SimpleTableInternal.
   * 
   * @param simple_table_internal The reference to a SimpleTableInternal.
   */
  virtual void setInternalToCompare(const Ref<SimpleTableInternal>& simple_table_internal) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
