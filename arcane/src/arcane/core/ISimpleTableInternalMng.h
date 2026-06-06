// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ISimpleTableInternalMng.h                                   (C) 2000-2025 */
/*                                                                           */
/* Interface representing a manager for SimpleTableInternal.                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ISIMPLETABLEINTERNALMNG_H
#define ARCANE_CORE_ISIMPLETABLEINTERNALMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Array.h"
#include "arcane/utils/Array2.h"

#include "arcane/core/SimpleTableInternal.h"
#include "arcane/core/ISubDomain.h"
#include "arcane/core/ItemTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/**
 * \brief Class interface representing a manager
 * for SimpleTableInternal (aka STI). 
 * 
 * This manager allows for several types of operations
 * on the STI: adding rows, columns, values, etc.
 * 
 * There are two modes of operation (which can be mixed): 
 * - using the names or positions of rows/columns,
 * - using a position pointer within the array.
 * 
 * The first mode is the easiest to use and is sufficient
 * for most users. You provide a name (or position)
 * of a row or column and a value, and this value is placed
 * after the other values in the row or column.
 * 
 * The second mode is more advanced and is mainly used to replace
 * elements already present or to optimize performance (if there are 
 * 40 rows, 40 values to add sequentially, and you use the 
 * column names 40 times, this results in 40 String searches in a 
 * StringUniqueArray, which is not optimal performance).
 * A pointer representing the last added element is present in
 * STI. You can modify elements around this pointer (top, bottom,
 * left, right) using the available methods.
 * This pointer can be placed anywhere using the element() methods.
 * This pointer is not read by the methods of the first mode but is
 * updated by them.
 */
class ARCANE_CORE_EXPORT ISimpleTableInternalMng
{
 public:

  virtual ~ISimpleTableInternalMng() = default;

 public:

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /**
   * \brief Method to clear the content
   * of the SimpleTableInternal.
   */
  virtual void clearInternal() = 0;

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /**
   * \brief Method to add a row.
   * 
   * \param row_name The name of the row. Must not be empty.
   * \return Integer The position of the row in the array 
   *                 (-1 if the given name is incorrect).
   */
  virtual Integer addRow(const String& row_name) = 0;

  /**
   * \brief Method to add a row.
   * 
   * If the number of elements in 'elements' is greater than the
   * number of columns, the addition still takes place (but the
   * extra elements will not be added).
   * 
   * \param row_name The name of the row. Must not be empty.
   * \param elements The elements to insert into the row.
   * \return Integer The position of the row in the array.
   *                 (-1 if the given name is incorrect).
   */
  virtual Integer addRow(const String& row_name, ConstArrayView<Real> elements) = 0;

  /**
   * \brief Method to add multiple rows.
   * 
   * \param rows_names The names of the rows. Each name must not be empty.
   * \return true If all rows were created.
   * \return false If not all rows were created.
   */
  virtual bool addRows(StringConstArrayView rows_names) = 0;

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /**
   * \brief Method to add a column.
   * 
   * \param column_name The name of the column. Must not be empty.
   * \return Integer The position of the column in the array.
   *                 (-1 if the given name is incorrect).
   */
  virtual Integer addColumn(const String& column_name) = 0;

  /**
   * \brief Method to add a column.
   * 
   * If the number of elements in 'elements' is greater than the
   * number of rows, the addition still takes place (but the
   * extra elements will not be added).
   * 
   * \param column_name The name of the column. Must not be empty.
   * \param elements The elements to add to the column.
   * \return Integer The position of the column in the array.
   *                 (-1 if the given name is incorrect).
   */
  virtual Integer addColumn(const String& column_name, ConstArrayView<Real> elements) = 0;

  /**
   * \brief Method to add multiple columns.
   * 
   * \param rows_names The names of the columns. Each name must not be empty.
   * \return true If all columns were created.
   * \return false If not all columns were created.
   */
  virtual bool addColumns(StringConstArrayView columns_names) = 0;

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /**
   * \brief Method to add an element to a row.
   * 
   * \param position The position of the row.
   * \param element The element to add.
   * \return true If the element was successfully added.
   * \return false If the element could not be added.
   */
  virtual bool addElementInRow(Integer position, Real element) = 0;

  /**
   * \brief Method to add an element to a row.
   * 
   * \param row_name The name of the row.
   * \param element The element to add.
   * \param create_if_not_exist To specify whether the row should be created
   *                            if it does not already exist.
   * \return true If the element was successfully added.
   * \return false If the element could not be added.
   */
  virtual bool addElementInRow(const String& row_name, Real element, bool create_if_not_exist = true) = 0;

  /**
   * \brief Method to add an element to the row 
   * most recently manipulated.
   * 
   * This method differs from 'editElementRight()' because here, an element is added
   * to the end of the row, not necessarily after the
   * last added element.
   * 
   * \param element The element to add.
   * \return true If the element was added.
   * \return false If the element could not be added.
   */
  virtual bool addElementInSameRow(Real element) = 0;

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /**
   * \brief Method to add multiple elements to a row.
   * 
   * If the number of elements in 'elements' is greater than the
   * number of available columns, the addition still takes place (but the
   * extra elements will not be added) and a return value of false will be returned.
   * 
   * \param position The position of the row.
   * \param elements The array of elements to add.
   * \return true If all elements were added.
   * \return false If [0;len(elements)[ elements were added.
   */
  virtual bool addElementsInRow(Integer position, ConstArrayView<Real> elements) = 0;

  /**
   * \brief Method to add multiple elements to a row.
   * 
   * If the number of elements in 'elements' is greater than the
   * number of available columns, the addition still takes place (but the
   * extra elements will not be added) and a return value of false will be returned.
   * 
   * \param row_name The name of the row.
   * \param elements The array of elements to add.
   * \param create_if_not_exist To specify whether the row should be created
   *                            if it does not already exist.
   * \return true If all elements were added.
   * \return false If [0;len(elements)[ elements were added.
   */
  virtual bool addElementsInRow(const String& row_name, ConstArrayView<Real> elements, bool create_if_not_exist = true) = 0;

  /**
   * \brief Method to add multiple elements to the 
   * row most recently manipulated.
   * 
   * If the number of elements in 'elements' is greater than the
   * number of available columns, the addition still takes place (but the
   * extra elements will not be added) and a return value of false will be returned.
   * 
   * Apart from the fact that we are manipulating an array here, this method differs
   * from 'editElementRight()' because here, elements are added to the end of the row,
   * not necessarily after the last added element.
   * 
   * \param elements The array of elements to add.
   * \return true If all elements were added.
   * \return false If [0;len(elements)[ elements were added.
   */
  virtual bool addElementsInSameRow(ConstArrayView<Real> elements) = 0;

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /**
   * \brief Method to add an element to a column.
   * 
   * \param position The position of the column.
   * \param element The element to add.
   * \return true If the element was successfully added.
   * \return false If the element could not be added.
   */
  virtual bool addElementInColumn(Integer position, Real element) = 0;

  /**
   * \brief Method to add an element to a column.
   * 
   * \param column_name The name of the column.
   * \param element The element to add.
   * \param create_if_not_exist To specify whether the column should be created
   *                            if it does not already exist.
   * \return true If the element was successfully added.
   * \return false If the element could not be added.
   */
  virtual bool addElementInColumn(const String& column_name, Real element, bool create_if_not_exist = true) = 0;

  /**
   * \brief Method to add an element to the column
   * most recently manipulated.
   * 
   * This method differs from 'editElementDown()' because here, an element is added
   * to the end of the column, not necessarily after the last added element.
   * 
   * \param element The element to add.
   * \return true If the element was added.
   * \return false If the element could not be added.
   */
  virtual bool addElementInSameColumn(Real element) = 0;

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /**
   * \brief Method to add multiple elements to a column.
   * 
   * If the number of elements in 'elements' is greater than the
   * number of available rows, the addition still takes place (but the
   * extra elements will not be added) and a return value of false will be returned.
   * 
   * \param position The position of the column.
   * \param elements The array of elements to add.
   * \return true If all elements were added.
   * \return false If [0;len(elements)[ elements were added.
   */
  virtual bool addElementsInColumn(Integer position, ConstArrayView<Real> elements) = 0;

  /**
   * \brief Method to add multiple elements to a column.
   * 
   * If the number of elements in 'elements' is greater than the
   * number of available rows, the addition still takes place (but the
   * extra elements will not be added) and a return value of false will be returned.
   * 
   * \param column_name The name of the column.
   * \param elements The array of elements to add.
   * \param create_if_not_exist To specify whether the column should be created if
   *                            it does not already exist.
   * \return true If all elements were added.
   * \return false If [0;len(elements)[ elements were added.
   */
  virtual bool addElementsInColumn(const String& column_name, ConstArrayView<Real> elements, bool create_if_not_exist = true) = 0;

  /**
   * \brief Method to add multiple elements to the
   * column most recently manipulated.
   * 
   * If the number of elements in 'elements' is greater than the
   * number of available rows, the addition still takes place (but the
   * extra elements will not be added) and a return value of false will be returned.
   * 
   * Apart from the fact that we are manipulating an array here, this method differs
   * from 'editElementDown()' because here, elements are added to the end of the column,
   * not necessarily after the last added element.
   * 
   * \param elements The array of elements to add.
   * \return true If all elements were added.
   * \return false If [0;len(elements)[ elements were added.
   */
  virtual bool addElementsInSameColumn(ConstArrayView<Real> elements) = 0;

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /**
   * \brief Method to edit an element above the last
   * element most recently manipulated (row above/same column).
   * 
   * The element being modified thus becomes the last modified element
   * at the end of this method (if update_last_position = true).
   * 
   * \param element The element to modify.
   * \param update_last_position Should the "last modified element" cursor be moved?
   * \return true If the element was modified.
   * \return false If the element could not be modified.
   */
  virtual bool editElementUp(Real element, bool update_last_position = true) = 0;

  /**
   * \brief Method to edit an element below the last 
   * element most recently manipulated (row below/same column).
   * 
   * The element being modified thus becomes the last modified element 
   * at the end of this method (if update_last_position = true).
   * 
   * This method differs from 'addElementInSameColumn()' because here, an element is added
   * (or modified) below the last manipulated element, which is not
   * necessarily at the end of the column.
   * 
   * \param element The element to modify.
   * \param update_last_position Should the "last modified element" cursor be moved?
   * \return true If the element was modified.
   * \return false If the element could not be modified.
   */
  virtual bool editElementDown(Real element, bool update_last_position = true) = 0;

  /**
   * \brief Method to edit an element to the left of the last
   * element most recently manipulated (same row/column to the left).
   * 
   * The element being modified thus becomes the last modified element
   * at the end of this method (if update_last_position = true).
   * 
   * \param element The element to modify.
   * \param update_last_position Should the "last modified element" cursor be moved?
   * \return true If the element was modified.
   * \return false If the element could not be modified.
   */
  virtual bool editElementLeft(Real element, bool update_last_position = true) = 0;

  /**
   * \brief Method allowing editing an element to the right of the last
   * element recently manipulated (same row/column to the right).
   * 
   * The element being modified thus becomes the last modified element
   * at the end of this method (if update_last_position = true).
   * 
   * This method differs from 'addElementInSameRow()' because here, we add 
   * (or modify) an element to the right of the last manipulated element, 
   * which is not necessarily at the end of the column.
   * 
   * \param element The element to modify.
   * \param update_last_position Should the "last modified element" cursor be moved?
   * \return true If the element was modified.
   * \return false If the element could not be modified.
   */
  virtual bool editElementRight(Real element, bool update_last_position = true) = 0;

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /**
   * \brief Method allowing retrieval of an element above the last
   * element recently manipulated (row above/same column).
   * 
   * The element retrieved thus becomes the last "modified" element
   * at the end of this method (if update_last_position = true).
   * 
   * \param update_last_position Should the "last modified element" cursor be moved?
   * \return Real The found element (0 if not found).
   */
  virtual Real elementUp(bool update_last_position = false) = 0;

  /**
   * \brief Method allowing retrieval of an element below the last
   * element recently manipulated (row below/same column).
   * 
   * The element retrieved thus becomes the last "modified" element
   * at the end of this method (if update_last_position = true).
   * 
   * \param update_last_position Should the "last modified element" cursor be moved?
   * \return Real The found element (0 if not found).
   */
  virtual Real elementDown(bool update_last_position = false) = 0;

  /**
   * \brief Method allowing retrieval of an element to the left of the last
   * element recently manipulated (same row/column to the left).
   * 
   * The element retrieved thus becomes the last "modified" element
   * at the end of this method (if update_last_position = true).
   * 
   * \param update_last_position Should the "last modified element" cursor be moved?
   * \return Real The found element (0 if not found).
   */
  virtual Real elementLeft(bool update_last_position = false) = 0;

  /**
   * \brief Method allowing retrieval of an element to the right of the last
   * element recently manipulated (same row/column to the right).
   * 
   * The element retrieved thus becomes the last "modified" element
   * at the end of this method (if update_last_position = true).
   * 
   * \param update_last_position Should the "last modified element" cursor be moved?
   * \return Real The found element (0 if not found).
   */
  virtual Real elementRight(bool update_last_position = false) = 0;

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /**
   * \brief Method allowing modification of an element in the table.
   * 
   * The x and y positions correspond to the location of the last
   * manipulated element.
   * 
   * This method is useful after using
   * 'elemUDLR(true)', for example.
   * 
   * \param element The replacement element.
   * \return true If the element was successfully replaced.
   * \return false If the element was not replaced.
   */
  virtual bool editElement(Real element) = 0;

  /**
   * \brief Method allowing modification of an element in the table.
   * 
   * \param position_x The position of the column to modify.
   * \param position_y The position of the row to modify.
   * \param element The replacement element.
   * \return true If the element was successfully replaced.
   * \return false If the element was not replaced.
   */
  virtual bool editElement(Integer position_x, Integer position_y, Real element) = 0;

  /**
   * \brief Method allowing modification of an element in the table.
   * 
   * \param column_name The name of the column where the element is located.
   * \param row_name The name of the row where the element is located.
   * \param element The replacement element.
   * \return true If the element was successfully replaced.
   * \return false If the element could not be replaced.
   */
  virtual bool editElement(const String& column_name, const String& row_name, Real element) = 0;

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /**
   * \brief Method allowing retrieval of a copy of an element.
   * 
   * The x and y positions correspond to the location of the last manipulated element.
   * 
   * \return Real The found element (0 if not found).
   */
  virtual Real element() = 0;

  /**
   * \brief Method allowing retrieval of a copy of an element.
   * 
   * \param position_x The position of the column where the element is located.
   * \param position_y The position of the row where the element is located.
   * \param update_last_position Should the "last modified element" cursor be moved?
   * \return Real The found element (0 if not found).
   */
  virtual Real element(Integer position_x, Integer position_y, bool update_last_position = false) = 0;

  /**
   * \brief Method allowing retrieval of a copy of an element.
   * 
   * \param column_name The name of the column where the element is located.
   * \param row_name The name of the row where the element is located.
   * \param update_last_position Should the "last modified element" cursor be moved?
   * \return Real The found element (0 if not found).
   */
  virtual Real element(const String& column_name, const String& row_name, bool update_last_position = false) = 0;

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /**
   * \brief Method allowing retrieval of a copy of a row.
   * 
   * \param position The position of the row.
   * \return RealUniqueArray The copy of the row (empty array if not found).
   */
  virtual RealUniqueArray row(Integer position) = 0;

  /**
   * \brief Method allowing retrieval of a copy of a row.
   * 
   * \param row_name The name of the row.
   * \return RealUniqueArray The copy of the row (empty array if not found).
   */
  virtual RealUniqueArray row(const String& row_name) = 0;

  /**
   * \brief Method allowing retrieval of a copy of a column.
   * 
   * \param position The position of the column.
   * \return RealUniqueArray The copy of the column (empty array if not found).
   */
  virtual RealUniqueArray column(Integer position) = 0;

  /**
   * \brief Method allowing retrieval of a copy of a column.
   * 
   * \param column_name The name of the column.
   * \return RealUniqueArray The copy of the column (empty array if not found).
   */
  virtual RealUniqueArray column(const String& column_name) = 0;

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /**
   * \brief Method allowing retrieval of the size of a row.
   * Including hypothetical 'gaps' in the row.
   * 
   * \param position The position of the row.
   * \return Integer The size of the row (0 if not found).
   */
  virtual Integer rowSize(Integer position) = 0;

  /**
   * \brief Method allowing retrieval of the size of a row.
   * Including hypothetical 'gaps' in the row.
   * 
   * \param position The name of the row.
   * \return Integer The size of the row (0 if not found).
   */
  virtual Integer rowSize(const String& row_name) = 0;

  /**
   * \brief Method allowing retrieval of the size of a column.
   * Including hypothetical 'gaps' in the column.
   * 
   * \param position The position of the column.
   * \return Integer The size of the column (0 if not found).
   */
  virtual Integer columnSize(Integer position) = 0;

  /**
   * \brief Method allowing retrieval of the size of a column.
   * Including hypothetical 'gaps' in the column.
   * 
   * \param position The name of the column.
   * \return Integer The size of the column (0 if not found).
   */
  virtual Integer columnSize(const String& column_name) = 0;

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /**
   * \brief Method allowing retrieval of the position of a row.
   * 
   * \param row_name The name of the row.
   * \return Integer The position of the row (-1 if not found).
   */
  virtual Integer rowPosition(const String& row_name) = 0;

  /**
   * \brief Method allowing retrieval of the position of a column.
   * 
   * \param row_name The name of the column.
   * \return Integer The position of the column (-1 if not found).
   */
  virtual Integer columnPosition(const String& column_name) = 0;

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /**
   * \brief Method allowing retrieval of the number of rows in the table.
   * This is, in a sense, the maximum number of elements a column can contain.
   * 
   * \return Integer The number of rows in the table.
   */
  virtual Integer numberOfRows() = 0;

  /**
   * \brief Method allowing retrieval of the number of columns in the table.
   * This is, in a sense, the maximum number of elements a row can contain.
   * 
   * \return Integer The number of columns in the table.
   */
  virtual Integer numberOfColumns() = 0;

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /**
   * \brief Method allowing retrieval of the name of a row
   * from its position.
   * 
   * \param position The position of the row.
   * \return String The name of the row 
   *         (empty string if the row was not found).
   */
  virtual String rowName(Integer position) = 0;

  /**
   * \brief Method allowing retrieval of the name of a column
   * from its position.
   * 
   * \param position The position of the column.
   * \return String The name of the column 
   *         (empty string if the column was not found).
   */
  virtual String columnName(Integer position) = 0;

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /**
   * \brief Method allowing changing the name of a row.
   * 
   * \param position The position of the row.
   * \param new_name The new name of the row. Must not be empty.
   * \return true If the change occurred.
   * \return false If the change did not occur.
   */
  virtual bool editRowName(Integer position, const String& new_name) = 0;

  /**
   * \brief Method allowing changing the name of a row.
   * 
   * \param row_name The current name of the row.
   * \param new_name The new name of the row. Must not be empty.
   * \return true If the change occurred.
   * \return false If the change did not occur.
   */
  virtual bool editRowName(const String& row_name, const String& new_name) = 0;

  /**
   * \brief Method allowing changing the name of a column.
   * 
   * \param position The position of the column.
   * \param new_name The new name of the column. Must not be empty.
   * \return true If the change occurred.
   * \return false If the change did not occur.
   */
  virtual bool editColumnName(Integer position, const String& new_name) = 0;

  /**
   * \brief Method allowing changing the name of a column.
   * 
   * \param column_name The current name of the column.
   * \param new_name The new name of the column. Must not be empty.
   * \return true If the change occurred.
   * \return false If the change did not occur.
   */
  virtual bool editColumnName(const String& column_name, const String& new_name) = 0;

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /**
   * \brief Method allowing creation of a column containing the average of
   * elements of each row.
   * 
   * \param column_name The name of the new column. Must not be empty.
   * \return Integer The position of the column.
   */
  virtual Integer addAverageColumn(const String& column_name) = 0;

  /**
   * \brief Method allowing retrieval of a reference to the object
   * SimpleTableInternal used.
   * 
   * \return Ref<SimpleTableInternal> A copy of the reference. 
   */
  virtual Ref<SimpleTableInternal> internal() = 0;

  /**
   * \brief Method allowing setting a reference to a
   * SimpleTableInternal.
   * 
   * \param simple_table_internal The reference to a SimpleTableInternal.
   */
  virtual void setInternal(const Ref<SimpleTableInternal>& simple_table_internal) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
