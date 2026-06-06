// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ISimpleTableOutput.h                                        (C) 2000-2025 */
/*                                                                           */
/* Interface for simple table output services.                               */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ISIMPLETABLEOUTPUT_H
#define ARCANE_CORE_ISIMPLETABLEOUTPUT_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ISimpleTableInternalMng.h"
#include "arcane/core/ISimpleTableWriterHelper.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/**
 * \ingroup StandardService
 * \brief Interface representing a simple table output.
 */
class ARCANE_CORE_EXPORT ISimpleTableOutput
{
 public:

  virtual ~ISimpleTableOutput() = default;

 public:

  /**
   * \brief Method to initialize the table.
   */
  virtual bool init() = 0;
  /**
   * \brief Method to initialize the table.
   * 
   * \param table_name The name of the table (and the output file).
   */
  virtual bool init(const String& table_name) = 0;
  /**
   * \brief Method to initialize the table.
   * 
   * \param table_name The name of the table (and the output file).
   * \param directory_name The name of the directory where the tables should be saved.
   */
  virtual bool init(const String& table_name, const String& directory_name) = 0;

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /**
   * \brief Method to clear the tables
   */
  virtual void clear() = 0;

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /**
   * \brief Method to add a row.
   * 
   * \param row_name The name of the row.
   * \return Integer The position of the row in the table.
   */
  virtual Integer addRow(const String& row_name) = 0;
  /**
   * \brief Method to add a row.
   * 
   * If the number of elements in 'elements' is greater than the
   * number of columns, the addition still takes place (but the
   * extra elements will not be added).
   * 
   * \param row_name The name of the row.
   * \param elements The elements to insert into the row.
   * \return Integer The position of the row in the table.
   */
  virtual Integer addRow(const String& row_name, ConstArrayView<Real> elements) = 0;
  /**
   * \brief Method to add multiple rows.
   * 
   * \param rows_names The names of the rows.
   * \return true If all rows were created.
   * \return false If not all rows were created.
   */
  virtual bool addRows(StringConstArrayView rows_names) = 0;

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /**
   * \brief Method to add a column.
   * 
   * \param column_name The name of the column.
   * \return Integer The position of the column in the table.
   */
  virtual Integer addColumn(const String& column_name) = 0;
  /**
   * \brief Method to add a column.
   * 
   * If the number of elements in 'elements' is greater than the
   * number of rows, the addition still takes place (but the
   * extra elements will not be added).
   * 
   * \param column_name The name of the column.
   * \param elements The elements to add to the column.
   * \return Integer The position of the column in the table.
   */
  virtual Integer addColumn(const String& column_name, ConstArrayView<Real> elements) = 0;
  /**
   * \brief Method to add multiple columns.
   * 
   * \param rows_names The names of the columns.
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
   * \param create_if_not_exist To determine whether the row should be created
   *                            if it does not yet exist.
   * \return true If the element was successfully added.
   * \return false If the element could not be added.
   */
  virtual bool addElementInRow(const String& row_name, Real element, bool create_if_not_exist = true) = 0;
  /**
   * \brief Method to add an element to the last manipulated row.
   * 
   * This method differs from 'editElementRight()' because here, an element is
   * added to the end of the row, not necessarily after the last added element.
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
   * extra elements will not be added) and a return false will be returned.
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
   * extra elements will not be added) and a return false will be returned.
   * 
   * \param row_name The name of the row.
   * \param elements The array of elements to add.
   * \param create_if_not_exist To determine whether the row should be created
   *                            if it does not yet exist.
   * \return true If all elements were added.
   * \return false If [0;len(elements)[ elements were added.
   */
  virtual bool addElementsInRow(const String& row_name, ConstArrayView<Real> elements, bool create_if_not_exist = true) = 0;
  /**
   * \brief Method to add multiple elements to the last manipulated row.
   * 
   * If the number of elements in 'elements' is greater than the
   * number of available columns, the addition still takes place (but the
   * extra elements will not be added) and a return false will be returned.
   * 
   * Aside from the fact that we are manipulating an array here, this method differs
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
   * \param create_if_not_exist To determine whether the column should be created
   *                            if it does not yet exist.
   * \return true If the element was successfully added.
   * \return false If the element could not be added.
   */
  virtual bool addElementInColumn(const String& column_name, Real element, bool create_if_not_exist = true) = 0;
  /**
   * \brief Method to add an element to the last manipulated column.
   * 
   * This method differs from 'editElementDown()' because here, an element is
   * added to the end of the column, not necessarily after the last added element.
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
   * extra elements will not be added) and a return false will be returned.
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
   * extra elements will not be added) and a return false will be returned.
   * 
   * \param column_name The name of the column.
   * \param elements The array of elements to add.
   * \param create_if_not_exist To determine whether the column should be created if
   *                            it does not yet exist.
   * \return true If all elements were added.
   * \return false If [0;len(elements)[ elements were added.
   */
  virtual bool addElementsInColumn(const String& column_name, ConstArrayView<Real> elements, bool create_if_not_exist = true) = 0;
  /**
   * \brief Method to add multiple elements to the last manipulated column.
   * 
   * If the number of elements in 'elements' is greater than the
   * number of available rows, the addition still takes place (but the
   * extra elements will not be added) and a return false will be returned.
   * 
   * Aside from the fact that we are manipulating an array here, this method differs
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
   * element manipulated (row above/same column).
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
   * element manipulated (row below/same column).
   * 
   * The element being modified thus becomes the last modified element 
   * at the end of this method (if update_last_position = true).
   * 
   * This method differs from 'addElementInSameColumn()' because here, an element is added
   * (or modified) below the last manipulated element, which is not necessarily at the end of the column.
   * 
   * \param element The element to modify.
   * \param update_last_position Should the "last modified element" cursor be moved?
   * \return true If the element was modified.
   * \return false If the element could not be modified.
   */
  virtual bool editElementDown(Real element, bool update_last_position = true) = 0;
  /**
   * \brief Method to edit an element to the left of the last
   * element manipulated (same row/column to the left).
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
   * \brief Method to edit an element to the right of the last
   * element manipulated (same row/column to the right).
   * 
   * The element being modified thus becomes the last modified element
   * at the end of this method (if update_last_position = true).
   * 
   * This method differs from 'addElementInSameRow()' because here, an element is added
   * (or modified) to the right of the last manipulated element, which is not necessarily at the end of the column.
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
   * \brief Method to retrieve an element above the last
   * element manipulated (row above/same column).
   * 
   * The element retrieved thus becomes the last "modified" element
   * at the end of this method (if update_last_position = true).
   * 
   * \param update_last_position Should the "last modified element" cursor be moved?
   * \return Real The found element (0 if not found).
   */
  virtual Real elementUp(bool update_last_position = false) = 0;
  /**
   * \brief Method to retrieve an element below the last
   * element manipulated (row below/same column).
   * 
   * The element retrieved thus becomes the last "modified" element
   * at the end of this method (if update_last_position = true).
   * 
   * \param update_last_position Should the "last modified element" cursor be moved?
   * \return Real The found element (0 if not found).
   */
  virtual Real elementDown(bool update_last_position = false) = 0;
  /**
   * \brief Method to retrieve an element to the left of the last
   * element manipulated (same row/column to the left).
   * 
   * The element retrieved thus becomes the last "modified" element
   * at the end of this method (if update_last_position = true).
   * 
   * \param update_last_position Should the "last modified element" cursor be moved?
   * \return Real The found element (0 if not found).
   */
  virtual Real elementLeft(bool update_last_position = false) = 0;
  /**
   * \brief Method to retrieve an element to the right of the last
   * element manipulated (same row/column to the right).
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
   * \brief Method to modify an element in the table.
   * 
   * The x and y positions correspond to the location of the last
   * manipulated element.
   * 
   * This method is useful after using
   * 'elemUDLR(true)' for example.
   * 
   * \param element The replacement element.
   * \return true If the element was successfully replaced.
   * \return false If the element was not replaced.
   */
  virtual bool editElement(Real element) = 0;
  /**
   * \brief Method to modify an element in the table.
   * 
   * \param position_x The position of the column to modify.
   * \param position_y The position of the row to modify.
   * \param element The replacement element.
   * \return true If the element was successfully replaced.
   * \return false If the element was not replaced.
   */
  virtual bool editElement(Integer position_x, Integer position_y, Real element) = 0;
  /**
   * \brief Method allowing modification of an element in the array.
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
   * \brief Method allowing retrieval of the number of rows in the array.
   * This is, in a way, the maximum number of elements a column can contain.
   * 
   * \return Integer The number of rows in the array.
   */
  virtual Integer numberOfRows() = 0;
  /**
   * \brief Method allowing retrieval of the number of columns in the array.
   * This is, in a way, the maximum number of elements a row can contain.
   * 
   * \return Integer The number of columns in the array.
   */
  virtual Integer numberOfColumns() = 0;

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  virtual String rowName(Integer position) = 0;
  virtual String columnName(Integer position) = 0;

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /**
   * \brief Method allowing changing the name of a row.
   * 
   * \param position The position of the row.
   * \param new_name The new name of the row.
   * \return true If the change occurred.
   * \return false If the change did not occur.
   */
  virtual bool editRowName(Integer position, const String& new_name) = 0;
  /**
   * \brief Method allowing changing the name of a row.
   * 
   * \param row_name The current name of the row.
   * \param new_name The new name of the row.
   * \return true If the change occurred.
   * \return false If the change did not occur.
   */
  virtual bool editRowName(const String& row_name, const String& new_name) = 0;

  /**
   * \brief Method allowing changing the name of a column.
   * 
   * \param position The position of the column.
   * \param new_name The new name of the column.
   * \return true If the change occurred.
   * \return false If the change did not occur.
   */
  virtual bool editColumnName(Integer position, const String& new_name) = 0;
  /**
   * \brief Method allowing changing the name of a column.
   * 
   * \param column_name The current name of the column.
   * \param new_name The new name of the column.
   * \return true If the change occurred.
   * \return false If the change did not occur.
   */
  virtual bool editColumnName(const String& column_name, const String& new_name) = 0;

  /**
   * \brief Method allowing creation of a column containing the average of the
   * elements of each row.
   * 
   * \param column_name The name of the new column.
   * \return Integer The position of the column.
   */
  virtual Integer addAverageColumn(const String& column_name) = 0;

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/
  /**
   * \brief Method allowing display of the array.
   * Method performing collective operations.
   * 
   * \param rank The ID of the process that should display the array
   * (-1 to signify "all processes").
   */
  virtual void print(Integer rank = 0) = 0;

  virtual bool writeFile(const Directory& root_directory, Integer rank) = 0;

  /**
   * \brief Method allowing writing the array to a file.
   * Method performing collective operations.
   * If rank != -1, processes other than P0 return true.
   * 
   * \param rank The ID of the process that should write the array to a file
   * (-1 to signify "all processes").
   * \return true If the file was written correctly.
   * \return false If the file was not written correctly.
   */
  virtual bool writeFile(Integer rank = -1) = 0;
  /**
   * \brief Method allowing writing the array to a file.
   * Method performing collective operations.
   * If rank != -1, processes other than P0 return true.
   * 
   * \param directory The directory where the file will be written. The final
   * path will be "./[output_dir]/csv/[directory]/".
   * \param rank The ID of the process that should write the array to a file
   * (-1 to signify "all processes").
   * \return true If the file was written correctly.
   * \return false If the file was not written correctly.
   * 
   * \deprecated Use setOutputDirectory() then writeFile() instead.
   */
  virtual bool writeFile(const String& directory, Integer rank = -1) = 0;

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /**
   * \brief Method allowing retrieval of the precision currently
   * used for writing values.
   * 
   * \return Integer The precision.
   */
  virtual Integer precision() = 0;
  /**
   * \brief Method allowing modification of the print precision.
   * 
   * For both the 'print()' method and the 'writeFile()' methods.
   * 
   * \warning The "std::fixed" flag modifies the behavior of "setPrecision()".
   * If the "std::fixed" flag is disabled, precision defines the total number
   * of digits (before and after the comma); if the "std::fixed" flag is enabled,
   * precision defines the number of digits after the comma. Therefore, be
   * careful when using "std::numeric_limits<Real>::max_digits10" (for writing)
   * or "std::numeric_limits<Real>::digits10" (for reading), which should be used
   * without the "std::fixed" flag.
   * 
   * \param precision The new precision.
   */
  virtual void setPrecision(Integer precision) = 0;

  /**
   * \brief Method allowing knowledge of whether the 'std::fixed' flag is
   * active or not for writing values.
   * 
   * \return true If yes.
   * \return false If no.
   */
  virtual bool isFixed() = 0;
  /**
   * \brief Method allowing setting the 'std::fixed' flag or not.
   * 
   * For both the 'print()' method and the 'writetable()' method.
   * 
   * This flag allows 'forcing' the number of digits after the comma
   * to the desired precision. For example, if 'setPrecision(4)' is
   * called, and 'setFixed(true)' is called, the print of '6.1' will
   * yield '6.1000'.
   * 
   * \warning The "std::fixed" flag modifies the behavior of
   * "setPrecision()". If the "std::fixed" flag is disabled, precision
   * defines the total number of digits (before and after the comma);
   * if the "std::fixed" flag is enabled, precision defines the number
   * of digits after the comma. Therefore, be careful when using
   * "std::numeric_limits<Real>::max_digits10" (for writing) or
   * "std::numeric_limits<Real>::digits10" (for reading), which should
   * be used without the "std::fixed" flag.
   * 
   * \param fixed Whether the 'std::fixed' flag should be set or not.
   */
  virtual void setFixed(bool fixed) = 0;

  /**
   * \brief Method allowing knowledge of whether the 'std::scientific' flag is
   * active or not for writing values.
   * 
   * \return true If yes.
   * \return false If no.
   */
  virtual bool isForcedToUseScientificNotation() = 0;
  /**
   * \brief Method allowing setting the 'std::scientific' flag or not.
   * 
   * For both the 'print()' method and the 'writetable()' method.
   * 
   * This flag allows 'forcing' the display of values in scientific notation.
   * 
   * \param use_scientific Whether the 'std::scientific' flag should be set or not.
   */
  virtual void setForcedToUseScientificNotation(bool use_scientific) = 0;

  /**
   * \brief Accessor allowing retrieval of the name of the directory
   * where the arrays will be placed.
   * 
   * May be different for each process (depending on the implementation).
   * 
   * \return String The directory.
   */
  virtual String outputDirectory() = 0;
  /**
   * \brief Accessor allowing definition of the directory
   * in which the arrays will be saved.
   * 
   * May be different for each process (depending on the implementation).
   * 
   * \param directory The directory.
   */
  virtual void setOutputDirectory(const String& directory) = 0;

  /**
   * \brief Accessor allowing retrieval of the name of the arrays.
   * 
   * May be different for each process (depending on the implementation).
   * 
   * \return String The name.
   */
  virtual String tableName() = 0;
  /**
   * \brief Accessor allowing definition of the array name.
   * 
   * May be different for each process (depending on the implementation).
   * 
   * \param name The name.
   */
  virtual void setTableName(const String& name) = 0;

  /**
   * \brief Accessor allowing retrieval of the file name.
   * 
   * May be different for each process (depending on the implementation).
   * 
   * \return String The name.
   */
  virtual String fileName() = 0;

  /**
   * \brief Accessor allowing retrieval of the path where the arrays
   * will be saved.
   * 
   * Compared to rootPathOutput(), the return value may differ
   * depending on the "directory" and the "name".
   * 
   * \return String The path.
   */
  virtual Directory outputPath() = 0;

  /**
   * \brief Accessor allowing retrieval of the path where the
   * implementation saves these arrays.
   * 
   * Compared to pathOutput(), the return value does not depend on
   * "directory" or "name".
   * 
   * \return String The path.
   */
  virtual Directory rootPath() = 0;

  /**
   * \brief Method allowing knowledge of whether the parameters
   * currently held by the implementation allow it to write one
   * file per process.
   * 
   * \return true If yes, the implementation can write one file per process.
   * \return false Otherwise, only one file can be written.
   */
  virtual bool isOneFileByRanksPermited() = 0;

  /**
   * \brief Method allowing knowledge of the service's file type.
   * 
   * \return String The file type.
   */
  virtual String fileType() = 0;

  /**
   * \brief Method allowing retrieval of a reference to the
   * SimpleTableInternal object used.
   * 
   * \return Ref<SimpleTableInternal> A copy of the reference. 
   */
  virtual Ref<SimpleTableInternal> internal() = 0;

  /**
   * \brief Method allowing retrieval of a reference to the
   * ISimpleTableReaderWriter object used.
   * 
   * \return Ref<ISimpleTableReaderWriter> A copy of the reference. 
   */
  virtual Ref<ISimpleTableReaderWriter> readerWriter() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
