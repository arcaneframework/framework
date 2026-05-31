// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SimpleCsvReaderWriter.cc                                    (C) 2000-2022 */
/*                                                                           */
/* Class allowing reading and writing a file in CSV format.                  */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/std/SimpleCsvReaderWriter.h"

#include "arcane/Directory.h"
#include "arcane/utils/Iostream.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool SimpleCsvReaderWriter::
writeTable(const Directory& dst, const String& file_name)
{
  std::ofstream ofile((dst.file(file_name) + "." + fileType()).localstr());
  if (ofile.fail())
    return false;

  _print(ofile);

  ofile.close();
  return true;
}

bool SimpleCsvReaderWriter::
readTable(const Directory& src, const String& file_name)
{
  clearInternal();

  std::ifstream stream;

  // If there is no file, we return false.
  if (!_openFile(stream, src, file_name + "." + fileType())) {
    return false;
  }

  std::string line;

  // If there is no first line, we stop here.
  // A file written by SimpleCsvOutput always has at least one line.
  if (!std::getline(stream, line)) {
    _closeFile(stream);
    return false;
  }

  // Otherwise, we have the header line, containing the column names
  // (and the table name).
  String ligne(line);

  {
    StringUniqueArray tmp;
    ligne.split(tmp, m_separator);

    if (tmp.size() == 0) {
      _closeFile(stream);
      return false;
    }

    m_simple_table_internal->m_table_name = tmp[0];
    m_simple_table_internal->m_column_names = tmp.subConstView(1, tmp.size());
  }

  // If there are no other lines, it means there are only empty columns
  // (or no columns) and no lines.
  if (!std::getline(stream, line)) {
    _closeFile(stream);
    return true;
  }

  // Now that we have the number of columns, we can define dimension 2
  // of the values array.
  m_simple_table_internal->m_values.resize(1, m_simple_table_internal->m_column_names.size());

  Integer compt_line = 0;

  do {
    // We don't know the number of lines in advance, so we must resize
    // each time.
    m_simple_table_internal->m_values.resize(compt_line + 1);

    // We split the retrieved line.
    StringUniqueArray splitted_line;
    String ligne(line);
    ligne.split(splitted_line, m_separator);

    // If there is an empty line, we skip it.
    if (splitted_line.size() == 0) {
      continue;
    }

    // If the number of columns in the line does not match the number of
    // column names, there is an error in the file.
    if (splitted_line.size() != m_simple_table_internal->m_column_names.size() + 1) {
      _closeFile(stream);
      return false;
    }

    // The first element is the row name.
    m_simple_table_internal->m_row_names.add(splitted_line[0]);

    // The other elements are Reals.
    for (Integer i = 1; i < splitted_line.size(); i++) {
      std::string std_string = splitted_line[i].localstr();
      std::size_t pos_comma = std_string.find(',');

      if (pos_comma != std::string::npos) {
        std_string[pos_comma] = '.';
      }

      m_simple_table_internal->m_values[compt_line][i - 1] = std::stod(std_string);
    }

    compt_line++;
  } while (std::getline(stream, line));

  _closeFile(stream);

  // We haven't saved the row/column sizes so we set the max size
  // for each row/column.
  m_simple_table_internal->m_row_sizes.resize(m_simple_table_internal->m_row_names.size());
  m_simple_table_internal->m_row_sizes.fill(m_simple_table_internal->m_values.dim2Size());

  m_simple_table_internal->m_column_sizes.resize(m_simple_table_internal->m_column_names.size());
  m_simple_table_internal->m_column_sizes.fill(m_simple_table_internal->m_values.dim1Size());

  return true;
}

void SimpleCsvReaderWriter::
clearInternal()
{
  m_simple_table_internal->clear();
}

void SimpleCsvReaderWriter::
print()
{
  _print(std::cout);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer SimpleCsvReaderWriter::
precision()
{
  return m_precision_print;
}

void SimpleCsvReaderWriter::
setPrecision(Integer precision)
{
  if (precision < 1)
    m_precision_print = 1;
  else if (precision > (std::numeric_limits<Real>::max_digits10))
    m_precision_print = (std::numeric_limits<Real>::max_digits10);
  else
    m_precision_print = precision;
}

bool SimpleCsvReaderWriter::
isFixed()
{
  return m_is_fixed_print;
}

void SimpleCsvReaderWriter::
setFixed(bool fixed)
{
  m_is_fixed_print = fixed;
}

bool SimpleCsvReaderWriter::
isForcedToUseScientificNotation()
{
  return m_scientific_notation;
}

void SimpleCsvReaderWriter::
setForcedToUseScientificNotation(bool use_scientific)
{
  m_scientific_notation = use_scientific;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<SimpleTableInternal> SimpleCsvReaderWriter::
internal()
{
  return m_simple_table_internal;
}

void SimpleCsvReaderWriter::
setInternal(const Ref<SimpleTableInternal>& simple_table_internal)
{
  if (simple_table_internal.isNull())
    ARCANE_FATAL("The reference passed as a parameter is Null.");
  m_simple_table_internal = simple_table_internal;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool SimpleCsvReaderWriter::
_openFile(std::ifstream& stream, Directory directory, const String& file)
{
  stream.open(directory.file(file).localstr(), std::ifstream::in);
  return stream.good();
}

void SimpleCsvReaderWriter::
_closeFile(std::ifstream& stream)
{
  stream.close();
}

void SimpleCsvReaderWriter::
_print(std::ostream& stream)
{
  // We save the stream info to restore it at the end.
  std::ios_base::fmtflags save_flags = stream.flags();
  std::streamsize save_prec = stream.precision();

  if (m_is_fixed_print) {
    stream << std::setiosflags(std::ios::fixed);
  }
  if (m_scientific_notation) {
    stream << std::scientific;
  }
  stream << std::setprecision(m_precision_print);

  stream << m_simple_table_internal->m_table_name << m_separator;

  for (Integer j = 0; j < m_simple_table_internal->m_column_names.size(); j++) {
    stream << m_simple_table_internal->m_column_names[j] << m_separator;
  }
  stream << std::endl;

  for (Integer i = 0; i < m_simple_table_internal->m_values.dim1Size(); i++) {
    stream << m_simple_table_internal->m_row_names[i] << m_separator;
    ConstArrayView<Real> view = m_simple_table_internal->m_values[i];
    for (Integer j = 0; j < m_simple_table_internal->m_values.dim2Size(); j++) {
      stream << view[j] << m_separator;
    }
    stream << std::endl;
  }

  stream.flags(save_flags);
  stream.precision(save_prec);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
