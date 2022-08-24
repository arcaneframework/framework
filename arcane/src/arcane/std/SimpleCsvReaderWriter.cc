// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SimpleCsvReaderWriter.cc                                    (C) 2000-2022 */
/*                                                                           */
/* Classe permettant de lire et d'écrire un fichier au format csv.           */
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
  if (!SimpleTableReaderWriterUtils::createDirectoryOnlyProcess0(m_simple_table_internal->m_parallel_mng, dst)) {
    return false;
  }

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

  // S'il n'y a pas de fichier, on retourne false.
  if (!_openFile(stream, src, file_name + "." + fileType())) {
    return false;
  }

  std::string line;

  // S'il n'y a pas de première ligne, on arrête là.
  // Un fichier écrit par SimpleCsvOutput possède toujours au
  // moins une ligne.
  if (!std::getline(stream, line)) {
    _closeFile(stream);
    return false;
  }

  // Sinon, on a la ligne d'entête, contenant les noms
  // des colonnes (et le nom du tableau).
  String ligne(line);

  {
    StringUniqueArray tmp;
    ligne.split(tmp, m_separator);
    // Normalement, tmp[0] existe toujours (peut-être = à "" (vide)).
    m_simple_table_internal->m_table_name = tmp[0];
    m_simple_table_internal->m_column_names = tmp.subConstView(1, tmp.size());
  }

  // S'il n'y a pas d'autres lignes, c'est qu'il n'y a que des
  // colonnes vides (ou aucunes colonnes) et aucunes lignes.
  if (!std::getline(stream, line)) {
    _closeFile(stream);
    return true;
  }

  // Maintenant que l'on a le nombre de colonnes, on peut définir
  // la dimension 2 du tableau de valeurs.
  m_simple_table_internal->m_values.resize(1, m_simple_table_internal->m_column_names.size());

  Integer compt_line = 0;

  do {
    // On n'a pas le nombre de lignes en avance,
    // donc on doit resize à chaque tour.
    m_simple_table_internal->m_values.resize(compt_line + 1);

    // On split la ligne récupérée.
    StringUniqueArray splitted_line;
    String ligne(line);
    ligne.split(splitted_line, m_separator);

    // Le premier élement est le nom de ligne.
    m_simple_table_internal->m_row_names.add(splitted_line[0]);

    // Les autres élements sont des Reals.
    for (Integer i = 1; i < splitted_line.size(); i++) {
      std::string std_string = splitted_line[i].localstr();
      std::size_t pos_comma = std_string.find(',');

      if(pos_comma != std::string::npos) {
        std_string[pos_comma] = '.';
      }

      m_simple_table_internal->m_values[compt_line][i - 1] = std::stod(std_string);
    }

    compt_line++;
  } while (std::getline(stream, line));

  _closeFile(stream);

  // On n'a pas sauvegardé les tailles des lignes/colonnes donc on met la taille max
  // pour chaque ligne/colonne.
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
    ARCANE_FATAL("La réference passée en paramètre est Null.");
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
  // On enregistre les infos du stream pour les restaurer à la fin.
  std::ios_base::fmtflags save_flags = stream.flags();
  std::streamsize save_prec = stream.precision();

  if (m_is_fixed_print) {
    stream << std::setiosflags(std::ios::fixed);
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
