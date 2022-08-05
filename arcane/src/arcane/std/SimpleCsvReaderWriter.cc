// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* TODO                                   (C) 2000-2022 */
/*                                                                           */
/* .   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/std/SimpleCsvReaderWriter.h"

#include <arcane/Directory.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool SimpleCsvReaderWriter::
writeTable(Directory dst, const String& file_name)
{
  ARCANE_CHECK_PTR(m_sti);
  if(!SimpleTableReaderWriterUtils::createDirectoryOnlyP0(m_sti->m_sub_domain, dst)) {
    return false;
  }

  std::ofstream ofile((dst.file(file_name)+"."+typeFile()).localstr());
  if (ofile.fail())
    return false;

  _print(ofile);

  ofile.close();
  return true;
}

bool SimpleCsvReaderWriter::
readTable(Directory src, const String& file_name)
{
  ARCANE_CHECK_PTR(m_sti);
  clearInternal();

  std::ifstream stream;

  // Pas de fichier, pas de chocolats.
  if(!_openFile(stream, src, file_name+"."+typeFile())) {
    return false;
  }

  std::string line;

  // S'il n'y a pas de première ligne, on arrete là.
  // Un fichier écrit par SimpleCsvOutput possède toujours au
  // moins une ligne.
  if(!std::getline(stream, line)) {
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
    m_sti->m_name_tab = tmp[0];
    m_sti->m_name_columns = tmp.subConstView(1, tmp.size());
  }

  // S'il n'y a pas d'autres lignes, c'est qu'il n'y a que des 
  // colonnes vides (ou aucunes colonnes) et aucunes lignes.
  if(!std::getline(stream, line)) {
    _closeFile(stream);
    return true;
  }

  // Maintenant que l'on a le nombre de colonnes, on peut définir
  // la dimension 2 du tableau de valeurs.
  m_sti->m_values_csv.resize(1, m_sti->m_name_columns.size());

  Integer compt_line = 0;

  do{
    // On n'a pas le nombre de lignes en avance,
    // donc on doit resize à chaque tour.
    m_sti->m_values_csv.resize(compt_line+1);

    // On split la ligne récupéré.
    StringUniqueArray splitted_line;
    String ligne(line);
    ligne.split(splitted_line, m_separator);

    // Le premier élement est le nom de ligne.
    m_sti->m_name_rows.add(splitted_line[0]);

    // Les autres élements sont des Reals.
    for(Integer i = 1; i < splitted_line.size(); i++){
      m_sti->m_values_csv[compt_line][i-1] = std::stod(splitted_line[i].localstr());
    }

    compt_line++;
  } while(std::getline(stream, line));

  _closeFile(stream);

  m_sti->m_size_rows.resize(m_sti->m_name_rows.size());
  m_sti->m_size_rows.fill(m_sti->m_values_csv.dim2Size());

  m_sti->m_size_columns.resize(m_sti->m_name_columns.size());
  m_sti->m_size_columns.fill(m_sti->m_values_csv.dim1Size());

  return true;
}

void SimpleCsvReaderWriter::
clearInternal()
{
  ARCANE_CHECK_PTR(m_sti);
  m_sti->clear();
}

void SimpleCsvReaderWriter::
print()
{
  ARCANE_CHECK_PTR(m_sti);
  _print(std::cout);
}

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
  else if (precision > (std::numeric_limits<Real>::digits10 + 1))
    m_precision_print = (std::numeric_limits<Real>::digits10 + 1);
  else
    m_precision_print = precision;
}

bool SimpleCsvReaderWriter::
fixed()
{
  return m_is_fixed_print;
}

void SimpleCsvReaderWriter::
setFixed(bool fixed)
{
  m_is_fixed_print = fixed;
}


SimpleTableInternal* SimpleCsvReaderWriter::
internal() 
{
  return m_sti;
}

void SimpleCsvReaderWriter::
setInternal(SimpleTableInternal* sti) 
{
  ARCANE_CHECK_PTR(sti);
  m_sti = sti;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool SimpleCsvReaderWriter::
_openFile(std::ifstream& stream, Directory dir, const String& file)
{
  stream.open(dir.file(file).localstr(), std::ifstream::in);
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
  ARCANE_CHECK_PTR(m_sti);
  // On enregistre les infos du stream pour les restaurer à la fin.
  std::ios_base::fmtflags save_flags = stream.flags();
  std::streamsize save_prec = stream.precision();

  if (m_is_fixed_print) {
    stream << std::setiosflags(std::ios::fixed);
  }
  stream << std::setprecision(m_precision_print);

  stream << m_sti->m_name_tab << m_separator;

  for (Integer j = 0; j < m_sti->m_name_columns.size(); j++) {
    stream << m_sti->m_name_columns[j] << m_separator;
  }
  stream << std::endl;

  for (Integer i = 0; i < m_sti->m_values_csv.dim1Size(); i++) {
    stream << m_sti->m_name_rows[i] << m_separator;
    ConstArrayView<Real> view = m_sti->m_values_csv[i];
    for (Integer j = 0; j < m_sti->m_values_csv.dim2Size(); j++) {
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
