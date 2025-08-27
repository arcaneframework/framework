// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ICaseMeshReader.h                                           (C) 2000-2020 */
/*                                                                           */
/* Interface du service de lecture du maillage à partir du jeu de données.   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ICASEMESHREADER_H
#define ARCANE_ICASEMESHREADER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Informations nécessaires pour la lecture d'un fichier de maillage.
 *
 * \a isParallelRead() indique s'il est vrai que tous les rangs sur lequel le maillage
 * est défini vont lire le fichier et qu'il faut donc si possible le distribuer.
 * de manière équilibrée sur l'ensemble des rangs.
 *
 * \a format() indique le format du fichier. Par défaut il s'agit de l'extension
 * du nom de fichier. Par exemple, si le fichier est 'toto.vtk', alors le format
 * sera 'vtk'.
 */
class CaseMeshReaderReadInfo
{
 public:
  const String& fileName() const { return m_file_name; }
  const String& directoryName() const { return m_directory_name; }
  bool isParallelRead() const { return m_is_parallel_read; }
  const String& format() const { return m_format; }
  void setFileName(const String& v) { m_file_name = v; }
  void setDirectoryName(const String& v) { m_directory_name = v; }
  void setParallelRead(bool v) { m_is_parallel_read = v; }
  void setFormat(const String& v) { m_format = v; }
 private:
  String m_file_name;
  String m_directory_name;
  String m_format;
  bool m_is_parallel_read = true;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup StandardService
 * \brief Interface du service de lecture du maillage à partir du jeu de données.
 *
 * Cette interface est destinée à remplacer IMeshReader
 */
class ARCANE_CORE_EXPORT ICaseMeshReader
{
 public:

  //! Libère les ressources
  virtual ~ICaseMeshReader() = default;

 public:

  /*!
   * \brief Retourne un builder pour créer et lire le maillage dont les
   * informations sont spécifiées dans \a read_info.
   *
   * Si ce lecteur ne supporte pas le format spécifié dans \a read_info,
   * retourn nul.
   */
  virtual Ref<IMeshBuilder> createBuilder(const CaseMeshReaderReadInfo& read_info) const =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

