// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ICaseMeshReader.h                                           (C) 2000-2025 */
/*                                                                           */
/* Interface for the mesh reading service from the dataset.                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ICASEMESHREADER_H
#define ARCANE_CORE_ICASEMESHREADER_H
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
 * \brief Necessary information for reading a mesh file.
 *
 * \a isParallelRead() indicates whether it is true that all ranks on which the mesh
 * is defined will read the file and therefore must distribute it if possible.
 * in a balanced manner across all ranks.
 *
 * \a format() indicates the file format. By default, it is the file extension.
 * For example, if the file is 'toto.vtk', then the format
 * will be 'vtk'.
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
 * \brief Interface for the mesh reading service from the dataset.
 *
 * This interface is intended to replace IMeshReader
 */
class ARCANE_CORE_EXPORT ICaseMeshReader
{
 public:

  //! Deallocates resources
  virtual ~ICaseMeshReader() = default;

 public:

  /*!
   * \brief Returns a builder to create and read the mesh whose
   * information is specified in \a read_info.
   *
   * If this reader does not support the format specified in \a read_info,
   * it returns null.
   */
  virtual Ref<IMeshBuilder> createBuilder(const CaseMeshReaderReadInfo& read_info) const =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
