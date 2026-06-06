// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMeshReader.h                                               (C) 2000-2025 */
/*                                                                           */
/* Interface of a mesh reading service.                                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IMESHREADER_H
#define ARCANE_CORE_IMESHREADER_H
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
 * \ingroup StandardService
 * \brief Interface of the service managing the reading of a mesh.
 */
class ARCANE_CORE_EXPORT IMeshReader
{
 public:

  //! Types of return codes for a read or write operation
  enum eReturnType
  {
    RTOk, //!< Operation successfully performed
    RTError, //!< Error during the operation
    /*! \brief Not relevant to the operation.
     * This means that the file format does not match
     * this reader or that the service does not support
     * this operation.
     */
    RTIrrelevant
  };

 public:

  virtual ~IMeshReader() = default; //!< Frees resources

 public:

  //! Checks if the service supports files with the extension \a str
  virtual bool allowExtension(const String& str) = 0;

  /*! \brief Reads a mesh from a file.
   *
   * Reads the geometry of a mesh from the file \a file_name
   * as well as the corresponding partitioning information
   * and constructs the corresponding mesh in \a mesh.
   *
   * If \a use_internal_partition is true, it means that the partitioning
   * has not yet been done and will be done by Arcane. In this case,
   * only one processor can read the mesh. However, the others
   * must still create all possible groups.
   * This argument is only useful in parallel.

   * If \a dir_name is not null, this path serves as the base for reading
   * meshes and partitioning information.
   */
  virtual eReturnType readMeshFromFile(IPrimaryMesh* mesh,
                                       const XmlNode& mesh_element,
                                       const String& file_name,
                                       const String& dir_name,
                                       bool use_internal_partition) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
