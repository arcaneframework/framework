// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshReaderMng.h                                             (C) 2000-2025 */
/*                                                                           */
/* Mesh reader manager.                                                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_MESHREADERMNG_H
#define ARCANE_CORE_MESHREADERMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/String.h"
#include "arcane/core/ArcaneTypes.h"
#include "arcane/core/Parallel.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Mesh reader manager.
 */
class ARCANE_CORE_EXPORT MeshReaderMng
{
  class Impl;

 public:
 
  MeshReaderMng(ISubDomain* sd);
  MeshReaderMng(const MeshReaderMng&) = delete;
  ~MeshReaderMng();
  const MeshReaderMng& operator=(const MeshReaderMng&) = delete;

 public:

  /*!
   * \brief Reads the mesh whose file name is \a file_name.
   *
   * \a file_name must have an extension and the reader used is based
   * on this extension.
   * The created mesh is associated with a sequential \a IParallelMng
   * and will be named \a mesh_name.
   *
   * This method throws an exception if the mesh cannot be read.
   */
  IMesh* readMesh(const String& mesh_name,const String& file_name);

  /*!
   * \brief Reads the mesh whose file name is \a file_name.
   *
   * \a file_name must have an extension and the reader used is based
   * on this extension.
   * The created mesh is associated with the parallelism manager
   * \a parallel_mng and will be named \a mesh_name.
   *
   * This method throws an exception if the mesh cannot be read.
   */
  IMesh* readMesh(const String& mesh_name,const String& file_name,
                  IParallelMng* parallel_mng);

  /*!
   * \brief If true, indicates that the unit system possibly present
   * in the file format is used (\a true by default).
   *
   * This method must be called before calling readMesh() for it to
   * be taken into account.
   */
  void setUseMeshUnit(bool v);
  //! Indicates whether the unit system present in the file is used
  bool isUseMeshUnit() const;

 private:

  Impl* m_p;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
