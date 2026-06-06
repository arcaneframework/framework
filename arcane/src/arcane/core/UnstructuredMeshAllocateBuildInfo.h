// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* UnstructuredMeshAllocateBuildInfo.h                         (C) 2000-2023 */
/*                                                                           */
/* Information for allocating entities of an unstructured mesh.              */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_UNSTRUCTUREDMESHALLOCATEBUILDINFO_H
#define ARCANE_CORE_UNSTRUCTUREDMESHALLOCATEBUILDINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ItemTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class UnstructuredMeshAllocateBuildInfoInternal;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Information for allocating entities of an unstructured mesh.
 *
 * This class allows specifying the cells that will be added during
 * the initial mesh allocation.
 * You must call setMeshDimension() to specify the mesh dimension,
 * then call addCell() for each cell you wish to add. Once
 * all cells have been added, you must call allocateMesh().
 */
class ARCANE_CORE_EXPORT UnstructuredMeshAllocateBuildInfo
{
 public:

  class Impl;
  class Intenrnal;

 public:

  explicit UnstructuredMeshAllocateBuildInfo(IPrimaryMesh* mesh);
  ~UnstructuredMeshAllocateBuildInfo();

 public:

  UnstructuredMeshAllocateBuildInfo(UnstructuredMeshAllocateBuildInfo&& from) = delete;
  UnstructuredMeshAllocateBuildInfo(const UnstructuredMeshAllocateBuildInfo& from) = delete;
  UnstructuredMeshAllocateBuildInfo& operator=(UnstructuredMeshAllocateBuildInfo&& from) = delete;
  UnstructuredMeshAllocateBuildInfo& operator=(const UnstructuredMeshAllocateBuildInfo& from) = delete;

 public:

  /*!
   * \brief Pre-allocate the memory.
   *
   * Pre-allocates the arrays containing the connectivity to hold \a nb_cell
   * cells and \a nb_connectivity_node for the list of cell nodes.
   *
   * This method is optional and is only useful for
   * optimizing memory management.
   *
   * For example, if we know that our mesh will contain 300 quadrangles
   * then we can use preAllocate(300,300*4).
   */
  void preAllocate(Int32 nb_cell, Int64 nb_connectivity_node);

  //! Sets the mesh dimension
  void setMeshDimension(Int32 v);

  //! Adds a cell to the mesh
  void addCell(ItemTypeId type_id, Int64 cell_uid, SmallSpan<const Int64> nodes_uid);

  /*!
   * \brief Allocates the mesh with the cells added during the call to addCell().
   *
   * This method is collective.
   */
  void allocateMesh();

 public:

  //! Internal part reserved for Arcane
  UnstructuredMeshAllocateBuildInfoInternal* _internal();

 private:

  Impl* m_p = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
