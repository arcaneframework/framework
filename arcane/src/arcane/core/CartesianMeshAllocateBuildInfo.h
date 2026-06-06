// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CartesianMeshAllocateBuildInfo.h                            (C) 2000-2023 */
/*                                                                           */
/* Information for allocating entities of a Cartesian mesh.                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_CARTESIANMESHALLOCATEBUILDINFO_H
#define ARCANE_CORE_CARTESIANMESHALLOCATEBUILDINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ItemTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class CartesianMeshAllocateBuildInfoInternal;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Information for allocating entities of an unstructured mesh.
 *
 * This class allows specifying the cells that will be added during
 * the initial mesh allocation.
 * You must call setMeshDimension() to specify the mesh dimension
 * then call addCell() for each cell you wish to add. Once
 * all cells have been added, you must call allocateMesh().
 */
class ARCANE_CORE_EXPORT CartesianMeshAllocateBuildInfo
{
 public:

  class Impl;
  class Intenrnal;

 public:

  explicit CartesianMeshAllocateBuildInfo(IPrimaryMesh* mesh);
  ~CartesianMeshAllocateBuildInfo();

 public:

  CartesianMeshAllocateBuildInfo(CartesianMeshAllocateBuildInfo&& from) = delete;
  CartesianMeshAllocateBuildInfo(const CartesianMeshAllocateBuildInfo& from) = delete;
  CartesianMeshAllocateBuildInfo& operator=(CartesianMeshAllocateBuildInfo&& from) = delete;
  CartesianMeshAllocateBuildInfo& operator=(const CartesianMeshAllocateBuildInfo& from) = delete;

 public:

  //! Sets the information for a 3D mesh.
  ARCANE_DEPRECATED_REASON("Y2023: Use overload with first_own_cell_offset instead")
  void setInfos3D(std::array<Int64, 3> global_nb_cells,
                  std::array<Int32, 3> own_nb_cells,
                  Int64 cell_unique_id_offset,
                  Int64 node_unique_id_offset);

  //! Sets the information for a 3D mesh.
  void setInfos3D(const Int64x3& global_nb_cells,
                  const Int32x3& own_nb_cells,
                  const Int64x3& first_own_cell_offset,
                  Int64 cell_unique_id_offset);

  //! Sets the information for a 2D mesh.
  ARCANE_DEPRECATED_REASON("Y2023: Use overload with first_own_cell_offset instead")
  void setInfos2D(std::array<Int64, 2> global_nb_cells,
                  std::array<Int32, 2> own_nb_cells,
                  Int64 cell_unique_id_offset,
                  Int64 node_unique_id_offset);

  //! Sets the information for a 2D mesh.
  void setInfos2D(const Int64x2& global_nb_cells,
                  const Int32x2& own_nb_cells,
                  const Int64x2& first_own_cell_offset,
                  Int64 cell_unique_id_offset);

  /*!
   * \brief Allocates the mesh.
   *
   * It is necessary to have called setInfos() beforehand.
   *
   * This method is collective.
   */
  void allocateMesh();

 public:

  //! Internal part reserved for Arcane
  CartesianMeshAllocateBuildInfoInternal* _internal();

 private:

  Impl* m_p = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
