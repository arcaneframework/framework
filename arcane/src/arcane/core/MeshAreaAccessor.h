// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshAreaAccessor.h                                          (C) 2000-2025 */
/*                                                                           */
/* Access to information about a mesh area.                                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_MESHAREAACCESSOR_H
#define ARCANE_CORE_MESHAREAACCESSOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"
#include "arcane/core/ItemTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Access to information about a mesh area.
 */
class ARCANE_CORE_EXPORT MeshAreaAccessor
{
 public:

  explicit MeshAreaAccessor(IMeshArea* mesh_area);
  ~MeshAreaAccessor();

 public:

  //! Mesh area accessed by this accessor
  IMeshArea* meshArea();

  //! Sets the mesh area accessed by this accessor to \a mesh_area
  void setMeshArea(IMeshArea* mesh_area);

 public:

  //! Number of nodes in the mesh
  Integer nbNode();

  //! Number of cells in the mesh
  Integer nbCell();

 public:

  //! Group of all nodes in the area
  NodeGroup allNodes();

  //! Group of all cells in the area
  CellGroup allCells();

  //! Group of all own nodes in the area
  NodeGroup ownNodes();

  //! Group of all own cells in the area
  CellGroup ownCells();

 private:

  IMeshArea* m_mesh_area = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
