// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMeshArea.h                                                 (C) 2000-2025 */
/*                                                                           */
/* Interface of a mesh area.                                                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IMESHAREA_H
#define ARCANE_CORE_IMESHAREA_H
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
 * \ingroup Mesh
 *
 * \brief Interface of a mesh area.
 *
 * A mesh area is a subset of the mesh defined by
 * a list of cells and nodes.
 */
class ARCANE_CORE_EXPORT IMeshArea
{
 public:

  virtual ~IMeshArea() = default; //!< Releases resources

 public:

  //! Number of mesh nodes
  virtual Integer nbNode() = 0;

  //! Number of mesh cells
  virtual Integer nbCell() = 0;

 public:

  //! Associated sub-domain
  virtual ISubDomain* subDomain() = 0;

  //! Associated trace manager
  virtual ITraceMng* traceMng() = 0;

  //! Mesh to which the area belongs
  virtual IMesh* mesh() = 0;

 public:

  //! Group of all nodes
  virtual NodeGroup allNodes() = 0;

  //! Group of all cells
  virtual CellGroup allCells() = 0;

  //! Group of all nodes belonging to the domain
  virtual NodeGroup ownNodes() = 0;

  //! Group of all cells belonging to the domain
  virtual CellGroup ownCells() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
