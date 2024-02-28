// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ExtraGhostCellsBuilder.h                                    (C) 2000-2024 */
/*                                                                           */
/* Construction des mailles fantômes supplémentaires.                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESH_EXTRAGHOSTCELLSBUILDER_H
#define ARCANE_MESH_EXTRAGHOSTCELLSBUILDER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/Array.h"

#include "arcane/mesh/MeshGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class IExtraGhostCellsBuilder;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{
class DynamicMesh;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Construction des mailles fantômes supplémentaires.
 */
class ExtraGhostCellsBuilder
: public TraceAccessor
{
 public:

  explicit ExtraGhostCellsBuilder(DynamicMesh* mesh);

 public:

  void computeExtraGhostCells();
  void addExtraGhostCellsBuilder(IExtraGhostCellsBuilder* builder);
  void removeExtraGhostCellsBuilder(IExtraGhostCellsBuilder* builder);
  bool hasBuilder() const;

 private:

  DynamicMesh* m_mesh = nullptr;
  UniqueArray<IExtraGhostCellsBuilder*> m_builders;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
