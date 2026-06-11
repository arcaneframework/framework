// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* GeomShapeMng.cc                                             (C) 2000-2026 */
/*                                                                           */
/* Class managing the GeomShapes of a mesh.                                  */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/IMesh.h"

#include "arcane/geometry/GeomShapeMng.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::geometric
{
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

GeomShapeMng::
GeomShapeMng(IMesh* mesh, const String& cell_shape_name)
: m_name(cell_shape_name)
, m_cell_shape_nodes(VariableBuildInfo(mesh, cell_shape_name, IVariable::PNoDump))
, m_cell_shape_faces(VariableBuildInfo(mesh, cell_shape_name + "Face", IVariable::PNoDump))
, m_cell_shape_centers(VariableBuildInfo(mesh, cell_shape_name + "Center", IVariable::PNoDump))
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

GeomShapeMng::
GeomShapeMng(IMesh* mesh)
: m_name("GenericElement")
, m_cell_shape_nodes(VariableBuildInfo(mesh, "GenericElement", IVariable::PNoDump))
, m_cell_shape_faces(VariableBuildInfo(mesh, "GenericElementFace", IVariable::PNoDump))
, m_cell_shape_centers(VariableBuildInfo(mesh, "GenericElementCenter", IVariable::PNoDump))
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

GeomShapeMng::
GeomShapeMng(const GeomShapeMng& rhs)
: m_name(rhs.m_name)
, m_cell_shape_nodes(rhs.m_cell_shape_nodes)
, m_cell_shape_faces(rhs.m_cell_shape_faces)
, m_cell_shape_centers(rhs.m_cell_shape_centers)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void GeomShapeMng::
initialize()
{
  IMesh* mesh = m_cell_shape_nodes.variable()->meshHandle().mesh();
  //TODO: we must use the globalConnectivity() of IItemFamily
  // but for now this is not calculated correctly
  // during init.
  if (mesh->dimension() == 2) {
    // In 2D, we do not have cells containing more nodes than quads
    m_cell_shape_nodes.resize(4);
    //TODO: Check if this is necessary.
    m_cell_shape_faces.resize(4);
  }
  else {
    m_cell_shape_nodes.resize(ItemStaticInfo::MAX_CELL_NODE);
    m_cell_shape_faces.resize(ItemStaticInfo::MAX_CELL_FACE);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::geometric

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
