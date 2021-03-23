// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshAccessor.h                                              (C) 2000-2019 */
/*                                                                           */
/* Accès aux informations d'un maillage.                                     */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/FatalErrorException.h"

#include "arcane/MeshAccessor.h"
#include "arcane/ISubDomain.h"
#include "arcane/IMesh.h"
#include "arcane/IMeshSubMeshTransition.h"
#include "arcane/ItemGroup.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshAccessor::
MeshAccessor(ISubDomain* sd)
: m_mesh_handle(sd->defaultMeshHandle())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshAccessor::
MeshAccessor(IMesh* mesh)
{
  if (!mesh)
    ARCANE_FATAL("Can not create MeshAccessor with null mesh. Use constructor with MeshHandle");
  m_mesh_handle = mesh->handle();
}

MeshAccessor::
MeshAccessor(const MeshHandle& mesh_handle)
: m_mesh_handle(mesh_handle)
{
}

Integer MeshAccessor::nbCell() const { return mesh()->nbCell(); }
Integer MeshAccessor::nbFace() const { return mesh()->nbFace(); }
Integer MeshAccessor::nbEdge() const { return mesh()->nbEdge(); }
Integer MeshAccessor::nbNode() const { return mesh()->nbNode(); }
VariableNodeReal3& MeshAccessor::nodesCoordinates() const
{
  return mesh()->toPrimaryMesh()->nodesCoordinates();
}
NodeGroup MeshAccessor::allNodes() const { return mesh()->allNodes(); }
EdgeGroup MeshAccessor::allEdges() const { return mesh()->allEdges(); }
FaceGroup MeshAccessor::allFaces() const { return mesh()->allFaces(); }
CellGroup MeshAccessor::allCells() const { return mesh()->allCells(); }
FaceGroup MeshAccessor::outerFaces() const { return mesh()->outerFaces(); }
NodeGroup MeshAccessor::ownNodes() const { return mesh()->ownNodes(); }
CellGroup MeshAccessor::ownCells() const { return mesh()->ownCells(); }
FaceGroup MeshAccessor::ownFaces() const { return mesh()->ownFaces(); }
EdgeGroup MeshAccessor::ownEdges() const { return mesh()->ownEdges(); }

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

