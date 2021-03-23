// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMesh.h                                                     (C) 2000-2020 */
/*                                                                           */
/* Informations de connectivité d'un maillage non structuré.                 */
/*---------------------------------------------------------------------------*/

#include "arcane/UnstructuredMeshConnectivity.h"

#include "arcane/utils/FatalErrorException.h"

#include "arcane/IMesh.h"
#include "arcane/IItemFamily.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemInternalConnectivityList* UnstructuredItemConnectivityBaseView::
 _getConnectivityList(IItemFamily* family)
{
  return family->_unstructuredItemInternalConnectivityList();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IItemFamily* UnstructuredMeshConnectivityViewTraits<Node>::
family(IMesh* m)
{
  return m->nodeFamily();
}

IItemFamily* UnstructuredMeshConnectivityViewTraits<Edge>::
family(IMesh* m)
{
  return m->edgeFamily();
}

IItemFamily* UnstructuredMeshConnectivityViewTraits<Face>::
family(IMesh* m)
{
  return m->faceFamily();
}

IItemFamily* UnstructuredMeshConnectivityViewTraits<Cell>::
family(IMesh* m)
{
  return m->cellFamily();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void UnstructuredMeshConnectivityView::
setMesh(IMesh* mesh)
{
  m_mesh = mesh;

  m_cell_node_connectivity_view.init(mesh);
  m_cell_edge_connectivity_view.init(mesh);
  m_cell_face_connectivity_view.init(mesh);

  m_face_node_connectivity_view.init(mesh);
  m_face_edge_connectivity_view.init(mesh);
  m_face_cell_connectivity_view.init(mesh);

  m_node_edge_connectivity_view.init(mesh);
  m_node_face_connectivity_view.init(mesh);
  m_node_cell_connectivity_view.init(mesh);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void UnstructuredMeshConnectivityView::
_checkValid() const
{
  if (!m_mesh)
    ARCANE_FATAL("Can not use unitialised UnstructuredMeshConnectivityView. Call the methode setMesh() before");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
