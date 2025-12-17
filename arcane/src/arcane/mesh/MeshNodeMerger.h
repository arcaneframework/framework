// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshNodeMerger.h                                            (C) 2000-2025 */
/*                                                                           */
/* Fusion de nœuds d'un maillage.                                            */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESH_MESHNODEMERGER_H
#define ARCANE_MESH_MESHNODEMERGER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/mesh/MeshGlobal.h"

#include <map>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class IMesh;
}

namespace Arcane::mesh
{
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class NodeFamily;
class EdgeFamily;
class FaceFamily;
class CellFamily;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Fusion de nœuds d'un maillage.
 *
 * L'implémentation actuelle ne traite que les maillages 'classiques'
 * et ne traite que les familles de noeuds, arêtes, faces et mailles.
 *
 * De plus, pour la partie 3D n'a pas été testée. Elle devrait fonctionner
 * tant que les arêtes ne sont pas actives.
 * Idem pour la partie parallèle mais elle n'a pas été testée non plus.
 */
class MeshNodeMerger
: public TraceAccessor
{
public:

  explicit MeshNodeMerger(IMesh* mesh);

 public:

  void mergeNodes(Int32ConstArrayView nodes_local_id,
                  Int32ConstArrayView nodes_to_merge_local_id,
                  bool allow_non_corresponding_face = false);

 private:

  IMesh* m_mesh = nullptr;
  NodeFamily* m_node_family = nullptr;
  EdgeFamily* m_edge_family = nullptr;
  FaceFamily* m_face_family = nullptr;
  CellFamily* m_cell_family = nullptr;
  std::map<Node,Node> m_nodes_correspondance;
  std::map<Face,Face> m_faces_correspondance;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
