// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CartesianMeshPatch.h                                        (C) 2000-2023 */
/*                                                                           */
/* Informations sur un patch AMR d'un maillage cartésien.                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CARTESIANMESH_CARTESIANMESHPATCH_H
#define ARCANE_CARTESIANMESH_CARTESIANMESHPATCH_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"

#include "arcane/core/ItemTypes.h"
#include "arcane/core/VariableTypes.h"

#include "arcane/cartesianmesh/ICartesianMeshPatch.h"
#include "arcane/cartesianmesh/CellDirectionMng.h"
#include "arcane/cartesianmesh/FaceDirectionMng.h"
#include "arcane/cartesianmesh/NodeDirectionMng.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneCartesianMesh
 * \brief Informations par direction pour chaque genre d'entité du maillage.
 *
 * En cas de maillage AMR par patch, un numéro de patch est associé à chaque
 * direction. Ce numéro vaut (-1) pour les entités de niveau 0 ou dans le
 * cas où il n'y a pas d'AMR.
 */
class CartesianMeshPatch
: public TraceAccessor
, public ICartesianMeshPatch
{
  friend CartesianMeshImpl;
 public:
  CartesianMeshPatch(ICartesianMesh* cmesh,Integer patch_index);
  ~CartesianMeshPatch() override;
 public:
  CellGroup cells() override;
  Integer index() override
  {
    return m_amr_patch_index;
  }
  CellDirectionMng& cellDirection(eMeshDirection dir) override
  {
    return m_cell_directions[dir];
  }

  CellDirectionMng& cellDirection(Integer idir) override
  {
    return m_cell_directions[idir];
  }

  FaceDirectionMng& faceDirection(eMeshDirection dir) override
  {
    return m_face_directions[dir];
  }

  FaceDirectionMng& faceDirection(Integer idir) override
  {
    return m_face_directions[idir];
  }

  NodeDirectionMng& nodeDirection(eMeshDirection dir) override
  {
    return m_node_directions[dir];
  }

  NodeDirectionMng& nodeDirection(Integer idir) override
  {
    return m_node_directions[idir];
  }
  void checkValid() const override;
 private:
  void _internalComputeNodeCellInformations(Cell cell0,Real3 cell0_coord,VariableNodeReal3& nodes_coord);
  void _computeNodeCellInformations2D(Cell cell0,Real3 cell0_coord,VariableNodeReal3& nodes_coord);
  void _computeNodeCellInformations3D(Cell cell0,Real3 cell0_coord,VariableNodeReal3& nodes_coord);
 private:
  ICartesianMesh* m_mesh;
  CellDirectionMng m_cell_directions[3];
  FaceDirectionMng m_face_directions[3];
  NodeDirectionMng m_node_directions[3];
  Integer m_amr_patch_index;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

