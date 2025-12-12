// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AMRPatchPositionSignature.h                                 (C) 2000-2025 */
/*                                                                           */
/* Calcul des signatures d'une position de patch.                            */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CARTESIANMESH_INTERNAL_AMRPATCHPOSITIONSIGNATURE_H
#define ARCANE_CARTESIANMESH_INTERNAL_AMRPATCHPOSITIONSIGNATURE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/cartesianmesh/CartesianMeshGlobal.h"

#include "arcane/utils/UniqueArray.h"

#include "arcane/cartesianmesh/AMRPatchPosition.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class AMRPatchPositionSignature
{
 public:

  AMRPatchPositionSignature();
  AMRPatchPositionSignature(AMRPatchPosition patch, ICartesianMesh* cmesh, AMRPatchPositionLevelGroup* all_patches);
  AMRPatchPositionSignature(AMRPatchPosition patch, ICartesianMesh* cmesh, AMRPatchPositionLevelGroup* all_patches, Integer nb_cut);
  ~AMRPatchPositionSignature();

 public:

  void compress();
  void fillSig();
  bool isValid() const;
  bool canBeCut() const;
  void compute();
  Real efficacity() const;
  std::pair<AMRPatchPositionSignature, AMRPatchPositionSignature> cut(Integer dim, Integer cut_point) const;
  bool isIn(Integer x, Integer y, Integer z) const;

  ConstArrayView<Integer> sigX() const;
  ConstArrayView<Integer> sigY() const;
  ConstArrayView<Integer> sigZ() const;
  AMRPatchPosition patch() const;
  ICartesianMesh* mesh() const;
  bool stopCut() const;
  void setStopCut(bool stop_cut);
  bool isComputed() const;

 private:

  bool m_is_null;
  AMRPatchPosition m_patch;
  ICartesianMesh* m_mesh;
  Integer m_nb_cut;
  bool m_stop_cut;

  ICartesianMeshNumberingMngInternal* m_numbering;

  bool m_have_cells;
  bool m_is_computed;

  UniqueArray<Integer> m_sig_x;
  UniqueArray<Integer> m_sig_y;
  UniqueArray<Integer> m_sig_z;

  AMRPatchPositionLevelGroup* m_all_patches;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
