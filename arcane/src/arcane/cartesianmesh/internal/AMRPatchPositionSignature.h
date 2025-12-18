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
  AMRPatchPositionSignature(const AMRPatchPosition& patch, ICartesianMesh* cmesh, AMRPatchPositionLevelGroup* all_patches);
  ~AMRPatchPositionSignature() = default;

 private:

  AMRPatchPositionSignature(const AMRPatchPosition& patch, ICartesianMesh* cmesh, AMRPatchPositionLevelGroup* all_patches, Int32 nb_cut);

 public:

  void compress();
  void fillSig();
  bool isValid() const;
  bool canBeCut() const;
  void compute();
  Real efficacity() const;
  std::pair<AMRPatchPositionSignature, AMRPatchPositionSignature> cut(Integer dim, CartCoord cut_point) const;

  ConstArrayView<CartCoord> sigX() const;
  ConstArrayView<CartCoord> sigY() const;
  ConstArrayView<CartCoord> sigZ() const;
  AMRPatchPosition patch() const;
  ICartesianMesh* mesh() const;
  bool stopCut() const;
  void setStopCut(bool stop_cut);
  bool isComputed() const;

 private:

  bool m_is_null;
  AMRPatchPosition m_patch;
  ICartesianMesh* m_mesh;
  Int32 m_nb_cut;
  bool m_stop_cut;

  ICartesianMeshNumberingMngInternal* m_numbering;

  bool m_have_cells;
  bool m_is_computed;

  UniqueArray<CartCoord> m_sig_x;
  UniqueArray<CartCoord> m_sig_y;
  UniqueArray<CartCoord> m_sig_z;

  AMRPatchPositionLevelGroup* m_all_patches;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
