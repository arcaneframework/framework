// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AMRPatchPosition.h                                        (C) 2000-2025 */
/*                                                                           */
/* Informations sur un patch AMR d'un maillage cartésien.                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CARTESIANMESH_AMRPATCHPOSITION_H
#define ARCANE_CARTESIANMESH_AMRPATCHPOSITION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Vector3.h"
#include "arcane/cartesianmesh/CartesianMeshGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_CARTESIANMESH_EXPORT AMRPatchPosition
{
 public:

  AMRPatchPosition();
  // TODO Faire operator=
  AMRPatchPosition(const AMRPatchPosition& src);

  ~AMRPatchPosition();

 public:

  Integer level() const;
  void setLevel(Integer level);

  Int64x3 minPoint() const;
  void setMinPoint(Int64x3 min_point);
  Int64x3 maxPoint() const;
  void setMaxPoint(Int64x3 max_point);

  Integer overlapLayerSize() const;
  void setOverlapLayerSize(Integer layer_size);

  Int64x3 minPointWithOverlap() const;
  Int64x3 maxPointWithOverlap() const;


  Int64 nbCells() const;
  std::pair<AMRPatchPosition, AMRPatchPosition> cut(Int64 cut_point, Integer dim) const;
  bool canBeFusion(const AMRPatchPosition& other_patch) const;
  void fusion(const AMRPatchPosition& other_patch);
  bool isNull() const;

  AMRPatchPosition patchUp(Integer dim) const;
  AMRPatchPosition patchDown(Integer dim) const;

  Int64x3 length() const;

  bool isIn(Integer x, Integer y, Integer z) const;
  bool isInWithOverlap(Integer x, Integer y, Integer z) const;
  bool isInWithOverlap(Integer x, Integer y, Integer z, Integer overlap) const;

  bool haveIntersection(const AMRPatchPosition& other) const;

 private:

  Integer m_level;
  Int64x3 m_min_point;
  Int64x3 m_max_point;
  Integer m_overlap_layer_size;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

