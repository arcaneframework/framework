// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AMRPatchPositionSignatureCut.h                              (C) 2000-2025 */
/*                                                                           */
/* Méthodes de découpages de patchs selon leurs signatures.                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CARTESIANMESH_INTERNAL_AMRPATCHPOSITIONSIGNATURECUT_H
#define ARCANE_CARTESIANMESH_INTERNAL_AMRPATCHPOSITIONSIGNATURECUT_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/cartesianmesh/CartesianMeshGlobal.h"
#include "arcane/utils/UniqueArray.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class AMRPatchPositionSignatureCut
{
 public:
  AMRPatchPositionSignatureCut();
  ~AMRPatchPositionSignatureCut();

 public:

  static CartCoordType _cutDim(ConstArrayView<CartCoordType> sig);
  static std::pair<AMRPatchPositionSignature, AMRPatchPositionSignature> cut(const AMRPatchPositionSignature& sig);
  static void cut(UniqueArray<AMRPatchPositionSignature>& sig_array_a);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

