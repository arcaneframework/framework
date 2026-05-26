// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AMRPatchPositionSignatureCut.h                              (C) 2000-2026 */
/*                                                                           */
/* Patch cutting methods based on their signatures.                          */
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

/*!
 * \brief Class allowing a patch to be cut into several smaller patches.
 */
class AMRPatchPositionSignatureCut
{
 public:
  AMRPatchPositionSignatureCut();
  ~AMRPatchPositionSignatureCut();

 public:

  /*!
  * \brief Method allowing searching for the best point to perform a cut.
  * \param sig The signature on which the search must be performed.
  * \return The best point for the cut (-1 if problem).
  */
  static CartCoord _cutDim(ConstArrayView<CartCoord> sig);

  /*!
  * \brief Method allowing a patch to be cut into two.
  * \param sig The patch to be cut.
  * \return The two patches resulting from the cut.
  */
  static std::pair<AMRPatchPositionSignature, AMRPatchPositionSignature> cut(const AMRPatchPositionSignature& sig);

  /*!
  * \brief Method allowing the patch or patches in the array \a sig_array_a to be cut.
  * \param sig_array_a [IN/OUT] The array of patches.
  */
  static void cut(UniqueArray<AMRPatchPositionSignature>& sig_array_a);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
