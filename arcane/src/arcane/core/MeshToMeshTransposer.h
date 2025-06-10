// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshToMeshTransposer.h                                      (C) 2000-2025 */
/*                                                                           */
/* Opérateur de transposition entre sous-maillages.                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_MESGTOMESHTRANSPOSE_H
#define ARCANE_CORE_MESGTOMESHTRANSPOSE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ItemVectorView.h"
#include "arcane/ItemVector.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Transposeur d'items entre sous-maillages
class ARCANE_CORE_EXPORT MeshToMeshTransposer
{
 public:

  //! Transpose \a itemsA from \a meshB to \a items on \a meshB
  static ItemVector transpose(IMesh* meshA, IMesh* meshB,
                              ItemVectorView itemsA,
                              bool do_fatal = false);

  //! Transpose \a itemsA from \a familyA to \a items on \a familyB
  static ItemVector transpose(IItemFamily* familyA, IItemFamily* familyB,
                              ItemVectorView itemsA, bool do_fatal = false);

  //! Transpose le genre \a kindA du maillage \a meshA en le genre associé dans \a meshB
  static eItemKind kindTranspose(eItemKind kindA, IMesh* meshA, IMesh* meshB);

 private:

  static ItemVector _transpose(IItemFamily* familyA, IItemFamily* familyB,
                               const ItemVectorView& itemsA, bool do_fatal);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
