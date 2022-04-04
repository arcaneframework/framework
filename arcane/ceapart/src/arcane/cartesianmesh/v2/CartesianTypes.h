// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CartesianTypes.h                                            (C) 2000-2021 */
/*                                                                           */
/* Types pour les maillage cartésiens.                                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CARTESIANMESH_CARTESIANTYPES_H
#define ARCANE_CARTESIANMESH_CARTESIANTYPES_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/cartesianmesh/CartesianMeshGlobal.h"

#include <array>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::CartesianMesh::V2
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief
 * Type pour les triplets cartésiens (i,j,k) et les triplets des dimensions (ni,nj,nk)
 */
using LocalIdType3 = Arcane::LocalIdType[3];

using UniqueIdType3 = Arcane::UniqueIdType[3];

using LocalIdType4 = Arcane::LocalIdType[4];

using IdxType = std::array<Arcane::Int64, 3>;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief
 * Type de cote (previous ou next) pour une direction donnee
 */
enum eMeshSide
{
  //! Côté précédent
  MS_previous = 0,
  //! Côté suivant
  MS_next = 1,
  //! Nb maximal de côtés valides
  MS_max = 2,
  //! Côté invalide ou non initialisé
  MS_invalid = (-1)
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::CartesianMesh::V2

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
