// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CartesianMeshGlobal.h                                       (C) 2000-2025 */
/*                                                                           */
/* Déclarations de la composante 'arcane_cartesianmesh'.                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CARTESIANMESH_CARTESIANMESHGLOBAL_H
#define ARCANE_CARTESIANMESH_CARTESIANMESHGLOBAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"
#include "arcane/utils/UtilsTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifdef ARCANE_COMPONENT_arcane_cartesianmesh
#define ARCANE_CARTESIANMESH_EXPORT ARCANE_EXPORT
#else
#define ARCANE_CARTESIANMESH_EXPORT ARCANE_IMPORT
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class CartesianMeshImpl;
class CellDirectionMng;
class NodeDirectionMng;
class FaceDirectionMng;
class ICartesianMesh;
class ICartesianMeshPatch;
class CartesianMeshPatch;
class CartesianConnectivity;
class CartesianMeshCoarsening;
class CartesianMeshCoarsening2;
class CartesianMeshRenumberingInfo;
class ICartesianMeshInternal;
class CartesianMeshPatchListView;
class CartesianPatch;
class CartesianMeshAMRMng;
class AMRZonePosition;
class AMRPatchPosition;
class AMRPatchPositionLevelGroup;
class AMRPatchPositionSignature;
class AMRPatchPositionSignatureCut;
class CartesianPatchGroup;
class ICartesianMeshAMRPatchMng;
class ICartesianMeshNumberingMngInternal;
class ICartesianMeshPatchInternal;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Position des noeuds de la maille par direction pour les maillages
 * cartésiens.
 */
enum eCellNodePosition
{
  CNP_NextLeft = 0,
  CNP_NextRight = 1,
  CNP_PreviousRight = 2,
  CNP_PreviousLeft = 3,

  CNP_TopNextLeft = 4,
  CNP_TopNextRight = 5,
  CNP_TopPreviousRight = 6,
  CNP_TopPreviousLeft = 7
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

typedef Int32 CartCoordType;
typedef Int32x3 CartCoord3Type;
typedef Int32x2 CartCoord2Type;

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
