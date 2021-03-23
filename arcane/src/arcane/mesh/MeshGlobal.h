// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshGlobal.h                                                (C) 2000-2017 */
/*                                                                           */
/* Déclarations générales de la composante Maillage de Arcane.               */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESH_MESHGLOBAL_H
#define ARCANE_MESH_MESHGLOBAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#define ARCANE_MESH_BEGIN_NAMESPACE  namespace mesh {
#define ARCANE_MESH_END_NAMESPACE    }

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE
ARCANE_MESH_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ItemFamily;
class NodeFamily;
class EdgeFamily;
class FaceFamily;
class CellFamily;

class IncrementalItemConnectivity;
template<typename ItemType>
class CompactIncrementalItemConnectivityT;

class ItemConnectivitySelector;
template<typename LeagcyType,typename CustomType>
class ItemConnectivitySelectorT;

class NodeCompactItemConnectivityAccessor;
class EdgeCompactItemConnectivityAccessor;
class FaceCompactItemConnectivityAccessor;
class CellCompactItemConnectivityAccessor;
class HParentCompactItemConnectivityAccessor;
class HChildCompactItemConnectivityAccessor;

typedef CompactIncrementalItemConnectivityT<NodeCompactItemConnectivityAccessor>
NodeCompactIncrementalItemConnectivity;

typedef CompactIncrementalItemConnectivityT<EdgeCompactItemConnectivityAccessor>
EdgeCompactIncrementalItemConnectivity;

typedef CompactIncrementalItemConnectivityT<FaceCompactItemConnectivityAccessor>
FaceCompactIncrementalItemConnectivity;

typedef CompactIncrementalItemConnectivityT<CellCompactItemConnectivityAccessor>
CellCompactIncrementalItemConnectivity;

typedef CompactIncrementalItemConnectivityT<HParentCompactItemConnectivityAccessor>
HParentCompactIncrementalItemConnectivity;

typedef CompactIncrementalItemConnectivityT<HChildCompactItemConnectivityAccessor>
HChildCompactIncrementalItemConnectivity;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_MESH_END_NAMESPACE
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
