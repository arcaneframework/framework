// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CompactIncrementalItemConnectivity.h                        (C) 2000-2020 */
/*                                                                           */
/* Gestion des connectivités utilisant la méthode compacte.                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESH_COMPACTINCREMENTALITEMCONNECTIVITY_H
#define ARCANE_MESH_COMPACTINCREMENTALITEMCONNECTIVITY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"

#include "arcane/ConnectivityItemVector.h"
#include "arcane/MeshUtils.h"

#include "arcane/mesh/IncrementalItemConnectivity.h"
#include "arcane/mesh/MeshGlobal.h"
#include "arcane/mesh/ItemFamily.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ItemFamily;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Classe de base de gestion des connectivités utilisant la méthode compacte.
 *
 * La méthode compacte est la méthode historique Arcane qui regroupe
 * toutes les connectivités dans un seul bloc mémoire contigu. Cela permet
 * de réduire l'empreinte mémoire mais fige les connectivités possibles à la
 * compilation.
 *
 * \note Cette classe a besoin d'avoir la vision directe de ItemFamily.
 *
 * Cette classe est abstraite et il doit exister une implémentation spécifique
 * par couple (source_family,target_family).
 *
 * Pour l'instant, les implémentations suivantes sont disponibles:
 * - NodeFaceCompactIncrementalItemConnectivity.
 */
class ARCANE_MESH_EXPORT CompactIncrementalItemConnectivity
{
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/* TODO: Cette classe n'a rien de spécifique aux connectivités compactes
 * donc on pourrait la rendre accessible ailleurs.
 */
/*!
 * \brief Les classes suivantes permettant de gérer la connectivité historique
 * de ItemInternal pour une connectiivité donnée.
 * Ces classes doivent nécéssairement implémenter les méthodes suivantes:
 \begincode
 * static Integer connectivityIndex();
 * static Integer nbConnectedItem(ItemInternal* item);
 * static Int32 connectedItemLocalId(ItemInternal* item,Integer index);
 * static Int32ArrayView connectedItemsLocalId(ItemInternal* item);
 * static void replaceConnectedItem(ItemInternal* item,Integer index,Int32 target_lid);
 * static void updateSharedInfoRemove(ItemFamily::CompactConnectivityHelper& helper,
 *                                    ItemInternal* item,Integer nb_sub_item);
 * static void updateSharedInfoAdded(ItemFamily::CompactConnectivityHelper& helper,
 *                                   ItemInternal* item,Integer nb_sub_item);
 \endcode
 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Spécialisation de \a CompactItemConnectivityAccessorT pour accéder aux noeuds
class NodeCompactItemConnectivityAccessor
{
 public:
  static Integer connectivityIndex() { return ItemInternalConnectivityList::NODE_IDX; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Spécialisation de \a CompactItemConnectivityAccessorT pour accéder aux arêtes
class EdgeCompactItemConnectivityAccessor
{
 public:
  static Integer connectivityIndex() { return ItemInternalConnectivityList::EDGE_IDX; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Spécialisation de \a CompactItemConnectivityAccessorT pour accéder aux faces
class FaceCompactItemConnectivityAccessor
{
 public:
  static Integer connectivityIndex() { return ItemInternalConnectivityList::FACE_IDX; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Spécialisation de \a CompactItemConnectivityAccessorT pour accéder aux mailles
class CellCompactItemConnectivityAccessor
{
 public:
  static Integer connectivityIndex() { return ItemInternalConnectivityList::CELL_IDX; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Spécialisation de \a CompactItemConnectivityAccessorT pour accéder aux HParent
class HParentCompactItemConnectivityAccessor
{
 public:
  static Integer connectivityIndex() { return ItemInternalConnectivityList::HPARENT_IDX; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Spécialisation de \a CompactItemConnectivityAccessorT pour accéder aux HParent
class HChildCompactItemConnectivityAccessor
{
 public:
  static Integer connectivityIndex() { return ItemInternalConnectivityList::HCHILD_IDX; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe de base de la connectivité compacte dont la famille cible
 * utilise l'accesseur \a AccessorType
 */
template<typename AccessorType>
class CompactIncrementalItemConnectivityT
: public CompactIncrementalItemConnectivity
{
 public:
  static Integer connectivityIndex() { return AccessorType::connectivityIndex(); }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
