// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IndirectItemFamilySerializer.h                              (C) 2000-2016 */
/*                                                                           */
/* Sérialisation/Désérialisation indirecte des familles d'entités.           */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESH_INDIRECTFAMILYSERIALIZER_H
#define ARCANE_MESH_INDIRECTFAMILYSERIALIZER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"

#include "arcane/IItemFamilySerializer.h"

#include "arcane/mesh/MeshGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_MESH_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Sérialisation/Désérialisation indirecte des familles d'entités.
 *
 * Une sérialisation/désérialisation est indirecte si elle est faite
 * indirectement par une autre famille. C'est le cas par exemple pour les
 * noeuds, les arêtes et les faces car leur sérialisation/désérialisation
 * se fait via la famille de maille associée.
 *
 * Le seul rôle de cette instance est alors de sérialiser les uniqueId()
 * des entités et associer lors de la désérialisation les localId() des
 * nouvelles entités ajoutées.
 *
 * Cela signifie aussi qu'il faut sérialiser/désérialiser les entités dont
 * dépend cette famille avant.
 */
class ARCANE_MESH_EXPORT IndirectItemFamilySerializer
: public TraceAccessor
, public IItemFamilySerializer
{
 public:
  IndirectItemFamilySerializer(IItemFamily* family);
 public:
  void serializeItems(ISerializer* buf,Int32ConstArrayView local_ids) override;
  void deserializeItems(ISerializer* buf,Int32Array* local_ids) override;
  void serializeItemRelations(ISerializer* buf,Int32ConstArrayView cells_local_id) override {ARCANE_UNUSED(buf);ARCANE_UNUSED(cells_local_id);}
  void deserializeItemRelations(ISerializer* buf,Int32Array* cells_local_id) override {ARCANE_UNUSED(buf);ARCANE_UNUSED(cells_local_id);}
  IItemFamily* family() const override;
 private:
  IItemFamily* m_family;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_MESH_END_NAMESPACE
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

