// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ParticleFamilySerializer.h                                  (C) 2000-2017 */
/*                                                                           */
/* Sérialisation/Désérialisation des familles de particules.                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESH_PARTICLEFAMILYSERIALIZER_H
#define ARCANE_MESH_PARTICLEFAMILYSERIALIZER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"

#include "arcane/mesh/MeshGlobal.h"
#include "arcane/IItemFamilySerializer.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_MESH_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ParticleFamily;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Sérialisation/Désérialisation des familles de liens.
 */
class ARCANE_MESH_EXPORT ParticleFamilySerializer
: public TraceAccessor
, public IItemFamilySerializer
{
 public:
  ParticleFamilySerializer(ParticleFamily* family);
 public:
  void serializeItems(ISerializer* buf,Int32ConstArrayView local_ids) override;
  void deserializeItems(ISerializer* buf,Int32Array* local_ids) override;
  void serializeItemRelations(ISerializer* buf,Int32ConstArrayView cells_local_id) override {ARCANE_UNUSED(buf);ARCANE_UNUSED(cells_local_id);}
  void deserializeItemRelations(ISerializer* buf,Int32Array* cells_local_id) override {ARCANE_UNUSED(buf);ARCANE_UNUSED(cells_local_id);}
  IItemFamily* family() const override;
 private:
  ParticleFamily* m_family;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_MESH_END_NAMESPACE
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

