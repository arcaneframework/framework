// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IndirectItemFamilySerializer.h                              (C) 2000-2016 */
/*                                                                           */
/* Indirect serialization/deserialization of entity families.                */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESH_INDIRECTFAMILYSERIALIZER_H
#define ARCANE_MESH_INDIRECTFAMILYSERIALIZER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"

#include "arcane/core/IItemFamilySerializer.h"

#include "arcane/mesh/MeshGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Indirect serialization/deserialization of entity families.
 *
 * Serialization/deserialization is indirect if it is done
 * indirectly by another family. This is the case, for example, for nodes, edges, and faces because their serialization/deserialization
 * is done via the associated mesh family.
 *
 * The only role of this instance is then to serialize the uniqueId()
 * of the entities and associate the localId() of the
 * new added entities during deserialization.
 *
 * This also means that the entities that
 * this family depends on must be serialized/deserialized first.
 */
class ARCANE_MESH_EXPORT IndirectItemFamilySerializer
: public TraceAccessor
, public IItemFamilySerializer
{
 public:

  IndirectItemFamilySerializer(IItemFamily* family);

 public:

  void serializeItems(ISerializer* buf, Int32ConstArrayView local_ids) override;
  void deserializeItems(ISerializer* buf, Int32Array* local_ids) override;
  void serializeItemRelations(ISerializer* buf, Int32ConstArrayView cells_local_id) override
  {
    ARCANE_UNUSED(buf);
    ARCANE_UNUSED(cells_local_id);
  }
  void deserializeItemRelations(ISerializer* buf, Int32Array* cells_local_id) override
  {
    ARCANE_UNUSED(buf);
    ARCANE_UNUSED(cells_local_id);
  }
  IItemFamily* family() const override;

 private:

  IItemFamily* m_family;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
