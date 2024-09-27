// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CellFamilySerializer.h                                      (C) 2000-2024 */
/*                                                                           */
/* Sérialisation/Désérialisation des familles de mailles.                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESH_CELLFAMILYSERIALIZER_H
#define ARCANE_MESH_CELLFAMILYSERIALIZER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"

#include "arcane/core/IItemFamilySerializer.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class DynamicMeshIncrementalBuilder;
class CellFamily;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Sérialisation/Désérialisation des familles de mailles.
 */
class ARCANE_MESH_EXPORT CellFamilySerializer
: public TraceAccessor
, public IItemFamilySerializer
{
 public:

  CellFamilySerializer(CellFamily* family, bool use_flags,
                       DynamicMeshIncrementalBuilder* mesh_builder);

 public:

  void serializeItems(ISerializer* buf, Int32ConstArrayView cells_local_id) override;
  void deserializeItems(ISerializer* buf, Int32Array* cells_local_id) override;
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

  DynamicMeshIncrementalBuilder* m_mesh_builder;
  CellFamily* m_family;
  bool m_use_flags;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

