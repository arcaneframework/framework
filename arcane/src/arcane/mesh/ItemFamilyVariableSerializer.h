// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemFamilyVariableSerializer.h                              (C) 2000-2016 */
/*                                                                           */
/* Manages the serialization/deserialization of variables within a family.   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESH_ITEMFAMILYVARIABLESERIALIZER_H
#define ARCANE_MESH_ITEMFAMILYVARIABLESERIALIZER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"

#include "arcane/core/IItemFamilySerializeStep.h"

#include "arcane/mesh/MeshGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class IVariable;
class ItemFamilySerializeArgs;
} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Manages the serialization/deserialization of variables within a family.
 */
class ARCANE_MESH_EXPORT ItemFamilyVariableSerializer
: public TraceAccessor
, public IItemFamilySerializeStep
{
 public:

  ItemFamilyVariableSerializer(IItemFamily* family);
  ~ItemFamilyVariableSerializer();

 public:

  void initialize() override;
  void notifyAction(const NotifyActionArgs&) override {}
  void serialize(const ItemFamilySerializeArgs& args) override;
  void finalize() override {}
  ePhase phase() const override { return IItemFamilySerializeStep::PH_Variable; }
  IItemFamily* family() const override { return m_item_family; }

 protected:

  IItemFamily* _family() const { return m_item_family; }

 private:

  IItemFamily* m_item_family;
  /*!
   * \brief List of variables to exchange.
    
   IMPORTANT: This list must be identical for all sub-domains
   otherwise the deserializations will give incorrect results.
  */
  UniqueArray<IVariable*> m_variables_to_exchange;

 private:

  void _serializePartialVariable(IVariable* var, ISerializer* sbuf, Int32ConstArrayView local_ids);
  void _checkSerialization(ISerializer* sbuf, Int32ConstArrayView local_ids);
  void _checkSerializationVariable(ISerializer* sbuf, IVariable* var);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
