// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemFamilyPolicyMng.h                                       (C) 2000-2017 */
/*                                                                           */
/* Manager for the policies of a family of nodes.                            */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESH_ITEMFAMILYPOLICYMNG_H
#define ARCANE_MESH_ITEMFAMILYPOLICYMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Array.h"

#include "arcane/core/IItemFamilyPolicyMng.h"

#include "arcane/mesh/MeshGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class IItemFamilyCompactPolicy;
} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ItemFamily;
class ItemsExchangeInfo2;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Manager for the policies of a family of entities.
 */
class ARCANE_MESH_EXPORT ItemFamilyPolicyMng
: public IItemFamilyPolicyMng
{
 public:

  explicit ItemFamilyPolicyMng(ItemFamily* family,
                               IItemFamilyCompactPolicy* compact_policy = nullptr)
  : m_item_family(family)
  , m_compact_policy(compact_policy)
  {}
  ~ItemFamilyPolicyMng() override;

 public:

  IItemFamilyCompactPolicy* compactPolicy() override
  {
    return m_compact_policy;
  }
  IItemFamilyExchanger* createExchanger() override;
  IItemFamilySerializer* createSerializer(bool with_flags) override;
  void addSerializeStep(IItemFamilySerializeStepFactory* factory) override;
  void removeSerializeStep(IItemFamilySerializeStepFactory* factory) override;

 protected:

  virtual ItemsExchangeInfo2* _createExchanger();

 private:

  ItemFamily* m_item_family;
  IItemFamilyCompactPolicy* m_compact_policy;
  UniqueArray<IItemFamilySerializeStepFactory*> m_serialize_step_factories;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
