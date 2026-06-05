// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshMaterialExchangeMng.cc                                  (C) 2000-2023 */
/*                                                                           */
/* Management of material exchange between sub-domains.                      */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/FunctorUtils.h"

#include "arcane/core/IItemFamilySerializeStep.h"
#include "arcane/core/IMesh.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/IItemFamilyPolicyMng.h"
#include "arcane/core/ItemFamilySerializeArgs.h"
#include "arcane/core/ISerializer.h"

#include "arcane/materials/MeshMaterialExchangeMng.h"
#include "arcane/materials/MeshMaterialIndirectModifier.h"
#include "arcane/materials/IMeshMaterialVariable.h"

#include "arcane/materials/internal/MeshMaterialMng.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class MeshMaterialExchangeMng::ExchangeCellStep
: public TraceAccessor
, public IItemFamilySerializeStep
{
 public:

  ExchangeCellStep(MeshMaterialExchangeMng* exchange_mng, IItemFamily* family)
  : TraceAccessor(family->traceMng())
  , m_exchange_mng(exchange_mng)
  , m_material_mng(exchange_mng->m_material_mng)
  , m_family(family)
  , m_indirect_modifier(nullptr)
  {
  }
  ~ExchangeCellStep()
  {
    info() << "DESTROY SERIALIZE_CELLS_MATERIAL";
  }

 public:

  void initialize() override
  {
    if (m_exchange_mng->m_is_in_mesh_material_exchange)
      ARCANE_FATAL("Already in an exchange");
    m_exchange_mng->m_is_in_mesh_material_exchange = false;
    // Creation of the indirect modifier allowing materials to be updated
    // after group updates following entity deletion during exchange.
    // TODO: check if using uniqueId() is necessary.
    m_indirect_modifier = new MeshMaterialIndirectModifier(m_material_mng);
    m_indirect_modifier->beginUpdate();
  }
  void notifyAction(const NotifyActionArgs& args) override
  {
    if (args.action() == eAction::AC_BeginReceive) {
      // Before deserializing, update the materials because the groups associated
      // with materials and media have changed during the exchange phase:
      // some meshes have been deleted and others added.
      // Normally after this phase, the materials and media are
      // correct and the variable values are OK for the meshes
      // that were present in this sub-domain before the exchange.
      // We still need to update the variable values for
      // the meshes that have just been added. This is done during
      // deserialization.
      info() << "NOTIFY_ACTION BEGIN_RECEIVE";
      m_indirect_modifier->endUpdate();
      delete m_indirect_modifier;
      m_indirect_modifier = nullptr;
    }
    if (args.action() == eAction::AC_EndReceive) {
      info() << "NOTIFY_ACTION END_RECEIVE";
      // Now that the values are good for the variables, we must
      // save them because once the receptions are finished, there will be
      // a compaction and for now this can cause problems
      // because the group update via observers is not processed.
      // So we will update everything during finalize();
      m_indirect_modifier = new MeshMaterialIndirectModifier(m_material_mng);
      m_indirect_modifier->beginUpdate();
    }
  }
  void serialize(const ItemFamilySerializeArgs& args) override
  {
    info() << "SERIALIZE_CELLS_MATERIAL rank=" << args.rank()
           << " n=" << args.localIds().size();
    ISerializer* sbuf = args.serializer();

    // Serialize each variable
    auto serialize_variables_func = [&](IMeshMaterialVariable* mv) {
      info() << "SERIALIZE_MESH_MATERIAL_VARIABLE name=" << mv->name();
      mv->serialize(sbuf, args.localIds());
    };
    functor::apply(m_material_mng, &MeshMaterialMng::visitVariables, serialize_variables_func);
  }
  void finalize() override
  {
    // Reconstruct all information about groups with the correct variable values.
    m_indirect_modifier->endUpdate();
    delete m_indirect_modifier;
    m_indirect_modifier = nullptr;
    m_exchange_mng->m_is_in_mesh_material_exchange = false;
  }
  ePhase phase() const override { return IItemFamilySerializeStep::PH_Variable; }
  IItemFamily* family() const override { return m_family; }

 public:

  MeshMaterialExchangeMng* m_exchange_mng;
  MeshMaterialMng* m_material_mng;
  IItemFamily* m_family;
  MeshMaterialIndirectModifier* m_indirect_modifier;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class MeshMaterialExchangeMng::ExchangeCellFactory
: public IItemFamilySerializeStepFactory
{
 public:

  ExchangeCellFactory(MeshMaterialExchangeMng* exchange_mng)
  : m_exchange_mng(exchange_mng)
  {}

 public:

  IItemFamilySerializeStep* createStep(IItemFamily* family) override
  {
    // Only constructs an instance if we want to keep the values
    // of the media and materials; otherwise, there is nothing to do. The
    // user code must then call IMeshMaterialMng::forceRecompute()
    // to update the material information.
    if (m_exchange_mng->materialMng()->isKeepValuesAfterChange())
      return new ExchangeCellStep(m_exchange_mng, family);
    return nullptr;
  }

 public:

  MeshMaterialExchangeMng* m_exchange_mng;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshMaterialExchangeMng::
MeshMaterialExchangeMng(MeshMaterialMng* material_mng)
: TraceAccessor(material_mng->traceMng())
, m_material_mng(material_mng)
, m_serialize_cells_factory(nullptr)
, m_is_in_mesh_material_exchange(false)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshMaterialExchangeMng::
~MeshMaterialExchangeMng()
{
  if (m_serialize_cells_factory) {
    IItemFamily* cell_family = m_material_mng->mesh()->cellFamily();
    cell_family->policyMng()->removeSerializeStep(m_serialize_cells_factory);
    delete m_serialize_cells_factory;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialExchangeMng::
build()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Registers the factory for exchanges.
 *
 * This method must only be called once IMeshMaterialMng
 * is initialized.
 */
void MeshMaterialExchangeMng::
registerFactory()
{
  if (m_serialize_cells_factory)
    ARCANE_FATAL("factory already registered");
  m_serialize_cells_factory = new ExchangeCellFactory(this);
  IItemFamily* cell_family = m_material_mng->mesh()->cellFamily();
  cell_family->policyMng()->addSerializeStep(m_serialize_cells_factory);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IMeshMaterialMng* MeshMaterialExchangeMng::
materialMng() const
{
  return m_material_mng;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
