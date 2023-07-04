// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshMaterialExchangeMng.cc                                  (C) 2000-2023 */
/*                                                                           */
/* Gestion de l'échange des matériaux entre sous-domaines.                   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/FunctorUtils.h"

#include "arcane/IItemFamilySerializeStep.h"
#include "arcane/IMesh.h"
#include "arcane/IItemFamily.h"
#include "arcane/IItemFamilyPolicyMng.h"
#include "arcane/ItemFamilySerializeArgs.h"
#include "arcane/ISerializer.h"

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
  ExchangeCellStep(MeshMaterialExchangeMng* exchange_mng,IItemFamily* family)
  : TraceAccessor(family->traceMng()), m_exchange_mng(exchange_mng),
    m_material_mng(exchange_mng->m_material_mng), m_family(family),
    m_indirect_modifier(nullptr)
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
    // Création du modificateur indirect permettant de remettre à jour les
    // matériaux après la mise à jour des groupes suite à la suppression
    // des entités lors de l'échange.
    // TODO: vérifier si l'utilisation des uniqueId() est nécessaire.
    m_indirect_modifier = new MeshMaterialIndirectModifier(m_material_mng);
    m_indirect_modifier->beginUpdate();
  }
  void notifyAction(const NotifyActionArgs& args) override
  {
    if (args.action()==eAction::AC_BeginReceive){
      // Avant de désérialiser, met à jour les matériaux car les groupes associés
      // aux matériaux et milieux ont changé lors de la phase d'échange:
      // certaines mailles ont été supprimées et d'autres ajoutées.
      // Normalement après cette phase les matériaux et milieux sont
      // corrects et les valeurs des variables sont OK pour les mailles
      // qui étaient présentes dans ce sous-domaine avant l'échange.
      // Il reste à mettre à jour les valeurs des variables pour
      // les mailles qui viennent d'être ajoutées. Cela se fait dans
      // la désérialisation.
      info() << "NOTIFY_ACTION BEGIN_RECEIVE";
      m_indirect_modifier->endUpdate();
      delete m_indirect_modifier;
      m_indirect_modifier = nullptr;
    }
    if (args.action()==eAction::AC_EndReceive){
      info() << "NOTIFY_ACTION END_RECEIVE";
      // Maintenant que les valeurs sont bonnes pour les variables, on doit
      // les sauvegarder car une fois les réceptions terminées il va y avoir
      // un compactage et pour l'instant cela peut poser des problèmes
      // car la mise à jour des groupes via les observers n'est pas traitée.
      // Du coup on remettra tout à jour lors du finalize();
      m_indirect_modifier = new MeshMaterialIndirectModifier(m_material_mng);
      m_indirect_modifier->beginUpdate();
    }
  }
  void serialize(const ItemFamilySerializeArgs& args) override
  {
    info() << "SERIALIZE_CELLS_MATERIAL rank=" << args.rank()
           << " n=" << args.localIds().size();
    ISerializer* sbuf = args.serializer();

    // Sérialise chaque variable
    auto serialize_variables_func = [&](IMeshMaterialVariable* mv){
      info() << "SERIALIZE_MESH_MATERIAL_VARIABLE name=" << mv->name();
      mv->serialize(sbuf,args.localIds());
    };
    functor::apply(m_material_mng,&MeshMaterialMng::visitVariables,serialize_variables_func);
  }
  void finalize() override
  {
    // Reconstruit toutes les informations sur les groupes avec les bonnes valeurs
    // des variables.
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
  : m_exchange_mng(exchange_mng){}
 public:
  IItemFamilySerializeStep* createStep(IItemFamily* family) override
  {
    // Ne construit une instance que si on souhaite conserver les valeurs
    // des milieux et matériaux, sinon il n'y a rien à faire. Le code
    // utilisateur devra alors appeler IMeshMaterialMng::forceRecompute()
    // pour mettre à jour les informations des matériaux.
    if (m_exchange_mng->materialMng()->isKeepValuesAfterChange())
      return new ExchangeCellStep(m_exchange_mng,family);
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
  if (m_serialize_cells_factory){
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
 * \brief Enregistre la fabrique pour les échanges.
 *
 * Cette méthode ne doit être appelé qu'une fois le IMeshMaterialMng
 * initialisé.
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

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
