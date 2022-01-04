// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* TiedInterfaceExchanger.h                                    (C) 2000-2016 */
/*                                                                           */
/* Echangeur entre sous-domaines des interfaces liées.                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESH_TIEDINTERFACEEXCHANGER_H
#define ARCANE_MESH_TIEDINTERFACEEXCHANGER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"

#include "arcane/mesh/MeshGlobal.h"

#include "arcane/ItemInternalVectorView.h"
#include "arcane/IItemFamilySerializeStep.h"

#include <map>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ISubDomain;
class ItemFamilySerializeArgs;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_MESH_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class DynamicMesh;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Echangeur entre sous-domaines les interfaces liées.

 Une instance de cette classe gère l'échange des informations sur
 les interfaces liées (ITiedInterface) lors d'un échange d'entités
 entre sous-domaines.

 Cette classe est utilisée via ItemsExchangeInfo2.
 
 Une instance n'est valide que pour un échange et est ensuite détruite.

 L'ordre d'appel est le suivant:
 1. computeExchangeInfos() qui calcule les infos à sérialiser.
 2. serialize() en écriture pour chaque rang à envoyer
 3. serialize() en lecture pour chaque rang reçu
 4. rebuildTiedInterfaces() pour reconstruire les interfaces.
*/
class TiedInterfaceExchanger
: public TraceAccessor
, public IItemFamilySerializeStep
{
  class OneSubDomainInfo;
  class DeserializedInfo;

 public:

  TiedInterfaceExchanger(DynamicMesh* mesh);
  ~TiedInterfaceExchanger();

 public:

  void initialize() override;
  void notifyAction(const NotifyActionArgs&) override {}
  void serialize(const ItemFamilySerializeArgs& args) override;
  void finalize() override;
  IItemFamily* family() const override;
  ePhase phase() const override { return IItemFamilySerializeStep::PH_Item; }

 private:

  DynamicMesh* m_mesh;
  ISubDomain* m_sub_domain;
  DeserializedInfo* m_deserialized_info;
  Int32 m_my_rank;
  typedef std::map<Int32,OneSubDomainInfo*> SubDomainInfoMap;
  SubDomainInfoMap m_infos;

 private:

  inline OneSubDomainInfo* _getInfo(Int32 rank);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_MESH_END_NAMESPACE
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

