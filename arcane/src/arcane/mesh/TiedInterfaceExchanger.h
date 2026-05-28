// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* TiedInterfaceExchanger.h                                    (C) 2000-2016 */
/*                                                                           */
/* Exchanger between sub-domains of linked interfaces.                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESH_TIEDINTERFACEEXCHANGER_H
#define ARCANE_MESH_TIEDINTERFACEEXCHANGER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"

#include "arcane/mesh/MeshGlobal.h"

#include "arcane/core/ItemInternalVectorView.h"
#include "arcane/core/IItemFamilySerializeStep.h"

#include <map>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class ISubDomain;
class ItemFamilySerializeArgs;
} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class DynamicMesh;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Exchanger between sub-domains of linked interfaces.

 An instance of this class manages the exchange of information about
 linked interfaces (ITiedInterface) during an exchange of entities
 between sub-domains.

 This class is used via ItemsExchangeInfo2.
 
 An instance is only valid for one exchange and is then destroyed.

 The order of calls is as follows:
 1. computeExchangeInfos() which calculates the info to be serialized.
 2. serialize() in writing mode for each rank to be sent
 3. serialize() in reading mode for each rank received
 4. rebuildTiedInterfaces() to reconstruct the interfaces.
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
  typedef std::map<Int32, OneSubDomainInfo*> SubDomainInfoMap;
  SubDomainInfoMap m_infos;

 private:

  inline OneSubDomainInfo* _getInfo(Int32 rank);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
