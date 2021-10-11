// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ParticleFamilyPolicyMng.cc                                  (C) 2000-2016 */
/*                                                                           */
/* Gestionnaire des politiques d'une famille de particules.                  */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/NotImplementedException.h"

#include "arcane/IItemFamilyCompactPolicy.h"
#include "arcane/ItemFamilyCompactInfos.h"
#include "arcane/IMeshCompacter.h"
#include "arcane/IMesh.h"

#include "arcane/mesh/ItemFamilyPolicyMng.h"
#include "arcane/mesh/ParticleFamilySerializer.h"
#include "arcane/mesh/ParticleFamily.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE
ARCANE_MESH_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Politique de compactage pour les particules.
 *
 * Les particules ne sont compactées que si on a des particules fantômes
 * possibles.
 * NOTE GG: je pense qu'on pourrait compacter même pour les autres
 * familles de particule pour lesquelles getEnableGhostItems() est faux.
 */
class ParticleFamilyCompactPolicy
: public TraceAccessor
, public IItemFamilyCompactPolicy
{
 public:
  ParticleFamilyCompactPolicy(ParticleFamily* family)
  : TraceAccessor(family->traceMng()), m_family(family)
  {
    m_cell_family = family->mesh()->cellFamily();
  }
 public:
  void beginCompact(ItemFamilyCompactInfos& compact_infos) override
  {
    if (_checkWantCompact(compact_infos))
      m_family->beginCompactItems(compact_infos);
  }
  void compactVariablesAndGroups(const ItemFamilyCompactInfos& compact_infos) override
  {
    if (_checkWantCompact(compact_infos))
      m_family->compactVariablesAndGroups(compact_infos);
  }
  void updateInternalReferences(IMeshCompacter* compacter) override
  {
  }
  void endCompact(ItemFamilyCompactInfos& compact_infos) override
  {
    if (_checkWantCompact(compact_infos))
      m_family->finishCompactItems(compact_infos);
  }
  void finalizeCompact(IMeshCompacter* compacter) override
  {
    ARCANE_UNUSED(compacter);
  }
  void compactConnectivityData() override
  {
    // NOTE GG: pour être conforme au code existant on appelle compactReference()
    // que si la famille possède la notion de fantôme
    // mais je pense qu'il faudrait le faire tout le temps.
    if (m_family->getEnableGhostItems())
      m_family->compactReferences();
  }
  IItemFamily* family() const override { return m_family; }
 private:
  bool _checkWantCompact(const ItemFamilyCompactInfos& compact_infos)
  {
    // Pour des raisons de compatibilité avec l'existant (version 2.4.1
    // et antérieure on ne compacte pas la famille de particule si elle
    // n'a pas de fantômes, sauf si c'est la seule famille en cours de compactage
    // (ce qui correspond à un appel direct à ItemFamily::compactItems()).
    // TODO: une fois qu'on sera sur que cela est OK il faudra toujours compacter
    // quelle que soit la famille.
    if (m_family->getEnableGhostItems())
      return true;
    ItemFamilyCollection families = compact_infos.compacter()->families();
    if (families.count()==1 && families.front()==m_family)
      return true;
    return false;
  }
 private:
  ParticleFamily* m_family;
  IItemFamily* m_cell_family;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Gestionnaire des politiques d'une famille de particules.
 */
class ARCANE_MESH_EXPORT ParticleFamilyPolicyMng
: public ItemFamilyPolicyMng
{
 public:
  ParticleFamilyPolicyMng(ParticleFamily* family)
  : ItemFamilyPolicyMng(family,new ParticleFamilyCompactPolicy(family))
  , m_family(family){}
 public:
  IItemFamilySerializer* createSerializer(bool use_flags) override
  {
    if (use_flags)
      throw NotSupportedException(A_FUNCINFO,"serialisation with 'use_flags==true'");
    return new ParticleFamilySerializer(m_family);
  }
 private:
  ParticleFamily* m_family;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_MESH_EXPORT IItemFamilyPolicyMng*
createParticleFamilyPolicyMng(ItemFamily* family)
{
  ParticleFamily* f = ARCANE_CHECK_POINTER(dynamic_cast<ParticleFamily*>(family));
  return new ParticleFamilyPolicyMng(f);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_MESH_END_NAMESPACE
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
