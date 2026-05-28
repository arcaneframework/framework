// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ParticleFamilyPolicyMng.cc                                  (C) 2000-2016 */
/*                                                                           */
/* Manager for the policies of a particle family.                            */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/NotImplementedException.h"

#include "arcane/core/IItemFamilyCompactPolicy.h"
#include "arcane/core/ItemFamilyCompactInfos.h"
#include "arcane/core/IMeshCompacter.h"
#include "arcane/core/IMesh.h"

#include "arcane/mesh/ItemFamilyPolicyMng.h"
#include "arcane/mesh/ParticleFamilySerializer.h"
#include "arcane/mesh/ParticleFamily.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Compaction policy for particles.
 *
 * Particles are only compacted if ghost particles are possible.
 * NOTE GG: I think we could compact even for other particle families for which getEnableGhostItems() is false.
 */
class ParticleFamilyCompactPolicy
: public TraceAccessor
, public IItemFamilyCompactPolicy
{
 public:

  ParticleFamilyCompactPolicy(ParticleFamily* family)
  : TraceAccessor(family->traceMng())
  , m_family(family)
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
  void updateInternalReferences(IMeshCompacter*) override
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
  }
  IItemFamily* family() const override { return m_family; }

 private:

  bool _checkWantCompact(const ItemFamilyCompactInfos& compact_infos)
  {
    // For compatibility reasons with the existing code (version 2.4.1
    // and earlier, we do not compact the particle family if it does not have ghosts,
    // unless it is the only family being compacted
    // (which corresponds to a direct call to ItemFamily::compactItems()).
    // TODO: Once we are sure this is OK, we must always compact
    // regardless of the family.
    if (m_family->getEnableGhostItems())
      return true;
    ItemFamilyCollection families = compact_infos.compacter()->families();
    if (families.count() == 1 && families.front() == m_family)
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
 * \brief Manager for the policies of a particle family.
 */
class ARCANE_MESH_EXPORT ParticleFamilyPolicyMng
: public ItemFamilyPolicyMng
{
 public:

  ParticleFamilyPolicyMng(ParticleFamily* family)
  : ItemFamilyPolicyMng(family, new ParticleFamilyCompactPolicy(family))
  , m_family(family)
  {}

 public:

  IItemFamilySerializer* createSerializer(bool use_flags) override
  {
    if (use_flags)
      throw NotSupportedException(A_FUNCINFO, "serialisation with 'use_flags==true'");
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

} // namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
