// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ParticleFamily.h                                            (C) 2000-2021 */
/*                                                                           */
/* Famille de particules.                                                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESH_PARTICLEFAMILY_H
#define ARCANE_MESH_PARTICLEFAMILY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/MultiBuffer.h"

#include "arcane/VariableTypes.h"
#include "arcane/IParticleFamily.h"

#include "arcane/mesh/ItemFamily.h"
#include "arcane/mesh/ItemInternalConnectivityIndex.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{
class IncrementalItemConnectivity;
class OneItemIncrementalItemConnectivity;
class ItemSharedInfoWithType;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Famille de particules.
 */
class ARCANE_MESH_EXPORT ParticleFamily
: public ItemFamily
, public IParticleFamily
{
 private:

  typedef ItemConnectivitySelectorT<CellInternalConnectivityIndex,OneItemIncrementalItemConnectivity> CellConnectivity;
  typedef ItemFamily BaseClass;

 public:

  static String const defaultFamilyName() {
    return "Particle" ;
  }

  ParticleFamily(IMesh* mesh,const String& name);
  virtual ~ParticleFamily(); //<! Libère les ressources

 public:

  virtual void build() override;

  void setEnableGhostItems(bool value) override {
    m_enable_ghost_items = value ;
  }
  bool getEnableGhostItems() const override {
    return m_enable_ghost_items ;
  }

  //! Nom de la famille
  String name() const override { return BaseClass::name(); }
  String fullName() const override { return BaseClass::fullName(); }
  Integer nbItem() const override { return BaseClass::nbItem(); }
  ItemGroup allItems() const override { return BaseClass::allItems(); }

  ParticleVectorView addParticles(Int64ConstArrayView unique_ids,
                                  Int32ArrayView items) override;
  ParticleVectorView addParticles2(Int64ConstArrayView unique_ids,
                                   Int32ConstArrayView owners,
                                   Int32ArrayView items) override;

  ParticleVectorView addParticles(Int64ConstArrayView unique_ids,
                                  Int32ConstArrayView cells_local_id,
                                  Int32ArrayView items_local_id) override;
  void removeParticles(Int32ConstArrayView items_local_id) override;

  void addItems(Int64ConstArrayView unique_ids,Int32ConstArrayView owners,Int32ArrayView items);

  void internalRemoveItems(Int32ConstArrayView local_ids,bool keep_ghost) override;
  void exchangeParticles() override;

  void setParticleCell(Particle particle,Cell new_cell) override;
  void setParticlesCell(ParticleVectorView particles,CellVectorView new_cells) override;

  void endUpdate() override { ItemFamily::endUpdate(); }

 public:
  
  void preAllocate(Integer nb_item);

 public:
  
  void prepareForDump() override;
  void readFromDump() override;

 public:

  void setHasUniqueIdMap(bool v) override;
  bool hasUniqueIdMap() const override;

 public:

  void computeSynchronizeInfos() override
  {
    if(m_enable_ghost_items)
      ItemFamily::computeSynchronizeInfos() ;
  }
  IItemFamily* itemFamily() override { return this; }
  IParticleFamily* toParticleFamily() override { return this; }

  void checkValidConnectivity() override;
  void removeNeedRemoveMarkedItems() override;

 private:
  
  ItemTypeInfo* m_particle_type_info;
  ItemSharedInfoWithType* m_particle_shared_info;
  Int32 m_sub_domain_id;
  bool m_enable_ghost_items;
  CellConnectivity* m_cell_connectivity;

  inline ItemInternal* _allocParticle(Int64 uid,bool& need_alloc);
  inline ItemInternal* _findOrAllocParticle(Int64 uid,bool& is_alloc);

  void _printInfos(Integer nb_added);
  void _setSharedInfo();
  inline void _setCell(ItemLocalId particle,ItemLocalId cell);
  inline void _initializeNewlyAllocatedParticle(ItemInternal* particle,Int64 uid);
  void _addItems(Int64ConstArrayView unique_ids,Int32ArrayView items);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
