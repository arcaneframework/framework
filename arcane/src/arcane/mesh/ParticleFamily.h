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
  virtual const String& name() const override { return BaseClass::name(); }
  virtual const String& fullName() const override { return BaseClass::fullName(); }
  virtual Integer nbItem() const override { return BaseClass::nbItem(); }
  virtual ItemGroup allItems() const override { return BaseClass::allItems(); }

  virtual ParticleVectorView addParticles(Int64ConstArrayView unique_ids,
                                          Int32ArrayView items) override;
  virtual ParticleVectorView addParticles2(Int64ConstArrayView unique_ids,
                                          Int32ConstArrayView owners,
                                          Int32ArrayView items) override;

  virtual ParticleVectorView addParticles(Int64ConstArrayView unique_ids,
                                          Int32ConstArrayView cells_local_id,
                                          Int32ArrayView items_local_id) override;
  virtual void removeParticles(Int32ConstArrayView items_local_id) override;

  virtual void addItems(Int64ConstArrayView unique_ids,Int32ArrayView items) override;
  virtual void addItems(Int64ConstArrayView unique_ids,Int32ConstArrayView owners,Int32ArrayView items);

  virtual void addItems(Int64ConstArrayView unique_ids,ArrayView<Item> items) override;
  virtual void addItems(Int64ConstArrayView unique_ids,ItemGroup item_group) override;
  virtual void internalRemoveItems(Int32ConstArrayView local_ids,bool keep_ghost) override;
  virtual void exchangeParticles() override;
  virtual void exchangeItems() override;


  virtual void setParticleCell(Particle particle,Cell new_cell) override;
  virtual void setParticlesCell(ParticleVectorView particles,CellVectorView new_cells) override;

  virtual void endUpdate() override { ItemFamily::endUpdate(); }

 public:
  
  void preAllocate(Integer nb_item);

 public:
  
  virtual void prepareForDump() override;
  virtual void readFromDump() override;

 public:

  virtual void setHasUniqueIdMap(bool v) override;
  virtual bool hasUniqueIdMap() const override;

 public:

  virtual void computeSynchronizeInfos() override
  {
    if(m_enable_ghost_items)
      ItemFamily::computeSynchronizeInfos() ;
  }
  virtual IItemFamily* itemFamily() override { return this; }
  virtual IParticleFamily* toParticleFamily() override { return this; }

  virtual void checkValidConnectivity() override;

  virtual void removeNeedRemoveMarkedItems() override;

 protected:

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
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
