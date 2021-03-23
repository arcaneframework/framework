// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ExtraGhostItemsManager.h                                    (C) 2000-2015 */
/*                                                                           */
/* Construction des items fantômes supplémentaires.                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_EXTRAGHOSTITEMSMANAGER_H_ 
#define ARCANE_EXTRAGHOSTITEMSMANAGER_H_ 
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"
#include "arcane/mesh/MeshGlobal.h"
#include "arcane/ISubDomain.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/ISerializer.h"
#include "arcane/utils/Array.h"
#include "arcane/IItemFamily.h"

#include "arcane/IExtraGhostItemsBuilder.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_MESH_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IExtraGhostItemsAdder
{
public:

  typedef Arcane::Int32ArrayView SubDomainItems;
  IExtraGhostItemsAdder(){}
  virtual ~IExtraGhostItemsAdder(){}

  virtual void serializeGhostItems(ISerializer* buffer,Int32ConstArrayView ghost_item_lids) = 0;
  virtual void addExtraGhostItems (ISerializer* buffer) = 0;
  virtual void updateSynchronizationInfo() = 0;
  virtual ISubDomain* subDomain() = 0;
  virtual IItemFamily* itemFamily() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ExtraGhostItemsManager
{
public:

  /** Constructeur de la classe */
  ExtraGhostItemsManager(IExtraGhostItemsAdder* extra_ghost_adder)
    : m_item_family(extra_ghost_adder->itemFamily())
    , m_trace_mng(m_item_family->traceMng())
    , m_extra_ghost_items_adder(extra_ghost_adder){}

  ExtraGhostItemsManager(IItemFamily* item_family)
    : m_item_family(item_family)
    , m_trace_mng(item_family->traceMng()){}

  /** Destructeur de la classe */
  virtual ~ExtraGhostItemsManager() {}

public:

  IExtraGhostItemsAdder* extraGhostItemsFamily() {return m_extra_ghost_items_adder;}

  void addExtraGhostItemsBuilder(IExtraGhostItemsBuilder* builder) {
     m_builders.add(builder);
   }

   ArrayView<IExtraGhostItemsBuilder*> extraGhostItemsBuilders() {
     return m_builders;
   }

   void computeExtraGhostItems();
   void computeExtraGhostItems2(); // REFACTORED ONE

 private:

   IItemFamily* m_item_family;
   UniqueArray<IExtraGhostItemsBuilder*> m_builders;
   ITraceMng* m_trace_mng;
   IExtraGhostItemsAdder* m_extra_ghost_items_adder;

};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_MESH_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif /* EXTRAGHOSTITEMSMANAGER_H_ */
