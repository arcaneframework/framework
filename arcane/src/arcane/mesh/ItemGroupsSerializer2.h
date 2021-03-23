// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemGroupsSerializer2.h                                     (C) 2000-2016 */
/*                                                                           */
/* Sérialisation des groupes d'entités.                                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESH_ITEMGROUPSSERIALIZER2_H
#define ARCANE_MESH_ITEMGROUPSSERIALIZER2_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/List.h"
#include "arcane/utils/TraceAccessor.h"

#include "arcane/ArcaneTypes.h"
#include "arcane/ItemGroup.h"

#include "arcane/mesh/MeshGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IMesh;
class SerializeBuffer;
class IParallelExchanger;
class ItemFamilySerializeArgs;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_MESH_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Sérialise les entités des groupes.
 */
class ItemGroupsSerializer2
: public TraceAccessor
{
 public:
  
  ItemGroupsSerializer2(IItemFamily* item_family,IParallelExchanger* exchanger);
  virtual ~ItemGroupsSerializer2();

 public:
  
  void prepareData(ConstArrayView< SharedArray<Int32> > items_exchange);
  void serialize(const ItemFamilySerializeArgs& args);
  void get(ISerializer* sbuf,Int64Array& items_in_groups_uid);

  ItemGroupList groups() { return m_groups_to_exchange; }
  IMesh* mesh() const { return m_mesh; }
  //eItemKind itemKind() const { return m_item_kind; }
  IItemFamily* itemFamily() const { return m_item_family; }

 protected:

 private:

  IParallelExchanger* m_exchanger;
  IMesh* m_mesh;
  IItemFamily* m_item_family;
  /*! \brief Liste des groupes à échanger.
    
    IMPORTANT: Cette liste doit être identique pour tous les sous-domaines
    sinon les désérialisations vont donner des résultats incorrects.
  */
  ItemGroupList m_groups_to_exchange;
  //! Liste des entités à échanger par processeur
  UniqueArray< SharedArray<Int64> > m_items_to_send;

 private:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_MESH_END_NAMESPACE
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

