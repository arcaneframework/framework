// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemConnectivity.h                                          (C) 2000-2023 */
/*                                                                           */
/* External connectivities. First version with DoF                           */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESH_ITEMCONNECTIVITY_H
#define ARCANE_MESH_ITEMCONNECTIVITY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/ITraceMng.h"

#include "arcane/IItemFamily.h"
#include "arcane/ItemVector.h"
#include "arcane/VariableTypes.h"
#include "arcane/ItemInternal.h"
#include "arcane/IItemConnectivity.h"
#include "arcane/ConnectivityItemVector.h"

#include "arcane/mesh/DoFFamily.h"
#include "arcane/mesh/ItemProperty.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe abstraite de gestion des connectivités.
 *
 * Cette classe gère les informations communes à tous les types de
 * connectivité comme son nom, les familles sources et cible, ...
 */
class ARCANE_MESH_EXPORT AbstractConnectivity
: public IItemConnectivity
{
 public:
  AbstractConnectivity(IItemFamily* source_family, IItemFamily* target_family, const String& connectivity_name)
  : m_source_family(source_family)
  , m_target_family(target_family)
  , m_name(connectivity_name)
  {
    m_families.add(m_source_family);
    m_families.add(m_target_family);
  }

 public:

  virtual const String& name() const
  {
    return m_name;
  }

 public:

  virtual ConstArrayView<IItemFamily*> families() const { return m_families.constView();}
  virtual IItemFamily* sourceFamily() const { return m_source_family;}
  virtual IItemFamily* targetFamily() const { return m_target_family;}
  virtual void _initializeStorage(ConnectivityItemVector*)
  {
    // Pour l'instant ne fait rien. A terme, cela pourra servir par exemple
    // pour dimensionner au nombre max d'entités connectées afin de ne pas
    // faire de réallocations lors de la récupération des entités via _connectedItems().
  }

 protected:

  ConstArrayView<IItemFamily*> _families() const { return m_families.constView();}
  IItemFamily* _sourceFamily() const { return m_source_family;}
  IItemFamily* _targetFamily() const { return m_target_family;}

 private:

  IItemFamily* m_source_family;
  IItemFamily* m_target_family;
  SharedArray<IItemFamily*> m_families;
  String m_name;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Connectivite item->item, exactement 1 item connecté par item (0 non admis).
 */
class ARCANE_MESH_EXPORT ItemConnectivity
: public AbstractConnectivity
{
public:
  typedef ItemScalarProperty<Int32> ItemPropertyType;

 public:
  ItemConnectivity(IItemFamily* source_family, IItemFamily* target_family,const String& aname)
  : AbstractConnectivity(source_family,target_family,aname)
  {
    compute();
  }

  ItemConnectivity(IItemFamily* source_family, IItemFamily* target_family,const ItemPropertyType& item_property, const String& aname)
    : AbstractConnectivity(source_family,target_family,aname) // IFPEN : voir le paramètre own de GE...
  {
    m_item_property.copy(item_property);
  }

 public:

  virtual ItemVectorView _connectedItems(ItemLocalId item,ConnectivityItemVector& con_items) const
  {
    ARCANE_ASSERT((con_items.accessor()==this),("Bad connectivity"));
    return con_items.setItem(m_item_property[item]);
 }

  virtual ConnectivityItemVectorCatalyst _connectedItems(ItemLocalId item) const
  {
    auto set   = [this](ConnectivityItemVector& civ)mutable{civ = ConnectivityItemVector((ItemConnectivity*)this);};
    auto apply = [this,item](ConnectivityItemVector& civ){this->_connectedItems(item,civ);};
    return {set,apply};
  }

  virtual void updateConnectivity(Int32ConstArrayView from_items, Int32ConstArrayView to_items);

 public:

  const Item operator() (ItemLocalId item) const
  {
    /*
    ARCANE_ASSERT((m_item_property[item] != NULL_ITEM_LOCAL_ID),
                  ("Item must be connected to one item in ItemConnectivity."));
    //TODO: conserver ItemInternalList pour des raisons de performance.
    return Item(_targetFamily()->itemsInternal()[m_item_property[item]]);
    */
    // Needed for IFPEN applicative test: eventually returns a null item (reasonable for perf ?)
    if (m_item_property[item] != NULL_ITEM_LOCAL_ID)
      return _targetFamily()->itemInfoListView()[m_item_property[item]];
    return Item();
  }

  ItemScalarProperty<Int32>& itemProperty() {return m_item_property;}

  void updateItemProperty(const ItemScalarProperty<Int32>& item_property) {m_item_property.copy(item_property);}

  virtual Integer nbConnectedItem(ItemLocalId lid) const
  {
    ARCANE_UNUSED(lid);
    return 1;
  }

  virtual Int32 connectedItemLocalId(ItemLocalId lid,[[maybe_unused]] Integer index) const
  {
    ARCANE_ASSERT((index==0),("Invalid value for index"))
    return m_item_property[lid];
  }

  //! Notifie la connectivité que la famille source est compactée.
  virtual void notifySourceFamilyLocalIdChanged(Int32ConstArrayView new_to_old_ids)
  {
    m_item_property.updateSupport(new_to_old_ids);
  }

  //! Notifie la connectivité que la famille cible est compactée.
  virtual void notifyTargetFamilyLocalIdChanged(Int32ConstArrayView old_to_new_ids);

 private:

  ItemScalarProperty<Int32> m_item_property;
  SharedArray<ItemInternal*> m_item_internals;

 private:

  void compute();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class FromItemType, class ToItemType>
class ItemConnectivityT
: public ItemConnectivity
{
 public:

  typedef typename FromItemType::LocalIdType FromLocalIdType;

 public:

  ItemConnectivityT(IItemFamily* source_family, IItemFamily* target_family,const String& connectivity_name)
  : ItemConnectivity(source_family,target_family,connectivity_name){}

  ItemConnectivityT(IItemFamily* source_family, IItemFamily* target_family,const ItemPropertyType& item_property,const String& connectivity_name)
  : ItemConnectivity(source_family,target_family,item_property, connectivity_name){}

 public:

  const ToItemType operator()(FromLocalIdType item) const
  {
    return ItemConnectivity::operator ()(item).itemBase();
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_MESH_EXPORT ItemArrayConnectivity
: public AbstractConnectivity
{
public:
  typedef ItemArrayProperty<Int32> ItemPropertyType;

 public:
  ItemArrayConnectivity(IItemFamily* source_family, IItemFamily* target_family,Integer nb_dof_per_item, const String& name)
  : AbstractConnectivity(source_family,target_family,name)
  , m_nb_dof_per_item(nb_dof_per_item)
  {
    compute();
  }

  ItemArrayConnectivity(IItemFamily* source_family, IItemFamily* target_family,const ItemPropertyType& item_property, const String& name)
  : AbstractConnectivity(source_family,target_family,name)
  , m_nb_dof_per_item(item_property.dim2Size())
    {
      m_item_property.copy(item_property);
    }

 public:

  virtual ItemVectorView _connectedItems(ItemLocalId item,ConnectivityItemVector& con_items) const
  {
    return this->operator()(item,con_items);
  }

  virtual ConnectivityItemVectorCatalyst _connectedItems(ItemLocalId item) const
  {
    return this->operator ()(item);
  }

  virtual void updateConnectivity(Int32ConstArrayView from_items, Int32ConstArrayView to_items);

 public:

  ItemArrayProperty<Int32>& itemProperty() { return m_item_property; }

  void updateItemProperty(const ItemArrayProperty<Int32>& item_property) {m_item_property.copy(item_property);}

  ItemVectorView operator() (ItemLocalId item,ConnectivityItemVector& con_items) const
  {
    ARCANE_ASSERT((con_items.accessor()==this),("Bad connectivity"));
    return con_items.resizeAndCopy(m_item_property[item]);
  }

  ConnectivityItemVectorCatalyst operator()(ItemLocalId item) const
  {
    auto set   = [this](ConnectivityItemVector& civ)mutable{civ = ConnectivityItemVector((ItemConnectivity*)this);};
    auto apply = [this,item](ConnectivityItemVector& civ){this->operator ()(item,civ);};
    return {set,apply};
  }

  virtual Integer nbConnectedItem(ItemLocalId lid) const
  {
    ARCANE_UNUSED(lid);
    return m_nb_dof_per_item;
  }

  virtual Int32 connectedItemLocalId(ItemLocalId lid,Integer index) const
  {
    return m_item_property[lid][index];
  }

  //! Notifie la connectivité que la famille source est compactée.
  virtual void notifySourceFamilyLocalIdChanged(Int32ConstArrayView new_to_old_ids)
  {
    m_item_property.updateSupport(new_to_old_ids);
  }

  //! Notifie la connectivité que la famille cible est compactée.
  virtual void notifyTargetFamilyLocalIdChanged(Int32ConstArrayView old_to_new_ids);

 private:

  Integer m_nb_dof_per_item;
  ItemArrayProperty<Int32> m_item_property;

 private:

  void compute();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class FromItemType,class ToItemType>
class ItemArrayConnectivityT
: public ItemArrayConnectivity
{
 public:

  typedef typename FromItemType::LocalIdType FromLocalIdType;

 public:

  ItemArrayConnectivityT(IItemFamily* source_family, IItemFamily* target_family,Integer nb_dof_per_item, const String& connectivity_name)
  : ItemArrayConnectivity(source_family,target_family,nb_dof_per_item,connectivity_name){}

  ItemArrayConnectivityT(IItemFamily* source_family, IItemFamily* target_family,const ItemPropertyType& item_property, const String& connectivity_name)
  : ItemArrayConnectivity(source_family,target_family,item_property,connectivity_name){}

  ItemVectorView operator() (FromLocalIdType item,ConnectivityItemVector& con_items) const
  {
    return ItemArrayConnectivity::operator()(item,con_items);
  }

  ConnectivityItemVectorCatalyst operator() (FromLocalIdType item) const
  {
    return ItemArrayConnectivity::operator()(item);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_MESH_EXPORT ItemMultiArrayConnectivity
: public AbstractConnectivity
{
public:
  typedef ItemMultiArrayProperty<Int32> ItemPropertyType;

 public:
  ItemMultiArrayConnectivity(IItemFamily* source_family, IItemFamily* target_family,IntegerConstArrayView nb_dof_per_item, const String& name)
 : AbstractConnectivity(source_family,target_family,name)
  {
    compute(nb_dof_per_item);
  }

  ItemMultiArrayConnectivity(IItemFamily* source_family, IItemFamily* target_family,const ItemPropertyType& item_property, const String& name)
 : AbstractConnectivity(source_family,target_family,name)
  {
    m_item_property.copy(item_property);
  }

 public:

  virtual ItemVectorView _connectedItems(ItemLocalId item,ConnectivityItemVector& con_items) const
  {
    return this->operator()(item,con_items);
  }

  virtual ConnectivityItemVectorCatalyst _connectedItems(ItemLocalId item) const
  {
    return this->operator ()(item);
  }

  virtual void updateConnectivity(Int32ConstArrayView from_items, Int32ConstArrayView to_items);

 public:
  
  ItemMultiArrayProperty<Int32>& itemProperty() {return m_item_property;}

  void updateItemProperty(ItemMultiArrayProperty<Int32>& item_property) {m_item_property.copy(item_property);}

  ItemVectorView operator() (ItemLocalId item,ConnectivityItemVector& con_items) const
  {
    ARCANE_ASSERT((con_items.accessor()==this),("Bad connectivity"));
    return con_items.resizeAndCopy(m_item_property[item]);
  }

  ConnectivityItemVectorCatalyst operator()(ItemLocalId item) const
  {
    auto set   = [this](ConnectivityItemVector& civ)mutable{civ = ConnectivityItemVector((ItemConnectivity*)this);};
    auto apply = [this,item](ConnectivityItemVector& civ){this->operator ()(item,civ);};
    return {set,apply};
  }


  virtual Integer nbConnectedItem(ItemLocalId lid) const
  {
    return m_item_property[lid].size();
  }

  virtual Int32 connectedItemLocalId(ItemLocalId lid,Integer index) const
  {
    return m_item_property[lid][index];
  }

  //! Notifie la connectivité que la famille source est compactée.
  virtual void notifySourceFamilyLocalIdChanged(Int32ConstArrayView new_to_old_ids)
  {
    m_item_property.updateSupport(new_to_old_ids);
  }

  //! Notifie la connectivité que la famille cible est compactée.
  virtual void notifyTargetFamilyLocalIdChanged(Int32ConstArrayView old_to_new_ids);

 private:

  ItemMultiArrayProperty<Int32> m_item_property;

 private:

  void compute(IntegerConstArrayView nb_dof_per_item);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class FromItemType,class ToItemType>
class ItemMultiArrayConnectivityT
: public ItemMultiArrayConnectivity
{
 public:

  typedef typename FromItemType::LocalIdType FromLocalIdType;

 public:

  ItemMultiArrayConnectivityT(IItemFamily* source_family, IItemFamily* target_family,const IntegerConstArrayView nb_dof_per_item, const String& connectivity_name)
  : ItemMultiArrayConnectivity(source_family,target_family,nb_dof_per_item,connectivity_name){}

  ItemMultiArrayConnectivityT(IItemFamily* source_family, IItemFamily* target_family,const ItemPropertyType& item_property, const String& connectivity_name)
  : ItemMultiArrayConnectivity(source_family,target_family,item_property,connectivity_name){}

 public:

  ItemVectorViewT<ToItemType> operator() (FromLocalIdType item,ConnectivityItemVector& con_items) const
  {
    return ItemMultiArrayConnectivity::operator() (item,con_items);
  }

  ConnectivityItemVectorCatalyst operator()(ItemLocalId item) const
  {
    return ItemMultiArrayConnectivity::operator() (item);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif /* CONNECTIVITY_H_ */
