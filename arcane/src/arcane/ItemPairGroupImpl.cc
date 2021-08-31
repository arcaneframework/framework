// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemPairGroupImpl.cc                                        (C) 2000-2021 */
/*                                                                           */
/* Implémentation d'un tableau de listes d'entités.                          */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/utils/Array.h"
#include "arcane/utils/IFunctor.h"

#include "arcane/ItemGroupObserver.h"
#include "arcane/ItemPairGroupImpl.h"
#include "arcane/ItemPairGroup.h"
#include "arcane/IItemFamily.h"
#include "arcane/ItemGroup.h"
#include "arcane/IMesh.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Classe d'un groupe nul.
 */
class ItemPairGroupImplNull
: public ItemPairGroupImpl
{
 public:

  ItemPairGroupImplNull() : ItemPairGroupImpl() {}
  virtual ~ItemPairGroupImplNull() {} //!< Libère les ressources

 public:
 private:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 */
class ItemPairGroupImplPrivate
{
 public:

  ItemPairGroupImplPrivate();
  ItemPairGroupImplPrivate(const ItemGroup& group,const ItemGroup& sub_group);
  ~ItemPairGroupImplPrivate();

 public:

  inline bool null() const { return m_is_null; }
  inline IMesh* mesh() const { return m_mesh; }
  inline eItemKind kind() const { return m_kind; }
  inline eItemKind subKind() const { return m_sub_kind; }

 public:

  IMesh* m_mesh; //!< Gestionnare de groupe associé
  IItemFamily* m_item_family; //!< Famille associée
  IItemFamily* m_sub_item_family; //!< Famille associée
  ItemGroup m_item_group;
  ItemGroup m_sub_item_group;
  bool m_is_null; //!< \a true si le groupe est nul
  eItemKind m_kind; //!< Genre de entités du groupe
  eItemKind m_sub_kind;
  bool m_need_recompute; //!< Vrai si le groupe doit être recalculé
  
  UniqueArray<Int64> m_indexes;
  UniqueArray<Int32> m_sub_items_local_id;
  IFunctor* m_compute_functor;

 public:

  void invalidate() { m_need_recompute = true; }

 private:
  
  void _init();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemPairGroupImplPrivate::
ItemPairGroupImplPrivate()
: m_mesh(nullptr)
, m_item_family(nullptr)
, m_sub_item_family(nullptr)
, m_is_null(true)
, m_kind(IK_Unknown)
, m_sub_kind(IK_Unknown)
, m_need_recompute(false)
, m_compute_functor(nullptr)
{
  _init();
}
  
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemPairGroupImplPrivate::
ItemPairGroupImplPrivate(const ItemGroup& group,const ItemGroup& sub_group)
: m_mesh(group.mesh())
, m_item_family(group.itemFamily())
, m_sub_item_family(sub_group.itemFamily())
, m_item_group(group)
, m_sub_item_group(sub_group)
, m_is_null(false)
, m_kind(group.itemKind())
, m_sub_kind(sub_group.itemKind())
, m_need_recompute(false)
, m_compute_functor(nullptr)
{
  _init();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemPairGroupImplPrivate::
~ItemPairGroupImplPrivate()
{
  delete m_compute_functor;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemPairGroupImplPrivate::
_init()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemPairGroupImpl* ItemPairGroupImpl::shared_null= 0;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemPairGroupImpl* ItemPairGroupImpl::
checkSharedNull()
{
  if (!shared_null){
    shared_null = new ItemPairGroupImplNull();
    shared_null->addRef();
  }
  return shared_null;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemPairGroupImpl::
ItemPairGroupImpl(const ItemGroup& group,const ItemGroup& sub_group)
: m_p (new ItemPairGroupImplPrivate(group,sub_group))
{
  m_p->m_item_group.internal()->attachObserver(this,newItemGroupObserverT(m_p,&ItemPairGroupImplPrivate::invalidate));
  m_p->m_sub_item_group.internal()->attachObserver(this,newItemGroupObserverT(m_p,&ItemPairGroupImplPrivate::invalidate));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemPairGroupImpl::
ItemPairGroupImpl()
: m_p (new ItemPairGroupImplPrivate())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemPairGroupImpl::
~ItemPairGroupImpl()
{
  m_p->m_item_group.internal()->detachObserver(this);
  m_p->m_sub_item_group.internal()->detachObserver(this);
  delete m_p;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \todo a supprimer...
 */
void ItemPairGroupImpl::
deleteMe()
{
  delete this;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemPairGroupImpl::
addRef()
{
  SharedReference::addRef();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemPairGroupImpl::
removeRef()
{
  SharedReference::removeRef();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IMesh* ItemPairGroupImpl::
mesh() const
{
  return m_p->mesh();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IItemFamily* ItemPairGroupImpl::
itemFamily() const
{
  return m_p->m_item_family;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IItemFamily* ItemPairGroupImpl::
subItemFamily() const
{
  return m_p->m_sub_item_family;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool ItemPairGroupImpl::
null() const
{
  return m_p->null();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

eItemKind ItemPairGroupImpl::
itemKind() const 
{
  return m_p->kind();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

eItemKind ItemPairGroupImpl::
subItemKind() const 
{
  return m_p->subKind();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const ItemGroup& ItemPairGroupImpl::
itemGroup() const
{
  return m_p->m_item_group;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const ItemGroup& ItemPairGroupImpl::
subItemGroup() const
{
  return m_p->m_sub_item_group;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemPairGroupImpl::
checkValid()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemPairGroupImpl::
invalidate(bool force_recompute)
{
  m_p->m_need_recompute = true;
  if (force_recompute)
    checkNeedUpdate();    
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool ItemPairGroupImpl::
checkNeedUpdate()
{
  if (m_p->m_need_recompute){
    m_p->m_need_recompute = false;
    if (m_p->m_compute_functor)
      m_p->m_compute_functor->executeFunctor();
    return true;
  }
  return false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Array<Int64>& ItemPairGroupImpl::
unguardedIndexes() const
{
  return m_p->m_indexes;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Array<Int32>& ItemPairGroupImpl::
unguardedLocalIds() const
{
  return m_p->m_sub_items_local_id;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ArrayView<Int64> ItemPairGroupImpl::
indexes()
{
  checkNeedUpdate();
  return m_p->m_indexes;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Span<const Int32> ItemPairGroupImpl::
subItemsLocalId()
{
  checkNeedUpdate();
  return m_p->m_sub_items_local_id;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemPairGroupImpl::
setComputeFunctor(IFunctor* functor)
{
  delete m_p->m_compute_functor;
  m_p->m_compute_functor = functor;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
