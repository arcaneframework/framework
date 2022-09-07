// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemEnumeratorBase.h                                        (C) 2000-2022 */
/*                                                                           */
/* Classe de base des énumérateurs sur les entités du maillage.              */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ITEMENUMERATORBASE_H
#define ARCANE_ITEMENUMERATORBASE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ItemInternalEnumerator.h"
#include "arcane/Item.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ItemEnumeratorCS;
class ItemGroupImpl;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe de base des énumérateurs sur une liste d'entité.
 *
 * Les instances de cette classes sont créées soit via ItemEnumerator, soit
 * via ItemEnumeratorT.
 */
template<typename ItemType>
class ItemEnumeratorBaseV1T
{
 private:

  using ItemInternalPtr = ItemInternal*;
  using LocalIdType = typename ItemType::LocalIdType;

 protected:

  ItemEnumeratorBaseV1T()
  : m_items(nullptr), m_local_ids(nullptr), m_index(0), m_count(0), m_group_impl(nullptr) { _init(); }
  ItemEnumeratorBaseV1T(const ItemInternalPtr* items,const Int32* local_ids,Integer n, const ItemGroupImpl * agroup = nullptr)
  : m_items(items), m_local_ids(local_ids), m_index(0), m_count(n), m_group_impl(agroup) { _init(); }
  ItemEnumeratorBaseV1T(const ItemInternalArrayView& items,const Int32ConstArrayView& local_ids, const ItemGroupImpl * agroup = nullptr)
  : m_items(items.data()), m_local_ids(local_ids.data()), m_index(0), m_count(local_ids.size()), m_group_impl(agroup) { _init(); }
  ItemEnumeratorBaseV1T(const ItemInternalVectorView& view, const ItemGroupImpl * agroup = nullptr)
  : m_items(view.items().data()), m_local_ids(view.localIds().data()),
    m_index(0), m_count(view.size()), m_group_impl(agroup) { _init(); }
  ItemEnumeratorBaseV1T(const ItemVectorView& rhs)
  : ItemEnumeratorBaseV1T((const ItemInternalVectorView&)rhs) {}
  ItemEnumeratorBaseV1T(const ItemVectorViewT<ItemType>& rhs)
  : ItemEnumeratorBaseV1T((const ItemInternalVectorView&)rhs) {}

  ItemEnumeratorBaseV1T(const ItemEnumerator& rhs);
  ItemEnumeratorBaseV1T(const ItemInternalEnumerator& rhs);

 public:

  //! Incrémente l'index de l'énumérateur
  constexpr void operator++() { ++m_index; }
  constexpr bool operator()() { return m_index<m_count; }

  //! Vrai si on n'a pas atteint la fin de l'énumérateur (index()<count())
  constexpr bool hasNext() { return m_index<m_count; }

  //! Nombre d'éléments de l'énumérateur
  constexpr Integer count() const { return m_count; }

  //! Indice courant de l'énumérateur
  constexpr Integer index() const { return m_index; }

  //! localId() de l'entité courante.
  constexpr Int32 itemLocalId() const { return m_local_ids[m_index]; }

  //! localId() de l'entité courante.
  constexpr Int32 localId() const { return m_local_ids[m_index]; }

  /*!
   * \internal
   * \brief Indices locaux.
   */
  constexpr const Int32* unguardedLocalIds() const { return m_local_ids; }

  /*!
   * \internal
   * \brief Liste des ItemInternal.
   */
  constexpr const ItemInternalPtr* unguardedItems() const { return m_items; }

  /*!
   * \internal
   * \brief Partie interne (pour usage interne uniquement).
   */
  constexpr ItemInternal* internal() const { return m_items[m_local_ids[m_index]]; }

  /*!
   * \brief Groupe sous-jacent s'il existe (nullptr sinon)
   *
   * \brief Ceci vise à pouvoir tester que les accès par ce énumérateur sur un objet partiel sont licites.
   */
  constexpr const ItemGroupImpl* group() const { return m_group_impl; }

  constexpr ItemType operator*() const { return m_items[ m_local_ids[m_index] ]; }
  constexpr const ItemType* operator->() const { m_item = m_items[ m_local_ids[m_index] ]; return &m_item; }

  constexpr LocalIdType asItemLocalId() const { return LocalIdType{m_local_ids[m_index]}; }

  static constexpr int version() { return 1; }

 public:

  ItemEnumerator toItemEnumerator() const;

 protected:

  const ItemInternalPtr* m_items;
  const Int32* ARCANE_RESTRICT m_local_ids;
  Int32 m_index;
  Int32 m_count;
  mutable ItemType m_item;
  const ItemGroupImpl* m_group_impl; // pourrait être retiré en mode release si nécessaire

 protected:

  //! Constructeur seulement utilisé par fromItemEnumerator()
  ItemEnumeratorBaseV1T(const ItemEnumerator& rhs,bool);

  ItemEnumeratorBaseV1T(const ItemInternalPtr* items,const Int32* local_ids,Int32 index,Int32 n,
                        const ItemGroupImpl * agroup,impl::ItemBase item_base)
  : m_items(items), m_local_ids(local_ids), m_index(index), m_count(n), m_item(item_base), m_group_impl(agroup)
  {
  }

 private:

  void _init() {}
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe de base des énumérateurs sur une liste d'entité.
 *
 * Les instances de cette classes sont créées soit via ItemEnumerator, soit
 * via ItemEnumeratorT.
 */
template<typename ItemType>
class ItemEnumeratorBaseV2T
{
 private:

  using ItemInternalPtr = ItemInternal*;
  using LocalIdType = typename ItemType::LocalIdType;

 protected:

  ItemEnumeratorBaseV2T()
  : m_items(nullptr), m_local_ids(nullptr), m_index(0), m_count(0), m_group_impl(nullptr) { _init(); }
  ItemEnumeratorBaseV2T(const ItemInternalPtr* items,const Int32* local_ids,Integer n, const ItemGroupImpl * agroup = nullptr)
  : m_items(items), m_local_ids(local_ids), m_index(0), m_count(n), m_group_impl(agroup) { _init(); }
  ItemEnumeratorBaseV2T(const ItemInternalArrayView& items,const Int32ConstArrayView& local_ids, const ItemGroupImpl * agroup = nullptr)
  : m_items(items.data()), m_local_ids(local_ids.data()), m_index(0), m_count(local_ids.size()), m_group_impl(agroup) { _init(); }
  ItemEnumeratorBaseV2T(const ItemInternalVectorView& view, const ItemGroupImpl * agroup = nullptr)
  : m_items(view.items().data()), m_local_ids(view.localIds().data()),
    m_index(0), m_count(view.size()), m_group_impl(agroup) { _init(); }
  ItemEnumeratorBaseV2T(const ItemVectorView& rhs)
  : ItemEnumeratorBaseV2T((const ItemInternalVectorView&)rhs) {}
  ItemEnumeratorBaseV2T(const ItemVectorViewT<ItemType>& rhs)
  : ItemEnumeratorBaseV2T((const ItemInternalVectorView&)rhs) {}

  ItemEnumeratorBaseV2T(const ItemEnumerator& rhs);
  ItemEnumeratorBaseV2T(const ItemInternalEnumerator& rhs);

 public:

  //! Incrémente l'index de l'énumérateur
  constexpr void operator++()
  {
    ++m_index;
    m_is_not_end = (m_index<m_count);
    if (m_is_not_end)
      m_base.m_local_id = m_local_ids[m_index];
  }
  constexpr bool operator()() { return m_is_not_end; }

  //! Vrai si on n'a pas atteint la fin de l'énumérateur (index()<count())
  constexpr bool hasNext() { return m_is_not_end; }

  //! Nombre d'éléments de l'énumérateur
  constexpr Integer count() const { return m_count; }

  //! Indice courant de l'énumérateur
  constexpr Integer index() const { return m_index; }

  //! localId() de l'entité courante.
  constexpr Int32 itemLocalId() const { return m_base.m_local_id; }

  //! localId() de l'entité courante.
  constexpr Int32 localId() const { return m_base.m_local_id; }

  /*!
   * \internal
   * \brief Indices locaux.
   */
  constexpr const Int32* unguardedLocalIds() const { return m_local_ids; }

  /*!
   * \internal
   * \brief Liste des ItemInternal.
   */
  constexpr const ItemInternalPtr* unguardedItems() const { return m_items; }

  /*!
   * \internal
   * \brief Partie interne (pour usage interne uniquement).
   */
  constexpr ItemInternal* internal() const { return m_items[m_base.m_local_id]; }

  /*!
   * \brief Groupe sous-jacent s'il existe (nullptr sinon)
   *
   * \brief Ceci vise à pouvoir tester que les accès par ce énumérateur sur un objet partiel sont licites.
   */
  constexpr const ItemGroupImpl* group() const { return m_group_impl; }

  constexpr ItemType operator*() const { return m_base; }
  constexpr const ItemType* operator->() const { return &m_base; }

  constexpr LocalIdType asItemLocalId() const { return LocalIdType{m_base.m_local_id}; }

  static constexpr int version() { return 2; }

 public:

  ItemEnumerator toItemEnumerator() const;

 protected:

  // TODO Rendre privé
  ItemType m_base;
  const ItemInternalPtr* m_items;
  const Int32* ARCANE_RESTRICT m_local_ids;
  Int32 m_index;
  Int32 m_count;
  bool m_is_not_end;
  const ItemGroupImpl* m_group_impl; // pourrait être retiré en mode release si nécessaire

 protected:

  //! Constructeur seulement utilisé par fromItemEnumerator()
  ItemEnumeratorBaseV2T(const ItemEnumerator& rhs,bool);

  ItemEnumeratorBaseV2T(const ItemInternalPtr* items,const Int32* local_ids,Int32 index,Int32 n,
                      const ItemGroupImpl * agroup,impl::ItemBase item_base)
  : m_base(item_base), m_items(items), m_local_ids(local_ids), m_index(index), m_count(n), m_group_impl(agroup)
  {
    m_is_not_end = (m_index<m_count);
  }

 private:

  void _init()
  {
    m_is_not_end = (m_index<m_count);
    if (m_is_not_end){
      Int32 lid = m_local_ids[m_index];
      // Vérifie qu'on n'indexe pas 'm_items' avec un localId() nul.
      // Cela n'est pas possible avec les groupes d'entités mais ca l'est avec
      // les ItemVector par exemple.
      // De même, 'm_items' peut-être nul si tous les localId() sont nuls.
      // Si c'est le cas, alors on prend comme ItemSharedInfo l'instance nulle.
      // A terme le ItemSharedInfo sera renseigné directement dans le constructeur de
      // cette classe.
      Int32 idx = lid;
      if (idx==NULL_ITEM_LOCAL_ID)
        idx = 0;
      ItemSharedInfo* isi = (m_items) ? m_items[idx]->sharedInfo() : ItemSharedInfo::nullInstance();
      m_base = ItemType(ItemBaseBuildInfo(lid,isi));
    }
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe de base des énumérateurs sur une liste d'entité.
 *
 * Les instances de cette classes sont créées soit via ItemEnumerator, soit
 * via ItemEnumeratorT.
 */
class ItemEnumeratorBaseV3
{
 private:

  using ItemInternalPtr = ItemInternal*;

 protected:

  ItemEnumeratorBaseV3()
  : m_local_ids(nullptr), m_index(0), m_count(0), m_group_impl(nullptr) { }
  ItemEnumeratorBaseV3(const ItemInternalPtr* items,const Int32* local_ids,Integer n, const ItemGroupImpl * agroup = nullptr)
  : m_local_ids(local_ids), m_index(0), m_count(n), m_group_impl(agroup) { _init(items); }
  ItemEnumeratorBaseV3(ItemSharedInfo* shared_info,const Int32* local_ids,Integer n)
  : m_shared_info(shared_info), m_local_ids(local_ids), m_index(0), m_count(n), m_group_impl(nullptr) { }
  ItemEnumeratorBaseV3(const ItemInternalArrayView& items,const Int32ConstArrayView& local_ids, const ItemGroupImpl * agroup = nullptr)
  : m_local_ids(local_ids.data()), m_index(0), m_count(local_ids.size()), m_group_impl(agroup) { _init(items.data()); }
  ItemEnumeratorBaseV3(const ItemInternalVectorView& view, const ItemGroupImpl* agroup = nullptr)
  : m_shared_info(view.m_shared_info), m_local_ids(view.localIds().data()),
    m_index(0), m_count(view.size()), m_group_impl(agroup) { }
  ItemEnumeratorBaseV3(const ItemVectorView& rhs)
  : ItemEnumeratorBaseV3((const ItemInternalVectorView&)rhs) {}

  ItemEnumeratorBaseV3(const ItemEnumerator& rhs);
  ItemEnumeratorBaseV3(const ItemInternalEnumerator& rhs);

 public:

  //! Incrémente l'index de l'énumérateur
  constexpr void operator++() { ++m_index; }
  constexpr bool operator()() { return m_index<m_count; }

  //! Vrai si on n'a pas atteint la fin de l'énumérateur (index()<count())
  constexpr bool hasNext() { return m_index<m_count; }

  //! Nombre d'éléments de l'énumérateur
  constexpr Integer count() const { return m_count; }

  //! Indice courant de l'énumérateur
  constexpr Integer index() const { return m_index; }

  //! localId() de l'entité courante.
  constexpr Int32 itemLocalId() const { return m_local_ids[m_index]; }

  //! localId() de l'entité courante.
  constexpr Int32 localId() const { return m_local_ids[m_index]; }

  /*!
   * \internal
   * \brief Indices locaux.
   */
  constexpr const Int32* unguardedLocalIds() const { return m_local_ids; }

  /*!
   * \internal
   * \brief Liste des ItemInternal.
   */
  constexpr const ItemInternalPtr* unguardedItems() const { return m_shared_info->m_items_internal.data(); }

  /*!
   * \internal
   * \brief Partie interne (pour usage interne uniquement).
   */
  constexpr ItemInternal* internal() const { return m_shared_info->m_items_internal[m_local_ids[m_index]]; }

  /*!
   * \brief Groupe sous-jacent s'il existe (nullptr sinon)
   *
   * \brief Ceci vise à pouvoir tester que les accès par ce énumérateur sur un objet partiel sont licites.
   */
  constexpr const ItemGroupImpl* group() const { return m_group_impl; }

  static constexpr int version() { return 3; }

 protected:

  ItemSharedInfo* m_shared_info = ItemSharedInfo::nullInstance();
  const Int32* ARCANE_RESTRICT m_local_ids;
  Int32 m_index;
  Int32 m_count;
  const ItemGroupImpl* m_group_impl; // pourrait être retiré en mode release si nécessaire

 protected:

  //! Constructeur seulement utilisé par fromItemEnumerator()
  ItemEnumeratorBaseV3(const ItemEnumerator& rhs,bool);

  ItemEnumeratorBaseV3(ItemSharedInfo* shared_info,const Int32* local_ids,Int32 index,Int32 n,
                       const ItemGroupImpl * agroup)
  : m_shared_info(shared_info), m_local_ids(local_ids), m_index(index), m_count(n), m_group_impl(agroup)
  {
  }

 private:

  void _init(const ItemInternalPtr* items)
  {
    m_shared_info = (items) ? items[0]->sharedInfo() : ItemSharedInfo::nullInstance();
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe de base des énumérateurs sur une liste d'entité.
 *
 * Les instances de cette classes sont créées soit via ItemEnumerator, soit
 * via ItemEnumeratorT.
 */
template<typename ItemType>
class ItemEnumeratorBaseV3T
: public ItemEnumeratorBaseV3
{
 private:

  using ItemInternalPtr = ItemInternal*;
  using LocalIdType = typename ItemType::LocalIdType;
  using BaseClass = ItemEnumeratorBaseV3;

 protected:

  ItemEnumeratorBaseV3T()
  : BaseClass() { }
  ItemEnumeratorBaseV3T(const ItemInternalPtr* items,const Int32* local_ids,Integer n, const ItemGroupImpl* agroup = nullptr)
  : BaseClass(items,local_ids,n,agroup) {}
  ItemEnumeratorBaseV3T(ItemSharedInfo* shared_info,const Int32* local_ids,Integer n)
  : BaseClass(shared_info,local_ids,n){}
  ItemEnumeratorBaseV3T(const ItemInternalArrayView& items,const Int32ConstArrayView& local_ids, const ItemGroupImpl * agroup = nullptr)
  : BaseClass(items,local_ids,agroup){}
  ItemEnumeratorBaseV3T(const ItemInternalVectorView& view, const ItemGroupImpl* agroup = nullptr)
  : BaseClass(view,agroup) {}
  ItemEnumeratorBaseV3T(const ItemVectorView& rhs)
  : ItemEnumeratorBaseV3T((const ItemInternalVectorView&)rhs) {}
  ItemEnumeratorBaseV3T(const ItemVectorViewT<ItemType>& rhs)
  : ItemEnumeratorBaseV3T((const ItemInternalVectorView&)rhs) {}

  ItemEnumeratorBaseV3T(const ItemEnumerator& rhs);
  ItemEnumeratorBaseV3T(const ItemInternalEnumerator& rhs);

 public:

  constexpr ItemType operator*() const { return ItemType(ItemBaseBuildInfo(m_local_ids[m_index],m_shared_info)); }
  constexpr const ItemType* operator->() const { m_item_for_operator_arrow = this->operator*(); return &m_item_for_operator_arrow; }

  constexpr LocalIdType asItemLocalId() const { return LocalIdType{m_local_ids[m_index]}; }

  ItemEnumerator toItemEnumerator() const;

 protected:

  mutable ItemType m_item_for_operator_arrow;

 protected:

  //! Constructeur seulement utilisé par fromItemEnumerator()
  ItemEnumeratorBaseV3T(const ItemEnumerator& rhs,bool);

  ItemEnumeratorBaseV3T(ItemSharedInfo* shared_info,const Int32* local_ids,Int32 index,Int32 n,
                        const ItemGroupImpl* agroup,impl::ItemBase item_base)
  : ItemEnumeratorBaseV3(shared_info,local_ids,index,n,agroup), m_item_for_operator_arrow(item_base)
  {
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
