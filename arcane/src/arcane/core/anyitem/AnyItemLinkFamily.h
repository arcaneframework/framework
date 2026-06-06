// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AnyItemLinkFamily.h                                         (C) 2000-2025 */
/*                                                                           */
/* Link family between any items.                                            */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ANYITEM_ANYITEMLINKFAMILY_H
#define ARCANE_CORE_ANYITEM_ANYITEMLINKFAMILY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/anyitem/AnyItemGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::AnyItem
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Tool to create a pair of items
 */
template <typename U, typename V>
class PairT
{
 public:

  PairT(U u, V v)
  : m_u(u)
  , m_v(v)
  {}
  U m_u;
  V m_v;
};

template <typename U, typename V>
inline PairT<U, V> Pair(U u, V v)
{
  return PairT<U, V>(u, v);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Internal AnyItem Link Family
 * The stored data are the local IDs of the items and the group offset within the family
 *
 * For example:
 *
 * AnyItem::LinkFamily link_family(family);
 * link_family.reserve(allFaces.size());
 *
 * ENUMERATE_FACE(iface, allCells().internalFaceGroup()) {
 *   AnyItem::LinkFamily::Link link = link_family.newLink();
 *   link(allFaces(),allCells()) << AnyItem::Pair(iface,iface->backCell());
 * }
 *
 */
class LinkFamilyInternal : public IFamilyObserver
{
 public:

  /*!
   * \brief Data per link
   */
  class LinkData
  {
    friend class LinkFamilyInternal;

#ifdef ARCANE_ANYITEM_USEPACKEDDATA
   public:

    LinkData()
    : m_packed_data(0)
    {}
    //! Identifier of the group associated with the item referenced by this LinkData
    Integer groupIndex() const { return m_packed_data >> m_group_shift; }
    //! LocalId identifier of the item referenced in its original IItemFamily
    Integer varIndex() const { return m_packed_data & m_integer_mask; }
    //! LocalId identifier of the item referenced in its AnyItemFamily
    Integer localId() const { return (m_packed_data >> m_local_id_shift) & m_integer_mask; }
    //! Comparison operator
    bool operator==(const LinkData& data) const { return m_packed_data == data.m_packed_data; }

   private:

    static const Integer m_integer_size = 26;
    static const Int64 m_integer_mask = (Int64(1) << m_integer_size) - 1;
    static const Integer m_short_size = 8;
    static const Int64 m_short_mask = (Int64(1) << m_short_size) - 1;
    static const Integer m_group_shift = 52;
    static const Int64 m_group_mask = m_short_mask << m_group_shift;
    static const Integer m_local_id_shift = m_integer_size;
    static const Int64 m_local_id_mask = m_integer_mask << m_local_id_shift;

    inline void setGroupIndex(Integer group_index)
    {
      ARCANE_ASSERT(((group_index & ~m_short_mask) == 0), ("Too large group index %d", group_index));
      m_packed_data = (m_packed_data & ~m_group_mask) | (Int64(group_index) << m_group_shift);
      ARCANE_ASSERT((groupIndex() == group_index), ("Corrupted data write detected"));
    }
    inline void setLocalId(Integer local_id)
    {
      ARCANE_ASSERT(((local_id & ~m_integer_mask) == 0), ("Too large local id %d", local_id));
      m_packed_data = (m_packed_data & ~m_local_id_mask) | (Int64(local_id) << m_local_id_shift);
      ARCANE_ASSERT((localId() == local_id), ("Corrupted data write detected"));
    }
    inline void setVarIndex(Integer item_local_id)
    {
      ARCANE_ASSERT(((item_local_id & ~m_integer_mask) == 0), ("Too large item local id %d", item_local_id));
      m_packed_data = (m_packed_data & ~m_integer_mask) | item_local_id;
      ARCANE_ASSERT((varIndex() == item_local_id), ("Corrupted data write detected"));
    }

   private:

    Int64 m_packed_data;
#else /* ARCANE_ANYITEM_USEPACKEDDATA */
   public:

    LinkData()
    : m_group_index(0)
    , m_local_id(0)
    , m_var_index(0)
    {}
    //! Identifier of the group associated with the item referenced by this LinkData
    Integer groupIndex() const { return m_group_index; }
    //! LocalId identifier of the item referenced in its original IItemFamily
    Integer varIndex() const { return m_var_index; }
    //! LocalId identifier of the item referenced in its AnyItemFamily
    Integer localId() const { return m_local_id; }
    //! Comparison operator
    bool operator==(const LinkData& data) const
    {
      return m_group_index == data.m_group_index && m_local_id == data.m_local_id && m_var_index == data.m_var_index;
    }

   private:

    inline void setGroupIndex(Integer group_index) { m_group_index = group_index; }
    inline void setLocalId(Integer local_id) { m_local_id = local_id; }
    inline void setVarIndex(Integer item_local_id) { m_var_index = item_local_id; }

   private:

    unsigned m_group_index; //!< Index of the group from which this item comes
    unsigned m_local_id; //!< Identifier in the global indexing of AnyItem::Family
    Integer m_var_index; //!< Index for partials, localId otherwise
#endif /* ARCANE_ANYITEM_USEPACKEDDATA */
  };

  /*!
   * \brief Link index
   *
   */
  class LinkIndex
  {
   public:

    LinkIndex()
    : m_index(0)
    {}
    LinkIndex(Integer index)
    : m_index(index)
    {}
    inline Integer index() const { return m_index; }

   protected:

    Integer m_index;
  };

  /*!
   * \brief Link enumerator
   */
  class Enumerator
  : public LinkIndex
  {
   public:

    Enumerator(const Arcane::Array<LinkData>& sources,
               const Arcane::Array<LinkData>& targets)
    : LinkIndex()
    , m_sources(sources)
    , m_targets(targets)
    {}
    Enumerator(const Enumerator& e)
    : LinkIndex()
    , m_sources(e.m_sources)
    , m_targets(e.m_targets)
    {}
    inline bool hasNext() const { return m_sources.size() != m_index; }
    inline void operator++() { m_index++; }
    //! Back link data
    inline const LinkData& back() const
    {
      return m_sources[m_index];
    }
    //! Front link data
    inline const LinkData& front() const
    {
      return m_targets[m_index];
    }

   private:

    //! All back data
    const Arcane::Array<LinkData>& m_sources;
    //! All front data
    const Arcane::Array<LinkData>& m_targets;
  };

  /*!
   * \brief Link
   */
  class Link
  : public LinkIndex
  {
   public:

    /*!
     * \brief Tool for link addition
     */
    template <typename U, typename V>
    struct LinkAdder
    {
      // Adder for a group pair
      LinkAdder(LinkFamilyInternal& family, ItemGroupT<U> a, ItemGroupT<V> b)
      : m_family(family)
      , m_a(a)
      , m_b(b)
      , m_used(false)
      {}
      ~LinkAdder()
      {
        ARCANE_ASSERT((m_used == true), ("LinkAdder never used"));
      }

      //! Addition of an item pair
      template <typename R, typename S>
      inline void operator<<(const PairT<R, S>& p)
      {
        ARCANE_ASSERT((m_used == false), ("VariableAdder already used"));
        m_family.addSourceNode(p.m_u, m_a);
        m_family.addTargetNode(p.m_v, m_b);
        m_used = true;
      }

     private:

      //! Link family
      LinkFamilyInternal& m_family;

      //! Back group
      ItemGroupT<U> m_a;

      //! Front group
      ItemGroupT<V> m_b;

      //! Indicator if the adder is used
      bool m_used;
    };

   public:

    Link(LinkFamilyInternal& family, Integer index)
    : LinkIndex(index)
    , m_family(family)
    , m_used(false)
    {}
    ~Link()
    {
      ARCANE_ASSERT((m_used == true), ("Link never used"));
    }

    //! Addition of links for groups a and b
    template <typename U, typename V>
    inline LinkAdder<U, V> operator()(const ItemGroupT<U>& a, const ItemGroupT<V>& b)
    {
      m_used = true;
      return LinkAdder<U, V>(m_family, a, b);
    }

   private:

    //! Link family
    LinkFamilyInternal& m_family;

    //! Indicator if the link is used
    bool m_used;
  };

 private:

  typedef std::set<ILinkFamilyObserver*> LinkFamilyObservers;

 public:

  //! Link family for an anyitem family
  LinkFamilyInternal(const Family& family)
  : m_family(family)
  , m_nb_link(0)
  {
    m_family.registerObserver(*this);
  }

  ~LinkFamilyInternal()
  {
    arcaneCallFunctionAndTerminateIfThrow([&]() { m_family.removeObserver(*this); });
  }

  //! Creation of a new empty link
  inline Link newLink()
  {
    if (m_nb_link >= capacity()) {
      m_source_nodes.reserve(2 * capacity());
      m_target_nodes.reserve(2 * capacity());
      _notifyFamilyIsReserved();
    }
    m_nb_link++;
    m_source_nodes.resize(m_nb_link);
    m_target_nodes.resize(m_nb_link);
    return Link(*this, m_nb_link - 1);
  }

  //! Reserves a capacity of links
  inline void reserve(Integer size)
  {
    m_source_nodes.reserve(size);
    m_target_nodes.reserve(size);
    _notifyFamilyIsReserved();
  }

  //! Link enumerators
  inline Enumerator enumerator() const { return Enumerator(m_source_nodes, m_target_nodes); }

  //! returns the capacity
  inline Integer capacity() const
  {
    return m_source_nodes.capacity();
  }

  //! Clears the family
  void clear()
  {
    m_nb_link = 0;
    m_source_nodes.clear();
    m_target_nodes.clear();
    _notifyFamilyIsInvalidate();
  }

  //! Registers a family observer
  void registerObserver(ILinkFamilyObserver& observer) const
  {
    LinkFamilyObservers::const_iterator it = m_observers.find(&observer);
    if (it != m_observers.end())
      throw FatalErrorException("LinkFamilyObserver already registered");
    m_observers.insert(&observer);
  }

  //! Removes a family observer
  void removeObserver(ILinkFamilyObserver& observer) const
  {
    LinkFamilyObservers::const_iterator it = m_observers.find(&observer);
    if (it == m_observers.end())
      throw FatalErrorException("LinkFamilyObserver not registered");
    m_observers.erase(it);
  }

  //! Notifies that the family is invalidated
  inline void notifyFamilyIsInvalidate()
  {
    // If the family changes, the link family is invalidated
    clear();
    _notifyFamilyIsInvalidate();
  }

  // Notifies that the family is increased
  inline void notifyFamilyIsIncreased()
  {
    // Do nothing in this case
  }

 public:

  template <typename T, typename V>
  inline void addSourceNode(const T& t, ItemGroupT<V> group)
  {
    //_addNode(t, group, m_source_nodes);
    initLinkData(m_source_nodes.back(), t, group);
  }

  template <typename T, typename V>
  inline void addTargetNode(const T& t, ItemGroupT<V> group)
  {
    //_addNode(t, group, m_target_nodes);
    initLinkData(m_target_nodes.back(), t, group);
  }

  const LinkData& source(const LinkIndex& link) const
  {
    return m_source_nodes[link.index()];
  }

  const LinkData& target(const LinkIndex& link) const
  {
    return m_target_nodes[link.index()];
  }

 public:

  const Family& family() const
  {
    return m_family;
  }

 private:

  // //! Adding link nodes by item type
  // // not optimal for partial variables, one must go look up the
  // // position of the item in the group
  // template<typename T>
  // inline void _addNode(const T& t, ItemGroupT<T> group, Arcane::Array<LinkData>& nodes) {
  //   LinkData& data = nodes.back();
  //   const Private::GroupIndexInfo * info = m_family.internal()->findGroupInfo(group);
  //   ARCANE_ASSERT((info != 0),("Inconsistent group info while adding new node"));
  //   data.setGroupIndex(info->group_index);
  //   const Integer local_index = (*(group.localIdToIndex()))[t.localId()];
  //   data.setLocalId(info->local_id_offset + local_index);
  //   if(info->is_partial) {
  //     const Integer local_index = (*(group.localIdToIndex()))[t.localId()];
  //     data.setVarIndex(local_index);
  //   } else {
  //     data.setVarIndex(t.localId());
  //   }
  // }

  // //! Adding link nodes by enumerators
  // // optimal for partial variables
  // template<typename T>
  // inline void _addNode(const ItemEnumeratorT<T>& t, ItemGroupT<T> group, Arcane::Array<LinkData>& nodes) {
  //   LinkData& data = nodes.back();
  //   const Private::GroupIndexInfo * info = m_family.internal()->findGroupInfo(group);
  //   ARCANE_ASSERT((info != 0),("Inconsistent group info while adding new node"));
  //   data.setGroupIndex(info->group_index);
  //   const Integer local_index = t.index();
  //   data.setLocalId(info->local_id_offset + local_index);
  //   if(info->is_partial) {
  //     data.setVarIndex(local_index);
  //   } else {
  //     data.setVarIndex(t.localId());
  //   }
  // }

 public:

  //! Adding link nodes by item type
  // not optimal for partial variables, one must go look up the
  // position of the item in the group
  template <typename T>
  void initLinkData(LinkData& data, const T& t, ItemGroupT<T> group) const
  {
    const Private::GroupIndexInfo* info = m_family.internal()->findGroupInfo(group);
    ARCANE_ASSERT((info != 0), ("Inconsistent group info while adding new node"));
    data.setGroupIndex(info->group_index);
    const Integer local_index = (*(group.localIdToIndex()))[t.localId()];
    data.setLocalId(info->local_id_offset + local_index);
    if (info->is_partial) {
      const Integer local_index = (*(group.localIdToIndex()))[t.localId()];
      data.setVarIndex(local_index);
    }
    else {
      data.setVarIndex(t.localId());
    }
  }

  //! Adding link nodes by enumerators
  // optimal for partial variables
  template <typename T>
  void initLinkData(LinkData& data, const ItemEnumeratorT<T>& t, ItemGroupT<T> group) const
  {
    const Private::GroupIndexInfo* info = m_family.internal()->findGroupInfo(group);
    ARCANE_ASSERT((info != 0), ("Inconsistent group info while adding new node"));
    data.setGroupIndex(info->group_index);
    const Integer local_index = t.index();
    data.setLocalId(info->local_id_offset + local_index);
    if (info->is_partial) {
      data.setVarIndex(local_index);
    }
    else {
      data.setVarIndex(t.localId());
    }
  }

  //! Returns the concrete item associated with this AnyItem
  Item item(const LinkData& link_data) const
  {
    return m_family.item(link_data);
  }

 private:

  void _notifyFamilyIsInvalidate()
  {
    for (LinkFamilyObservers::iterator it = m_observers.begin(); it != m_observers.end(); ++it)
      (*it)->notifyFamilyIsInvalidate();
  }

  void _notifyFamilyIsReserved()
  {
    for (LinkFamilyObservers::iterator it = m_observers.begin(); it != m_observers.end(); ++it)
      (*it)->notifyFamilyIsReserved();
  }

 private:

  //! AnyItem family
  const Family m_family;

  //! Back data
  Arcane::UniqueArray<LinkData> m_source_nodes;

  //! Front data
  Arcane::UniqueArray<LinkData> m_target_nodes;

  //! Number of links
  Integer m_nb_link;

  //! Family observers
  // So that objects built on the family cannot modify it
  mutable LinkFamilyObservers m_observers;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief AnyItem link family (flyweight pattern)
 */
class LinkFamily
: public IFamilyObserver
{
 public:

  typedef LinkFamilyInternal::Enumerator Enumerator;
  typedef LinkFamilyInternal::Link Link;
  typedef LinkFamilyInternal::LinkIndex LinkIndex;
  typedef LinkFamilyInternal::LinkData LinkData;

 public:

  LinkFamily(const Family& f)
  : m_internal(new LinkFamilyInternal(f))
  {}

  LinkFamily(const LinkFamily& f)
  : m_internal(f.m_internal)
  {}

  ~LinkFamily() {}

  //! Creation of a new empty link
  inline Link newLink()
  {
    return m_internal->newLink();
  }

  //! Reserves a capacity of links
  inline void reserve(Integer size)
  {
    m_internal->reserve(size);
  }

  //! Link enumerators
  inline Enumerator enumerator() const { return m_internal->enumerator(); }

  //! returns the capacity
  inline Integer capacity() const
  {
    return m_internal->capacity();
  }

  //! Clears the family
  void clear()
  {
    m_internal->clear();
  }

  //! Registers a family observer
  void registerObserver(ILinkFamilyObserver& observer) const
  {
    m_internal->registerObserver(observer);
  }

  //! Removes a family observer
  void removeObserver(ILinkFamilyObserver& observer) const
  {
    m_internal->removeObserver(observer);
  }

  //! Notifies that the family is invalidated
  inline void notifyFamilyIsInvalidate()
  {
    m_internal->notifyFamilyIsInvalidate();
  }

  //! Notifies that the family is increased
  inline void notifyFamilyIsIncreased()
  {
    m_internal->notifyFamilyIsIncreased();
  }

 public:

  template <typename T, typename V>
  inline void addSourceNode(const T& t, ItemGroupT<V> group)
  {
    m_internal->addSourceNode(t, group);
  }

  template <typename T, typename V>
  inline void addTargetNode(const T& t, ItemGroupT<V> group)
  {
    m_internal->addTargetNode(t, group);
  }

 public:

  LinkFamilyInternal* internal() const
  {
    return m_internal.get();
  }

 private:

  //! Internal link family
  SharedPtrT<LinkFamilyInternal> m_internal;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::AnyItem

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif /* ARCANE_ANYITEM_ANYITEMLINKFAMILY_H */
