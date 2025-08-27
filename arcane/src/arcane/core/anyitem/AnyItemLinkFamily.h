// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AnyItemLinkFamily.h                                         (C) 2000-2012 */
/*                                                                           */
/* Famille de liens entre any items.                                         */
/*---------------------------------------------------------------------------*/

#ifndef ARCANE_ANYITEM_ANYITEMLINKFAMILY_H
#define ARCANE_ANYITEM_ANYITEMLINKFAMILY_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/anyitem/AnyItemGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ANYITEM_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Outil pour créer une pair d'items
 */
template<typename U, typename V>
class PairT
{
public:
  PairT(U u, V v) : m_u(u), m_v(v) {}
  U m_u;
  V m_v;
};

template<typename U, typename V>
inline PairT<U,V> Pair(U u, V v) { return PairT<U,V>(u,v); }

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Famille de liens AnyItem partie interne
 * les données stockées sont les localid des items et l'offset du groupe dans la famille
 *
 * Par exemple :
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
   * \brief Données par liaisons
   */
  class LinkData
  {
    friend class LinkFamilyInternal;

#ifdef ARCANE_ANYITEM_USEPACKEDDATA
  public:
    LinkData() : m_packed_data(0) {}
    //! Identifiant du groupe auquel est associé l'item référencé par ce LinkData
    Integer groupIndex() const { return m_packed_data >> m_group_shift; }
    //! Identifiant localId de l'item référencé dans sa famille IItemFamily d'origine
    Integer varIndex() const { return m_packed_data & m_integer_mask; }
    //! Identifiant localId de l'item référencé dans sa famille AnyItemFamily
    Integer localId() const { return (m_packed_data >> m_local_id_shift) & m_integer_mask; }
    //! Opérateur de comparaison
    bool operator==(const LinkData& data) const { return m_packed_data == data.m_packed_data; }
private:
    static const Integer m_integer_size   = 26;
    static const Int64   m_integer_mask   = (Int64(1)<<m_integer_size)-1;
    static const Integer m_short_size     = 8;
    static const Int64   m_short_mask     = (Int64(1)<<m_short_size)-1;
    static const Integer m_group_shift    = 52;
    static const Int64   m_group_mask     = m_short_mask<<m_group_shift;
    static const Integer m_local_id_shift = m_integer_size;
    static const Int64   m_local_id_mask  = m_integer_mask<<m_local_id_shift;

    inline void setGroupIndex(Integer group_index) { 
      ARCANE_ASSERT(((group_index & ~m_short_mask) == 0),("Too large group index %d", group_index));
      m_packed_data = (m_packed_data & ~m_group_mask) | (Int64(group_index)<<m_group_shift);
      ARCANE_ASSERT((groupIndex() == group_index),("Corrupted data write detected"));
    }
    inline void setLocalId(Integer local_id) { 
      ARCANE_ASSERT(((local_id & ~m_integer_mask) == 0),("Too large local id %d", local_id));
      m_packed_data = (m_packed_data & ~m_local_id_mask) | (Int64(local_id)<<m_local_id_shift); 
      ARCANE_ASSERT((localId() == local_id),("Corrupted data write detected"));
    }
    inline void setVarIndex(Integer item_local_id) { 
      ARCANE_ASSERT(((item_local_id & ~m_integer_mask) == 0),("Too large item local id %d", item_local_id));
      m_packed_data = (m_packed_data & ~m_integer_mask) | item_local_id; 
      ARCANE_ASSERT((varIndex() == item_local_id),("Corrupted data write detected"));
    }
  private:
    Int64 m_packed_data;    
#else /* ARCANE_ANYITEM_USEPACKEDDATA */
  public:
    LinkData() : m_group_index(0), m_local_id(0), m_var_index(0) {}
    //! Identifiant du groupe auquel est associé l'item référencé par ce LinkData
    Integer groupIndex() const { return m_group_index; }
    //! Identifiant localId de l'item référencé dans sa famille IItemFamily d'origine
    Integer varIndex() const { return m_var_index; }
    //! Identifiant localId de l'item référencé dans sa famille AnyItemFamily
    Integer localId() const { return m_local_id; }
    //! Opérateur de comparaison
    bool operator==(const LinkData& data) const { 
      return m_group_index == data.m_group_index
        && m_local_id == data.m_local_id 
        && m_var_index == data.m_var_index; 
    }
  private:
    inline void setGroupIndex(Integer group_index) { m_group_index = group_index; }
    inline void setLocalId(Integer local_id) { m_local_id = local_id; }
    inline void setVarIndex(Integer item_local_id) { m_var_index = item_local_id; }
  private:
    unsigned m_group_index; //!< Index du groupe d'où vient cet item
    unsigned m_local_id; //!< Identifiant dans l(indexation globale de AnyItem::Family
    Integer m_var_index; //!< Index pour les partiels, localId sinon
#endif /* ARCANE_ANYITEM_USEPACKEDDATA */
  };
  
  /*!
   * \brief Indice par liaison
   * 
   */
  class LinkIndex 
  {
  public:
    LinkIndex() 
      : m_index(0) {}
    LinkIndex(Integer index)
      : m_index(index) {}
    inline Integer index() const { return m_index; }
  protected:
    Integer m_index;
  };
  
  /*!
   * \brief Enumérateur de liens
   */
  class Enumerator 
  : public LinkIndex
  {
  public:
    Enumerator(const Arcane::Array<LinkData>& sources, 
               const Arcane::Array<LinkData>& targets) 
    : LinkIndex(), m_sources(sources), m_targets(targets) {}
    Enumerator(const Enumerator& e) 
    : LinkIndex(), m_sources(e.m_sources), m_targets(e.m_targets) {}
    inline bool hasNext() const { return m_sources.size() != m_index; }
    inline void operator++() { m_index++; }
    //! Données du lien back
    inline const LinkData& back() const { 
      return m_sources[m_index];
    }
    //! Données du lien front
    inline const LinkData& front() const { 
      return m_targets[m_index];
    }
  private:
    //! Toutes les données back
    const Arcane::Array<LinkData>& m_sources;
    //! Toutes les données front
    const Arcane::Array<LinkData>& m_targets;
  };
   
  /*!
   * \brief Lien
   */
  class Link 
    : public LinkIndex
  {
  public:
    
    /*!
     * \brief Outil pour l'ajout de lien
     */
    template<typename U, typename V>
    struct LinkAdder
    {
      // Adder pour une pair de groupe
      LinkAdder(LinkFamilyInternal& family, ItemGroupT<U> a, ItemGroupT<V> b)
        : m_family(family), m_a(a), m_b(b), m_used(false) {}
      ~LinkAdder() 
      {
        ARCANE_ASSERT((m_used == true),("LinkAdder never used"));
      }
    
      //! Ajout d'une pair d'item
      template<typename R, typename S>
      inline void operator<<(const PairT<R,S>& p) 
      {
        ARCANE_ASSERT((m_used == false),("VariableAdder already used"));
        m_family.addSourceNode(p.m_u,m_a);
        m_family.addTargetNode(p.m_v,m_b);
        m_used = true;
      }
      
    private:
      //! Famille de liens
      LinkFamilyInternal& m_family;
      
      //! Groupe back
      ItemGroupT<U> m_a;

      //! groupe front
      ItemGroupT<V> m_b;
      
      //! Indicateur si le adder est utilisé
      bool m_used;
    };
    
  public:
    
    Link(LinkFamilyInternal& family, Integer index)
      : LinkIndex(index), m_family(family), m_used(false) {}
    ~Link() 
    {
      ARCANE_ASSERT((m_used == true),("Link never used"));
    }
    
    //! Ajout de liens pour les groupes a et b
    template<typename U, typename V>
    inline LinkAdder<U,V> operator()(const ItemGroupT<U>& a, const ItemGroupT<V>& b) 
    {
      m_used = true;
      return LinkAdder<U,V>(m_family,a,b);
    }

  private:
    
    //! Famille de liens
    LinkFamilyInternal& m_family;
    
    //! Indicateur si le lien est utilisé
    bool m_used;
  };

private:
  
  typedef std::set<ILinkFamilyObserver*> LinkFamilyObservers;
  
public:
  
  //! Famille de liens pour une famille anyitem
  LinkFamilyInternal(const Family& family)
    : m_family(family)
    , m_nb_link(0) 
  {
    m_family.registerObserver(*this);
  }

  ~LinkFamilyInternal() 
  {
    arcaneCallFunctionAndTerminateIfThrow([&]() { m_family.removeObserver(*this);});
  }
  
  //! Création d'un nouveau lien vide
  inline Link newLink() 
  {
    if(m_nb_link >= capacity()) {
      m_source_nodes.reserve(2 * capacity());
      m_target_nodes.reserve(2 * capacity());
      _notifyFamilyIsReserved();
    }
    m_nb_link++;
    m_source_nodes.resize(m_nb_link);
    m_target_nodes.resize(m_nb_link);
    return Link(*this,m_nb_link-1);
  }
 
  //! Réserve une capacité de liens
  inline void reserve(Integer size) {
    m_source_nodes.reserve(size);
    m_target_nodes.reserve(size);
    _notifyFamilyIsReserved();
  }

  //! Enumérateurs des liens
  inline Enumerator enumerator() const { return Enumerator(m_source_nodes, m_target_nodes); }

  //! retourne la capacité
  inline Integer capacity() const { 
    return m_source_nodes.capacity();
  }

  //! Vide la famille
  void clear() {
    m_nb_link = 0;
    m_source_nodes.clear();
    m_target_nodes.clear();
    _notifyFamilyIsInvalidate();
  }
  
  //! Enrgistre un observeur de la famille
  void registerObserver(ILinkFamilyObserver& observer) const
  {
    LinkFamilyObservers::const_iterator it = m_observers.find(&observer);
    if(it != m_observers.end())
      throw FatalErrorException("LinkFamilyObserver already registered");
    m_observers.insert(&observer);
  }
  
  //! Détruit un observeur de la famille
  void removeObserver(ILinkFamilyObserver& observer) const
  {
    LinkFamilyObservers::const_iterator it = m_observers.find(&observer);
    if(it == m_observers.end())
      throw FatalErrorException("LinkFamilyObserver not registered");
    m_observers.erase(it);
  }
  
  //! Notifie que la famille est invalidée
  inline void notifyFamilyIsInvalidate() {
    // Si la famille change, on invalide la famille des liaisons
    clear();
    _notifyFamilyIsInvalidate();
  }
  
  // Notifie que la famille est agrandie
  inline void notifyFamilyIsIncreased() {
    // On ne fait rien dans ce cas
  }

public:
  
  template<typename T, typename V>
  inline void addSourceNode(const T& t, ItemGroupT<V> group) {
    //_addNode(t, group, m_source_nodes);
    initLinkData(m_source_nodes.back(), t, group);
  }
 
  template<typename T, typename V>
  inline void addTargetNode(const T& t, ItemGroupT<V> group) {
    //_addNode(t, group, m_target_nodes);
    initLinkData(m_target_nodes.back(), t, group);
  }

  const LinkData& source(const LinkIndex& link) const {
    return m_source_nodes[link.index()];
  }

  const LinkData& target(const LinkIndex& link) const {
    return m_target_nodes[link.index()];
  }

public:

  const Family& family() const { 
    return m_family;
  }

private:
  
  // //! Ajout des noeuds des liaisons par type d'item
  // // non optimal pour les variables partielles, il faut aller checher la
  // // position de l'item dans le groupe
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
  
  // //! Ajout des noeuds des liaisons par énumerateurs
  // // optimal pour les variables partielles 
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

  //! Ajout des noeuds des liaisons par type d'item
  // non optimal pour les variables partielles, il faut aller checher la
  // position de l'item dans le groupe
  template<typename T>
  void initLinkData(LinkData& data, const T& t, ItemGroupT<T> group) const
  {
    const Private::GroupIndexInfo * info = m_family.internal()->findGroupInfo(group);
    ARCANE_ASSERT((info != 0),("Inconsistent group info while adding new node"));
    data.setGroupIndex(info->group_index);
    const Integer local_index = (*(group.localIdToIndex()))[t.localId()];
    data.setLocalId(info->local_id_offset + local_index);
    if(info->is_partial) {
      const Integer local_index = (*(group.localIdToIndex()))[t.localId()];
      data.setVarIndex(local_index);
    } else {
      data.setVarIndex(t.localId());
    }
  }

  //! Ajout des noeuds des liaisons par énumerateurs
  // optimal pour les variables partielles 
  template<typename T>
  void initLinkData(LinkData& data, const ItemEnumeratorT<T>& t, ItemGroupT<T> group) const
  {
    const Private::GroupIndexInfo * info = m_family.internal()->findGroupInfo(group);
    ARCANE_ASSERT((info != 0),("Inconsistent group info while adding new node"));
    data.setGroupIndex(info->group_index);
    const Integer local_index = t.index();
    data.setLocalId(info->local_id_offset + local_index);
    if(info->is_partial) {
      data.setVarIndex(local_index);
    } else {
      data.setVarIndex(t.localId());
    }
  }
  
  //! Retoune l'item concret associé à ce AnyItem
  Item item(const LinkData& link_data) const {
    return m_family.item(link_data);
  }

private:

  void _notifyFamilyIsInvalidate() {
    for(LinkFamilyObservers::iterator it = m_observers.begin(); it != m_observers.end(); ++it)
      (*it)->notifyFamilyIsInvalidate();
  }

  void _notifyFamilyIsReserved() {
    for(LinkFamilyObservers::iterator it = m_observers.begin(); it != m_observers.end(); ++it)
      (*it)->notifyFamilyIsReserved();
  }
  
private:
  
  //! Famille AnyItem
  const Family m_family;

  //! Données back
  Arcane::UniqueArray<LinkData> m_source_nodes;
  
  //! Données front
  Arcane::UniqueArray<LinkData> m_target_nodes;
  
  //! Nombre de liens
  Integer m_nb_link;
  
  //! Observeurs de la famille
  // Pour que les objets construits sur la famille ne puissent pas la modifier
  mutable LinkFamilyObservers m_observers;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Famille de liens AnyItem (pattern flyweight)
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
    : m_internal(new LinkFamilyInternal(f)) {}

  LinkFamily(const LinkFamily& f)
    : m_internal(f.m_internal) {}

  ~LinkFamily() {}
  
  //! Création d'un nouveau lien vide
  inline Link newLink() 
  {
    return m_internal->newLink();
  }
 
  //! Réserve une capacité de liens
  inline void reserve(Integer size) {
    m_internal->reserve(size);
  }

  //! Enumérateurs des liens
  inline Enumerator enumerator() const { return m_internal->enumerator(); }

  //! retourne la capacité
  inline Integer capacity() const { 
    return m_internal->capacity();
  }

  //! Vide la famille
  void clear() {
    m_internal->clear();
  }
  
  //! Enrgistre un observeur de la famille
  void registerObserver(ILinkFamilyObserver& observer) const
  {
    m_internal->registerObserver(observer);
  }
  
  //! Detruit un observeur de la famille
  void removeObserver(ILinkFamilyObserver& observer) const
  {
    m_internal->removeObserver(observer);
  }
  
  //! Notifie que la famille est invalidée
  inline void notifyFamilyIsInvalidate() {
    m_internal->notifyFamilyIsInvalidate();
  }
  
  //! Notifie que la famille est agrandie
  inline void notifyFamilyIsIncreased() {
    m_internal->notifyFamilyIsIncreased();
  }

public:
  
  template<typename T, typename V>
  inline void addSourceNode(const T& t, ItemGroupT<V> group) {
    m_internal->addSourceNode(t,group);
  }
 
  template<typename T, typename V>
  inline void addTargetNode(const T& t, ItemGroupT<V> group) {
    m_internal->addTargetNode(t,group);
  }

public:
  
  LinkFamilyInternal * internal() const {
    return m_internal.get();
  }

private:
  
  //! Famille de liens interne
  SharedPtrT<LinkFamilyInternal> m_internal;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ANYITEM_END_NAMESPACE
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif /* ARCANE_ANYITEM_ANYITEMLINKFAMILY_H */
