// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Item.h                                                      (C) 2000-2025 */
/*                                                                           */
/* Informations sur les éléments du maillage.                                */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ITEM_H
#define ARCANE_CORE_ITEM_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ItemTypes.h"
#include "arcane/core/ItemInternal.h"
#include "arcane/core/ItemLocalId.h"

#include <atomic>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Macro pour vérifier en mode Check que les conversions entre les
// les genres d'entités sont correctes.
#ifdef ARCANE_CHECK
#define ARCANE_CHECK_KIND(type) _checkKind(type())
#else
#define ARCANE_CHECK_KIND(type)
#endif

#ifdef ARCANE_CHECK
#define ARCANE_WANT_ITEM_STAT
#endif

#ifdef ARCANE_WANT_ITEM_STAT
#define ARCANE_ITEM_ADD_STAT(var) ++var
#else
#define ARCANE_ITEM_ADD_STAT(var)
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe de base d'un élément de maillage.
 *
 * \ingroup Mesh

 Les éléments du maillage sont les noeuds (Node), les mailles (Cell),
 les faces (Face), les arêtes (Edge), les particules (Particle) ou les
 degrés de liberté (DoF). Chacun de ses éléments est décrit
 dans la classe dérivée correspondante.

 Cette classe et les classes dérivées sont des objets légers qui s'utilisent par
 valeur plutôt que par référence et qui ne doivent pas être conservés entre deux
 modifications du la famille (IItemFamily) à laquelle ils sont associés.

 Quel que soit son type un élément du maillage possède un identifiant
 unique (localId()) pour son type et local au sous-domaine géré et un identifiant
 unique (uniqueId()) pour son type sur l'ensemble du domaine. La numérotation est
 <b>continue</b> et commence à <b>0</b>. L'identifiant local est utilisé par
 exemple pour accéder aux variables ou pour la connectivité.

 Par exemple, si un maillage possède 2 mailles hexaédriques qui se joignent
 par une face, il y 12 noeuds, 11 faces et 2 mailles. Dans ce cas, le premier
 noeud aura l'identifiant 0, le second 1 et ainsi de suite jusqu'à 11. La
 première face aura l'identifiant 0, la seconde 1 et ainsi de suite
 jusqu'à 10.

 Il existe une entité correspondant à un objet nul. C'est la seule
 pour laquelle null() est vrai. Aucune opération autre que l'appel à null()
 et les opérations de comparaisons ne sont valides sur l'entité nulle.
 */
class ARCANE_CORE_EXPORT Item
{
  // Pour accéder aux constructeurs privés
  friend class ItemEnumeratorBaseT<Item>;
  friend class ItemConnectedEnumeratorBaseT<Item>;
  friend class ItemVector;
  friend class ItemVectorView;
  friend class ItemVectorViewConstIterator;
  friend class ItemConnectedListViewConstIterator;
  friend class SimdItem;
  friend class SimdItemEnumeratorBase;
  friend class ItemInfoListView;
  friend class ItemLocalIdToItemConverter;
  template<typename ItemType> friend class ItemLocalIdToItemConverterT;
  friend class ItemPairEnumerator;
  template<int Extent> friend class ItemConnectedListView;
  template<typename ItemType> friend class ItemEnumeratorBaseT;

  // Pour accéder à _internal()
  friend class ItemCompatibility;

 public:

  typedef ItemInternal* ItemInternalPtr;
  
  //! Type du localId()
  typedef ItemLocalId LocalIdType;

  using ItemBase = impl::ItemBase;

 public:

  /*!
   * \brief Index d'un Item dans une variable.
   * \deprecated
   */
  class Index
  {
    // TODO Rendre obsolète lorsqu'on aura supprimer
    // les classes dérivées qui sont obsolètes.
    // On ne peut pas le faire avant car cela génère trop
    // d'avertissements de compilation.
   public:
    Index() : m_local_id(NULL_ITEM_LOCAL_ID){}
    explicit Index(Int32 id) : m_local_id(id){}
    Index(Item item) : m_local_id(item.localId()){}
    operator ItemLocalId() const { return ItemLocalId{m_local_id}; }
   public:
    Int32 localId() const { return m_local_id; }
   private:
    Int32 m_local_id;
  };

 public:

  /*!
   * \brief Type des éléments.
   *
   * Les valeurs des types doivent aller de 0 à #NB_TYPE par pas de 1.
   *
   * \deprecated. Utilise les types définis dans ArcaneTypes.h
   */
  enum
  {
    Unknown ARCANE_DEPRECATED_REASON("Use 'IT_NullType' instead") = IT_NullType, //!< Elément de type nul
    Vertex ARCANE_DEPRECATED_REASON("Use 'IT_Vertex' instead") = IT_Vertex, //!< Elément de type noeud (1 sommet 1D, 2D et 3D)
    Bar2 ARCANE_DEPRECATED_REASON("Use 'IT_Line2' instead") = IT_Line2, //!< Elément de type arête (2 sommets, 1D, 2D et 3D)
    Tri3 ARCANE_DEPRECATED_REASON("Use 'IT_Triangle3' instead") = IT_Triangle3, //!< Elément de type triangle (3 sommets, 2D)
    Quad4 ARCANE_DEPRECATED_REASON("Use 'IT_Quad4' instead") = IT_Quad4, //!< Elément de type quad (4 sommets, 2D)
    Pentagon5 ARCANE_DEPRECATED_REASON("Use 'IT_Pentagon5' instead") = IT_Pentagon5, //!< Elément de type pentagone (5 sommets, 2D)
    Hexagon6 ARCANE_DEPRECATED_REASON("Use 'IT_Hexagon6' instead") = IT_Hexagon6, //!< Elément de type hexagone (6 sommets, 2D)
    Tetra ARCANE_DEPRECATED_REASON("Use 'IT_Tetraedron4' instead") = IT_Tetraedron4, //!< Elément de type tétraédre (4 sommets, 3D)
    Pyramid ARCANE_DEPRECATED_REASON("Use 'IT_Pyramid5' instead") = IT_Pyramid5, //!< Elément de type pyramide  (5 sommets, 3D)
    Penta ARCANE_DEPRECATED_REASON("Use 'IT_Pentaedron6' instead") = IT_Pentaedron6, //!< Elément de type pentaèdre (6 sommets, 3D)
    Hexa ARCANE_DEPRECATED_REASON("Use 'IT_Hexaedron8' instead") = IT_Hexaedron8, //!< Elément de type hexaèdre  (8 sommets, 3D)
    Wedge7 ARCANE_DEPRECATED_REASON("Use 'IT_Heptaedron10' instead") = IT_Heptaedron10, //!< Elément de type prisme à 7 faces (base pentagonale)
    Wedge8 ARCANE_DEPRECATED_REASON("Use 'IT_Octaedron12' instead") = IT_Octaedron12 //!< Elément de type prisme à 8 faces (base hexagonale)
    // Réduit au minimum pour compatibilité.
  };

  //! Indice d'un élément nul
  static const Int32 NULL_ELEMENT = NULL_ITEM_ID;

  //! Nom du type de maille \a cell_type
  ARCCORE_DEPRECATED_2021("Use ItemTypeMng::typeName() instead")
  static String typeName(Int32 type);

 protected:

  //! Constructeur réservé pour les énumérateurs
  constexpr ARCCORE_HOST_DEVICE Item(Int32 local_id,ItemSharedInfo* shared_info)
  : m_shared_info(shared_info), m_local_id(local_id) {}

 public:

  //! Création d'une entité de maillage nulle
  Item() = default;

  //! Construit une référence à l'entité \a internal
  //ARCANE_DEPRECATED_REASON("Remove this overload")
  Item(ItemInternal* ainternal)
  {
    ARCANE_CHECK_PTR(ainternal);
    m_shared_info = ainternal->m_shared_info;
    m_local_id  = ainternal->m_local_id;
    ARCANE_ITEM_ADD_STAT(m_nb_created_from_internal);
  }

  // NOTE: Pour le constructeur suivant; il est indispensable d'utiliser
  // const& pour éviter une ambiguité avec le constructeur par recopie
  //! Construit une référence à l'entité \a abase
  constexpr ARCCORE_HOST_DEVICE Item(const ItemBase& abase)
  : m_shared_info(abase.m_shared_info)
  , m_local_id(abase.m_local_id)
  {
  }

  //! Construit une référence à l'entité \a internal
  Item(const ItemInternalPtr* internals,Int32 local_id)
  : Item(local_id,internals[local_id]->m_shared_info)
  {
    ARCANE_ITEM_ADD_STAT(m_nb_created_from_internalptr);
  }

  //! Opérateur de copie
  Item& operator=(ItemInternal* ainternal)
  {
    _set(ainternal);
    return (*this);
  }

 public:

  //! \a true si l'entité est nul (i.e. non connecté au maillage)
  constexpr bool null() const { return m_local_id==NULL_ITEM_ID; }

  //! Identifiant local de l'entité dans le sous-domaine du processeur
  constexpr Int32 localId() const { return m_local_id; }

  //! Identifiant local de l'entité dans le sous-domaine du processeur
  constexpr ItemLocalId itemLocalId() const { return ItemLocalId{ m_local_id }; }

  //! Identifiant unique sur tous les domaines
  ItemUniqueId uniqueId() const
  {
#ifdef ARCANE_CHECK
    if (m_local_id!=NULL_ITEM_LOCAL_ID)
      arcaneCheckAt((Integer)m_local_id,m_shared_info->m_unique_ids.size());
#endif
    // Ne pas utiliser l'accesseur normal car ce tableau peut etre utilise pour la maille
    // nulle et dans ce cas m_local_id vaut NULL_ITEM_LOCAL_ID (qui est negatif)
    // ce qui provoque une exception pour debordement de tableau.
    return ItemUniqueId(m_shared_info->m_unique_ids.data()[m_local_id]);
  }

  //! Numéro du sous-domaine propriétaire de l'entité
  Int32 owner() const { return m_shared_info->_ownerV2(m_local_id); }

  //! Type de l'entité
  Int16 type() const { return m_shared_info->_typeId(m_local_id); }

  //! Type de l'entité
  ItemTypeId itemTypeId() const { return ItemTypeId(type()); }

  //! Famille dont est issue l'entité
  IItemFamily* itemFamily() const { return m_shared_info->m_item_family; }

  //! Genre de l'entité
  eItemKind kind() const { return m_shared_info->m_item_kind; }

  //! \a true si l'entité est appartient au sous-domaine
  bool isOwn() const { return (_flags() & ItemFlags::II_Own)!=0; }

  /*!
   * \brief Vrai si l'entité est partagé d'autres sous-domaines.
   *
   * Une entité est considérée comme partagée si et seulement si
   * isOwn() est vrai et elle est fantôme pour un ou plusieurs
   * autres sous-domaines.
   *
   * Cette méthode n'est pertinente que si les informations de connectivité
   * ont été calculées (par un appel à IItemFamily::computeSynchronizeInfos()).
   */
  bool isShared() const { return (_flags() & ItemFlags::II_Shared)!=0; }

  //! Converti l'entité en le genre \a ItemWithNodes.
  inline ItemWithNodes toItemWithNodes() const;
  //! Converti l'entité en le genre \a Node.
  inline Node toNode() const;
  //! Converti l'entité en le genre \a Cell.
  inline Cell toCell() const;
  //! Converti l'entité en le genre \a Edge.
  inline Edge toEdge() const;
  //! Converti l'entité en le genre \a Edge.
  inline Face toFace() const;
  //! Converti l'entité en le genre \a Particle.
  inline Particle toParticle() const;
  //! Converti l'entité en le genre \a DoF.
  inline DoF toDoF() const;

  //! Nombre de parents pour les sous-maillages
  Int32 nbParent() const { return _nbParent(); }

  //! i-ème parent pour les sous-maillages
  Item parent(Int32 i) const { return m_shared_info->_parentV2(m_local_id,i); }

  //! premier parent pour les sous-maillages
  Item parent() const { return m_shared_info->_parentV2(m_local_id,0); }

 public:

  //! \a true si l'entité est du genre \a ItemWithNodes.
  bool isItemWithNodes() const
  {
    eItemKind ik = kind();
    return (ik==IK_Unknown || ik==IK_Edge || ik==IK_Face || ik==IK_Cell );
  }

  //! \a true si l'entité est du genre \a Node.
  bool isNode() const
  {
    eItemKind ik = kind();
    return (ik==IK_Unknown || ik==IK_Node);
  }
  //! \a true si l'entité est du genre \a Cell.
  bool isCell() const
  {
    eItemKind ik = kind();
    return (ik==IK_Unknown || ik==IK_Cell);
  }
  //! \a true si l'entité est du genre \a Edge.
  bool isEdge() const
  {
    eItemKind ik = kind();
    return (ik==IK_Unknown || ik==IK_Edge);
  }
  //! \a true si l'entité est du genre \a Edge.
  bool isFace() const
  {
    eItemKind ik = kind();
    return (ik==IK_Unknown || ik==IK_Face);
  }
  //! \a true is l'entité est du genre \a Particle.
  bool isParticle() const
  {
    eItemKind ik = kind();
    return (ik==IK_Unknown || ik==IK_Particle);
  }
  //! \a true is l'entité est du genre \a DoF
  bool isDoF() const
  {
    eItemKind ik = kind();
    return (ik==IK_Unknown || ik==IK_DoF);
  }

 public:

 /*!
   * \brief Partie interne de l'entité.
   *
   * \warning La partie interne de l'entité ne doit être modifiée que
   * par ceux qui savent ce qu'ils font.
   * \deprecated Utiliser itemBase() ou mutableItemBase() à la place pour
   * les cas l'instance retournée n'est pas conservée.
   */
  ARCANE_DEPRECATED_REASON("Y2024: This method is internal to Arcane. use itemBase() or mutableItemBase() instead")
  ItemInternal* internal() const
  {
    if (m_local_id!=NULL_ITEM_LOCAL_ID)
      return m_shared_info->m_items_internal[m_local_id];
    return ItemInternal::nullItem();
  }

 public:

  /*!
   * \brief Partie interne de l'entité.
   *
   * \warning La partie interne de l'entité ne doit être modifiée que
   * par ceux qui savent ce qu'ils font.
   */
  impl::ItemBase itemBase() const
  {
    return impl::ItemBase(m_local_id,m_shared_info);
  }

  /*!
   * \brief Partie interne modifiable de l'entité.
   *
   * \warning La partie interne de l'entité ne doit être modifiée que
   * par ceux qui savent ce qu'ils font.
   */
  impl::MutableItemBase mutableItemBase() const
  {
    return impl::MutableItemBase(m_local_id,m_shared_info);
  }

  /*!
   * \brief Infos sur le type de l'entité.
   *
   * Cette méthode permet d'obtenir les informations concernant
   * un type donné d'entité , comme par exemple les numérotations locales
   * de ces faces ou de ses arêtes.
   */
  const ItemTypeInfo* typeInfo() const { return m_shared_info->typeInfoFromId(type()); }

 public:

  ARCANE_DEPRECATED_REASON("Y2022: Do not use this operator. Use operator '.' instead")
  Item* operator->() { return this; }

  ARCANE_DEPRECATED_REASON("Y2022: Do not use this operator. Use operator '.' instead")
  const Item* operator->() const { return this; }

 private:

  //! Infos partagées entre toutes les entités ayant les mêmes caractéristiques
  ItemSharedInfo* m_shared_info = ItemSharedInfo::nullItemSharedInfoPointer;

 protected:

  /*!
   * \brief Numéro local (au sous-domaine) de l'entité.
   *
   * Pour des raisons de performance, le numéro local doit être
   * le premier champs de la classe.
   */
  Int32 m_local_id = NULL_ITEM_LOCAL_ID;

 protected:

  void _checkKind(bool is_valid) const
  {
    if (!is_valid)
      _badConversion();
  }
  void _badConversion() const;
  void _set(ItemInternal* ainternal)
  {
    _setFromInternal(ainternal);
  }
  void _set(const Item& rhs)
  {
    _setFromItem(rhs);
  }

 protected:

  //! Flags de l'entité
  Int32 _flags() const { return m_shared_info->_flagsV2(m_local_id); }
  //! Nombre de noeuds de l'entité
  Integer _nbNode() const { return _connectivity()->_nbNodeV2(m_local_id); }
  //! Nombre d'arêtes de l'entité ou nombre d'arêtes connectés à l'entités (pour les noeuds)
  Integer _nbEdge() const { return _connectivity()->_nbEdgeV2(m_local_id); }
  //! Nombre de faces de l'entité ou nombre de faces connectés à l'entités (pour les noeuds et arêtes)
  Integer _nbFace() const { return _connectivity()->_nbFaceV2(m_local_id); }
  //! Nombre de mailles connectées à l'entité (pour les noeuds, arêtes et faces)
  Integer _nbCell() const { return _connectivity()->_nbCellV2(m_local_id); }
  //! Nombre de parent pour l'AMR
  Int32 _nbHParent() const { return _connectivity()->_nbHParentV2(m_local_id); }
  //! Nombre d' enfants pour l'AMR
  Int32 _nbHChildren() const { return _connectivity()->_nbHChildrenV2(m_local_id); }
  //! Nombre de parent pour les sous-maillages
  Integer _nbParent() const { return m_shared_info->nbParent(); }
  NodeLocalId _nodeId(Int32 index) const { return NodeLocalId(_connectivity()->_nodeLocalIdV2(m_local_id,index)); }
  EdgeLocalId _edgeId(Int32 index) const { return EdgeLocalId(_connectivity()->_edgeLocalIdV2(m_local_id,index)); }
  FaceLocalId _faceId(Int32 index) const { return FaceLocalId(_connectivity()->_faceLocalIdV2(m_local_id,index)); }
  CellLocalId _cellId(Int32 index) const { return CellLocalId(_connectivity()->_cellLocalIdV2(m_local_id,index)); }
  Int32 _hParentId(Int32 index) const { return _connectivity()->_hParentLocalIdV2(m_local_id,index); }
  Int32 _hChildId(Int32 index) const { return _connectivity()->_hChildLocalIdV2(m_local_id,index); }
  impl::ItemIndexedListView<DynExtent> _nodeList() const { return _connectivity()->nodeList(m_local_id); }
  impl::ItemIndexedListView<DynExtent> _edgeList() const { return _connectivity()->edgeList(m_local_id); }
  impl::ItemIndexedListView<DynExtent> _faceList() const { return _connectivity()->faceList(m_local_id); }
  impl::ItemIndexedListView<DynExtent> _cellList() const { return _connectivity()->cellList(m_local_id); }
  NodeLocalIdView _nodeIds() const { return _connectivity()->nodeLocalIdsView(m_local_id); }
  EdgeLocalIdView _edgeIds() const { return _connectivity()->edgeLocalIdsView(m_local_id); }
  FaceLocalIdView _faceIds() const { return _connectivity()->faceLocalIdsView(m_local_id); }
  CellLocalIdView _cellIds() const { return _connectivity()->cellLocalIdsView(m_local_id); }

  inline Node _node(Int32 index) const;
  inline Edge _edge(Int32 index) const;
  inline Face _face(Int32 index) const;
  inline Cell _cell(Int32 index) const;

  ItemBase _hParentBase(Int32 index) const { return _connectivity()->hParentBase(m_local_id, index, m_shared_info); }
  ItemBase _hChildBase(Int32 index) const { return _connectivity()->hChildBase(m_local_id, index, m_shared_info); }
  ItemBase _toItemBase() const { return ItemBase(m_local_id,m_shared_info); }

  //! Nombre de noeuds de l'entité
  Int32 _nbLinearNode() const { return itemBase()._nbLinearNode(); }

 private:

  ItemInternalConnectivityList* _connectivity() const
  {
    return m_shared_info->m_connectivity;
  }
  void _setFromInternal(ItemBase* rhs)
  {
    ARCANE_ITEM_ADD_STAT(m_nb_set_from_internal);
    m_local_id = rhs->m_local_id;
    m_shared_info = rhs->m_shared_info;
  }
  void _setFromItem(const Item& rhs)
  {
    m_local_id = rhs.m_local_id;
    m_shared_info = rhs.m_shared_info;
  }

 public:

  static void dumpStats(ITraceMng* tm);
  static void resetStats();

 private:

  static std::atomic<int> m_nb_created_from_internal;
  static std::atomic<int> m_nb_created_from_internalptr;
  static std::atomic<int> m_nb_set_from_internal;

 private:

  ItemInternal* _internal() const
  {
    if (m_local_id!=NULL_ITEM_LOCAL_ID)
      return m_shared_info->m_items_internal[m_local_id];
    return ItemInternal::nullItem();
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Compare deux entités.
 *
 * \retval true si elles sont identiques (mêmes localId())
 * \retval false sinon
 */
inline bool
operator==(const Item& item1,const Item& item2)
{
  return item1.localId()==item2.localId();
}

/*!
 * \brief Compare deux entités.
 *
 * \retval true si elles sont différentes (différents localId())
 * \retval false sinon
 */
inline bool
operator!=(const Item& item1,const Item& item2)
{
  return item1.localId()!=item2.localId();
}

/*!
 * \brief Compare deux entités.
 *
 * \retval true si elles sont inferieures (sur localId())
 * \retval false sinon
 */
inline bool
operator<(const Item& item1,const Item& item2)
{
  return item1.localId()<item2.localId();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ItemVectorView.h"
#include "arcane/ItemConnectedListView.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Noeud d'un maillage.
 *
 * \ingroup Mesh
 */
class ARCANE_CORE_EXPORT Node
: public Item
{
  using ThatClass = Node;
  // Pour accéder aux constructeurs privés
  friend class ItemEnumeratorBaseT<ThatClass>;
  friend class ItemConnectedEnumeratorBaseT<ThatClass>;
  friend class ItemVectorT<ThatClass>;
  friend class ItemVectorViewT<ThatClass>;
  friend class ItemConnectedListViewT<ThatClass>;
  friend class ItemVectorViewConstIteratorT<ThatClass>;
  friend class ItemConnectedListViewConstIteratorT<ThatClass>;
  friend class SimdItemT<ThatClass>;
  friend class ItemInfoListViewT<ThatClass>;
  friend class ItemLocalIdToItemConverterT<ThatClass>;

 public:

  /*!
   * \brief Index d'un Node dans une variable.
   * \deprecated
   */  
  class ARCANE_DEPRECATED_REASON("Y2024: Use NodeLocalId instead") Index
  : public Item::Index
  {
   public:
    typedef Item::Index Base;
   public:
    explicit Index(Int32 id) : Base(id){}
    Index(Node item) : Base(item){}
    operator NodeLocalId() const { return NodeLocalId{localId()}; }
  };

 protected:

  //! Constructeur réservé pour les énumérateurs
  Node(Int32 local_id,ItemSharedInfo* shared_info)
  : Item(local_id,shared_info) {}

 public:

  //! Type du localId()
  typedef NodeLocalId LocalIdType;

  //! Création d'un noeud non connecté au maillage
  Node() = default;

  //! Construit une référence à l'entité \a internal
  Node(ItemInternal* ainternal) : Item(ainternal)
  { ARCANE_CHECK_KIND(isNode); }

  //! Construit une référence à l'entité \a abase
  Node(const ItemBase& abase) : Item(abase)
  { ARCANE_CHECK_KIND(isNode); }

  //! Construit une référence à l'entité \a abase
  explicit Node(const Item& aitem) : Item(aitem)
  { ARCANE_CHECK_KIND(isNode); }

  //! Construit une référence à l'entité \a internal
  Node(const ItemInternalPtr* internals,Int32 local_id) : Item(internals,local_id)
  { ARCANE_CHECK_KIND(isNode); }

  //! Opérateur de copie
  Node& operator=(ItemInternal* ainternal)
  {
    _set(ainternal);
    return (*this);
  }

 public:

  //! Identifiant local de l'entité dans le sous-domaine du processeur
  NodeLocalId itemLocalId() const { return NodeLocalId{ m_local_id }; }

  //! Nombre d'arêtes connectées au noeud
  Int32 nbEdge() const { return _nbEdge(); }

  //! Nombre de faces connectées au noeud
  Int32 nbFace() const { return _nbFace(); }

  //! Nombre de mailles connectées au noeud
  Int32 nbCell() const { return _nbCell(); }

  //! i-ème arête du noeud
  inline Edge edge(Int32 i) const;

  //! i-ème face du noeud
  inline Face face(Int32 i) const;

  //! i-ème maille du noeud
  inline Cell cell(Int32 i) const;

  //! i-ème arête du noeud
  EdgeLocalId edgeId(Int32 i) const { return _edgeId(i); }

  //! i-ème face du noeud
  FaceLocalId faceId(Int32 i) const { return _faceId(i); }

  //! i-ème maille du noeud
  CellLocalId cellId(Int32 i) const { return _cellId(i); }

  //! Liste des arêtes du noeud
  EdgeConnectedListViewType edges() const { return _edgeList(); }

  //! Liste des faces du noeud
  FaceConnectedListViewType faces() const { return _faceList(); }

  //! Liste des mailles du noeud
  CellConnectedListViewType cells() const { return _cellList(); }

  //! Liste des arêtes du noeud
  EdgeLocalIdView edgeIds() const { return _edgeIds(); }

  //! Liste des faces du noeud
  FaceLocalIdView faceIds() const { return _faceIds(); }

  //! Liste des mailles du noeud
  CellLocalIdView cellIds() const { return _cellIds(); }

  // AMR

  //! Enumére les mailles connectées au noeud
  CellVectorView _internalActiveCells(Int32Array& local_ids) const
  {
    return _toItemBase()._internalActiveCells2(local_ids);
  }

  ARCANE_DEPRECATED_REASON("Y2022: Do not use this operator. Use operator '.' instead")
  Node* operator->() { return this; }

  ARCANE_DEPRECATED_REASON("Y2022: Do not use this operator. Use operator '.' instead")
  const Node* operator->() const { return this; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline Node Item::
_node(Int32 index) const
{
  return Node(_connectivity()->nodeBase(m_local_id,index));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Elément de maillage s'appuyant sur des noeuds (Edge,Face,Cell).
 *
 * \ingroup Mesh
 */
class ARCANE_CORE_EXPORT ItemWithNodes
: public Item
{
  using ThatClass = ItemWithNodes;
  // Pour accéder aux constructeurs privés
  friend class ItemEnumeratorBaseT<ThatClass>;
  friend class ItemConnectedEnumeratorBaseT<ThatClass>;
  friend class ItemVectorT<ThatClass>;
  friend class ItemVectorViewT<ThatClass>;
  friend class ItemConnectedListViewT<ThatClass>;
  friend class ItemVectorViewConstIteratorT<ThatClass>;
  friend class ItemConnectedListViewConstIteratorT<ThatClass>;
  friend class SimdItemT<ThatClass>;
  friend class ItemInfoListViewT<ThatClass>;
  friend class ItemLocalIdToItemConverterT<ThatClass>;

 protected:

  //! Constructeur réservé pour les énumérateurs
  ItemWithNodes(Int32 local_id,ItemSharedInfo* shared_info)
  : Item(local_id,shared_info) {}

 public:
  
  //! Création d'une entité non connectée au maillage
  ItemWithNodes() = default;

  //! Construit une référence à l'entité \a internal
  ItemWithNodes(ItemInternal* ainternal) : Item(ainternal)
  { ARCANE_CHECK_KIND(isItemWithNodes); }

  //! Construit une référence à l'entité \a abase
  ItemWithNodes(const ItemBase& abase) : Item(abase)
  { ARCANE_CHECK_KIND(isItemWithNodes); }

  //! Construit une référence à l'entité \a aitem
  explicit ItemWithNodes(const Item& aitem) : Item(aitem)
  { ARCANE_CHECK_KIND(isItemWithNodes); }

  //! Construit une référence à l'entité \a internal
  ItemWithNodes(const ItemInternalPtr* internals,Int32 local_id)
  : Item(internals,local_id)
  { ARCANE_CHECK_KIND(isItemWithNodes); }

  //! Opérateur de copie
  ItemWithNodes& operator=(ItemInternal* ainternal)
  {
    _set(ainternal);
    return (*this);
  }

 public:

  //! Nombre de noeuds de l'entité
  Int32 nbNode() const { return _nbNode(); }

  //! i-ème noeud de l'entité
  Node node(Int32 i) const { return _node(i); }

  //! Liste des noeuds de l'entité
  NodeConnectedListViewType nodes() const { return _nodeList(); }

  //! Liste des noeuds de l'entité
  NodeLocalIdView nodeIds() const { return _nodeIds(); }

  //! i-ème noeud de l'entité.
  NodeLocalId nodeId(Int32 index) const { return _nodeId(index); }

  //! Nombre de noeuds de l'entité linéaire associée (si entité ordre 2 ou plus)
  Int32 nbLinearNode() const { return _nbLinearNode(); }

 public:

  ARCANE_DEPRECATED_REASON("Y2022: Do not use this operator. Use operator '.' instead")
  ItemWithNodes* operator->() { return this; }

  ARCANE_DEPRECATED_REASON("Y2022: Do not use this operator. Use operator '.' instead")
  const ItemWithNodes* operator->() const { return this; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Arête d'une maille.
 *
 * Les arêtes n'existent qu'en 3D. En 2D, il faut utiliser la structure
 * 'Face'.
 *
 * \ingroup Mesh
 */
class ARCANE_CORE_EXPORT Edge
: public ItemWithNodes
{
  using ThatClass = Edge;
  // Pour accéder aux constructeurs privés
  friend class ItemEnumeratorBaseT<ThatClass>;
  friend class ItemConnectedEnumeratorBaseT<ThatClass>;
  friend class ItemVectorT<ThatClass>;
  friend class ItemVectorViewT<ThatClass>;
  friend class ItemConnectedListViewT<ThatClass>;
  friend class ItemVectorViewConstIteratorT<ThatClass>;
  friend class ItemConnectedListViewConstIteratorT<ThatClass>;
  friend class SimdItemT<ThatClass>;
  friend class ItemInfoListViewT<ThatClass>;
  friend class ItemLocalIdToItemConverterT<ThatClass>;

 public:

  /*!
   * \brief Index d'une Edge dans une variable.
   * \deprecated
   */
  class ARCANE_DEPRECATED_REASON("Y2024: Use EdgeLocalId instead") Index
  : public Item::Index
  {
   public:
    typedef Item::Index Base;
   public:
    explicit Index(Int32 id) : Base(id){}
    Index(Edge item) : Base(item){}
    operator EdgeLocalId() const { return EdgeLocalId{localId()}; }
  };

  private:

  //! Constructeur réservé pour les énumérateurs
  Edge(Int32 local_id,ItemSharedInfo* shared_info)
  : ItemWithNodes(local_id,shared_info) {}

 public:

  //! Type du localId()
  typedef EdgeLocalId LocalIdType;

  //! Créé une arête nulle
  Edge() = default;

  //! Construit une référence à l'entité \a internal
  Edge(ItemInternal* ainternal) : ItemWithNodes(ainternal)
  { ARCANE_CHECK_KIND(isEdge); }

  //! Construit une référence à l'entité \a abase
  Edge(const ItemBase& abase) : ItemWithNodes(abase)
  { ARCANE_CHECK_KIND(isEdge); }

  //! Construit une référence à l'entité \a aitem
  explicit Edge(const Item& aitem) : ItemWithNodes(aitem)
  { ARCANE_CHECK_KIND(isEdge); }

  //! Construit une référence à l'entité \a internal
  Edge(const ItemInternalPtr* internals,Int32 local_id)
  : ItemWithNodes(internals,local_id)
  { ARCANE_CHECK_KIND(isEdge); }

  //! Opérateur de copie
  Edge& operator=(ItemInternal* ainternal)
  {
    _set(ainternal);
    return (*this);
  }

 public:

  //! Identifiant local de l'entité dans le sous-domaine du processeur
  EdgeLocalId itemLocalId() const { return EdgeLocalId{ m_local_id }; }

  //! Nombre de sommets de l'arête
  Int32 nbNode() const { return 2; }

  //! Nombre de faces connectées à l'arête
  Int32 nbFace() const { return _nbFace(); }

  //! Nombre de mailles connectées à l'arête
  Int32 nbCell() const { return _nbCell(); }

  //! i-ème maille de l'arête
  inline Cell cell(Int32 i) const;

  //! Liste des mailles de l'arête
  CellConnectedListViewType cells() const { return _cellList(); }

  //! i-ème maille de l'arête
  CellLocalId cellId(Int32 i) const { return _cellId(i); }

  //! Liste des mailles de l'arête
  CellLocalIdView cellIds() const { return _cellIds(); }

  //! i-ème face de l'arête
  inline Face face(Int32 i) const;

  //! Liste des faces de l'arête
  FaceConnectedListViewType faces() const { return _faceList(); }

  //! i-ème face de l'arête
  FaceLocalId faceId(Int32 i) const { return _faceId(i); }

  //! Liste des faces de l'arête
  FaceLocalIdView faceIds() const { return _faceIds(); }

  ARCANE_DEPRECATED_REASON("Y2022: Do not use this operator. Use operator '.' instead")
  Edge* operator->() { return this; }

  ARCANE_DEPRECATED_REASON("Y2022: Do not use this operator. Use operator '.' instead")
  const Edge* operator->() const { return this; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline Edge Item::
_edge(Int32 index) const
{
  return Edge(_connectivity()->edgeBase(m_local_id,index));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Face d'une maille.
 *
 * \ingroup Mesh
 *
 Une face est décrite par la liste ordonnée de ses sommets, ce qui lui
 donne une orientation.
 */
class ARCANE_CORE_EXPORT Face
: public ItemWithNodes
{
  using ThatClass = Face;
  // Pour accéder aux constructeurs privés
  friend class ItemEnumeratorBaseT<ThatClass>;
  friend class ItemConnectedEnumeratorBaseT<ThatClass>;
  friend class ItemVectorT<ThatClass>;
  friend class ItemVectorViewT<ThatClass>;
  friend class ItemConnectedListViewT<ThatClass>;
  friend class ItemVectorViewConstIteratorT<ThatClass>;
  friend class ItemConnectedListViewConstIteratorT<ThatClass>;
  friend class SimdItemT<ThatClass>;
  friend class ItemInfoListViewT<ThatClass>;
  friend class ItemLocalIdToItemConverterT<ThatClass>;

 public:

  /*!
   * \brief Index d'une Face dans une variable.
   * \deprecated
   */
  class ARCANE_DEPRECATED_REASON("Y2024: Use FaceLocalId instead") Index
  : public Item::Index
  {
   public:
    typedef Item::Index Base;
   public:
    explicit Index(Int32 id) : Base(id){}
    Index(Face item) : Base(item){}
    operator FaceLocalId() const { return FaceLocalId{localId()}; }
  };

 private:

  //! Constructeur réservé pour les énumérateurs
  Face(Int32 local_id,ItemSharedInfo* shared_info)
  : ItemWithNodes(local_id,shared_info) {}

 public:

  //! Type du localId()
  typedef FaceLocalId LocalIdType;

  //! Création d'une face non connecté au maillage
  Face() = default;

  //! Construit une référence à l'entité \a internal
  Face(ItemInternal* ainternal) : ItemWithNodes(ainternal)
  { ARCANE_CHECK_KIND(isFace); }

  //! Construit une référence à l'entité \a abase
  Face(const ItemBase& abase) : ItemWithNodes(abase)
  { ARCANE_CHECK_KIND(isFace); }

  //! Construit une référence à l'entité \a aitem
  explicit Face(const Item& aitem) : ItemWithNodes(aitem)
  { ARCANE_CHECK_KIND(isFace); }

  //! Construit une référence à l'entité \a internal
  Face(const ItemInternalPtr* internals,Int32 local_id)
  : ItemWithNodes(internals,local_id)
  { ARCANE_CHECK_KIND(isFace); }

  //! Opérateur de copie
  Face& operator=(ItemInternal* ainternal)
  {
    _set(ainternal);
    return (*this);
  }

 public:

  //! Identifiant local de l'entité dans le sous-domaine du processeur
  FaceLocalId itemLocalId() const { return FaceLocalId{ m_local_id }; }

  //! Nombre de mailles de la face (1 ou 2)
  Int32 nbCell() const { return _nbCell(); }

  //! i-ème maille de la face
  inline Cell cell(Int32 i) const;

  //! Liste des mailles de la face
  CellConnectedListViewType cells() const { return _cellList(); }

  //! i-ème maille de la face
  CellLocalId cellId(Int32 i) const { return _cellId(i); }

  //! Liste des mailles de la face
  CellLocalIdView cellIds() const { return _cellIds(); }

  /*!
   * \brief Indique si la face est au bord du sous-domaine (i.e nbCell()==1)
   *
   * \warning Une face au bord du sous-domaine n'est pas nécessairement au bord du maillage global.
   */
  bool isSubDomainBoundary() const { return (_flags() & ItemFlags::II_Boundary)!=0; }

  /*!
   * \a true si la face est au bord du sous-domaine.
   * \deprecated Utiliser isSubDomainBoundary() à la place.
   */
  ARCANE_DEPRECATED_118 bool isBoundary() const { return isSubDomainBoundary(); }

  //! Indique si la face est au bord t orientée vers l'extérieur.
  bool isSubDomainBoundaryOutside() const
  {
    return isSubDomainBoundary() && (_flags() & ItemFlags::II_HasBackCell);
  }

  /*!
   * \brief Indique si la face est au bord t orientée vers l'extérieur.
   *
   * \deprecated Utiliser isSubDomainBoundaryOutside()
   */
  ARCANE_DEPRECATED_118 bool isBoundaryOutside() const
  {
    return isSubDomainBoundaryOutside();
  }

  //! Maille associée à cette face frontière (maille nulle si aucune)
  inline Cell boundaryCell() const;

  //! Maille derrière la face (maille nulle si aucune)
  inline Cell backCell() const;

  //! Maille derrière la face (maille nulle si aucune)
  CellLocalId backCellId() const { return CellLocalId(_toItemBase().backCellId()); }

  //! Maille devant la face (maille nulle si aucune)
  inline Cell frontCell() const;

  //! Maille devant la face (maille nulle si aucune)
  CellLocalId frontCellId() const { return CellLocalId(_toItemBase().frontCellId()); }

  /*!
   * \brief Maille opposée de cette face à la maille \a cell.
   *
   * \pre backCell()==cell || frontCell()==cell.
   */
  inline Cell oppositeCell(Cell cell) const;

  /*!
   * \brief Maille opposée de cette face à la maille \a cell.
   *
   * \pre backCell()==cell || frontCell()==cell.
   */
  CellLocalId oppositeCellId(CellLocalId cell_id) const
  {
    ARCANE_ASSERT((backCellId()==cell_id || frontCellId()==cell_id),("cell is not connected to the face"));
    return (backCellId()==cell_id) ? frontCellId() : backCellId();
  }

  /*!
   * \brief Face maître associée à cette face.
   *
   * Cette face n'est non nul que si la face est liée à une interface
   * et est une face esclave de cette interface (i.e. isSlaveFace() est vrai)
   *
   * \sa ITiedInterface
   */
  Face masterFace() const { return _toItemBase().masterFace(); }

  //! \a true s'il s'agit de la face maître d'une interface
  bool isMasterFace() const { return _toItemBase().isMasterFace(); }

  //! \a true s'il s'agit d'une face esclave d'une interface
  bool isSlaveFace() const { return _toItemBase().isSlaveFace(); }

  //! \a true s'il s'agit d'une face esclave ou maître d'une interface
  bool isTiedFace() const { return isSlaveFace() || isMasterFace(); }

  /*!
   * \brief Liste des faces esclaves associées à cette face maître.
   *
   * Cette liste n'existe que pour les faces dont isMasterFace() est vrai.
   * Pour les autres, elle est vide.
   */
  FaceConnectedListViewType slaveFaces() const
  {
    if (_toItemBase().isMasterFace())
      return _faceList();
    return FaceConnectedListViewType();
  }

 public:

  //! Nombre d'arêtes de la face
  Int32 nbEdge() const { return _nbEdge(); }

  //! i-ème arête de la face
  Edge edge(Int32 i) const { return _edge(i); }

  //! Liste des arêtes de la face
  EdgeConnectedListViewType edges() const { return _edgeList(); }

  //! i-ème arête de la face
  EdgeLocalId edgeId(Int32 i) const { return _edgeId(i); }

  //! Liste des arêtes de la face
  EdgeLocalIdView edgeIds() const { return _edgeIds(); }

  ARCANE_DEPRECATED_REASON("Y2022: Do not use this operator. Use operator '.' instead")
  Face* operator->() { return this; }

  ARCANE_DEPRECATED_REASON("Y2022: Do not use this operator. Use operator '.' instead")
  const Face* operator->() const { return this; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline Face Item::
_face(Int32 index) const
{
  return Face(_connectivity()->faceBase(m_local_id,index));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Maille d'un maillage.
 *
 * \ingroup Mesh
 *
 Chaque maille utilise de la mémoire pour stocker sa connectivité. Cela
 permet aux modules d'écrire leur boucle de manière identique quelle que
 soit le type de la maille. Dans un premier temps, c'est le mécanisme le
 plus simple. On peut envisager par la suite d'utiliser des classes template
 pour traiter la même information de manière statique (i.e. toute la connectivité
 est gérée à la compilation).

 La connectivité utilise la numérotation <b>locale</b> des sommets de
 la maille. Elle est stockée dans les variables de classe #global_face_list
 pour les faces et #global_edge_list pour les arêtes.

 La connectivité utilisée est celle qui est décrite dans la notice
 LIMA version 3.1 à ceci près que la numérotation commence à zéro et non
 pas à un.

 LIMA ne décrivant pas la pyramide, la numérotation utilisée est celle
 de l'hexaèdre dégénérée en considérant que les sommets 4, 5, 6 et 7
 sont le sommet de la pyramide

 Dans la version actuelle (1.6), les arêtes ne sont pas prises en compte
 de manière globale (i.e: il n'y a pas d'entités Edge par maille).
*/
class ARCANE_CORE_EXPORT Cell
: public ItemWithNodes
{
  using ThatClass = Cell;
  // Pour accéder aux constructeurs privés
  friend class ItemEnumeratorBaseT<ThatClass>;
  friend class ItemConnectedEnumeratorBaseT<ThatClass>;
  friend class ItemVectorT<ThatClass>;
  friend class ItemVectorViewT<ThatClass>;
  friend class ItemConnectedListViewT<ThatClass>;
  friend class ItemVectorViewConstIteratorT<ThatClass>;
  friend class ItemConnectedListViewConstIteratorT<ThatClass>;
  friend class SimdItemT<ThatClass>;
  friend class ItemInfoListViewT<ThatClass>;
  friend class ItemLocalIdToItemConverterT<ThatClass>;

 public:

  /*!
   * \brief Index d'une Cell dans une variable.
   * \deprecated
   */  
  class ARCANE_DEPRECATED_REASON("Y2024: Use CellLocalId instead") Index
  : public Item::Index
  {
   public:
    typedef Item::Index Base;
   public:
    explicit Index(Int32 id) : Base(id){}
    Index(Cell item) : Base(item){}
    operator CellLocalId() const { return CellLocalId{localId()}; }
  };

 private:

  //! Constructeur réservé pour les énumérateurs
  Cell(Int32 local_id,ItemSharedInfo* shared_info)
  : ItemWithNodes(local_id,shared_info) {}

 public:

  //! Type du localId()
  typedef CellLocalId LocalIdType;

  //! Constructeur d'une maille nulle
  Cell() = default;

  //! Construit une référence à l'entité \a internal
  Cell(ItemInternal* ainternal) : ItemWithNodes(ainternal)
  { ARCANE_CHECK_KIND(isCell); }

  //! Construit une référence à l'entité \a abase
  Cell(const ItemBase& abase) : ItemWithNodes(abase)
  { ARCANE_CHECK_KIND(isCell); }

  //! Construit une référence à l'entité \a aitem
  explicit Cell(const Item& aitem) : ItemWithNodes(aitem)
  { ARCANE_CHECK_KIND(isCell); }

  //! Construit une référence à l'entité \a internal
  Cell(const ItemInternalPtr* internals,Int32 local_id)
  : ItemWithNodes(internals,local_id)
  { ARCANE_CHECK_KIND(isCell); }

  //! Opérateur de copie
  Cell& operator=(ItemInternal* ainternal)
  {
    _set(ainternal);
    return (*this);
  }

 public:

  //! Identifiant local de l'entité dans le sous-domaine du processeur
  CellLocalId itemLocalId() const { return CellLocalId{ m_local_id }; }

  //! Nombre de faces de la maille
  Int32 nbFace() const { return _nbFace(); }

  //! i-ème face de la maille
  Face face(Int32 i) const { return _face(i); }

  //! Liste des faces de la maille
  FaceConnectedListViewType faces() const { return _faceList(); }

  //! i-ème face de la maille
  FaceLocalId faceId(Int32 i) const { return _faceId(i); }

  //! Liste des faces de la maille
  FaceLocalIdView faceIds() const { return _faceIds(); }

  //! Nombre d'arêtes de la maille
  Int32 nbEdge() const { return _nbEdge(); }

  //! i-ème arête de la maille
  Edge edge(Int32 i) const { return _edge(i); }

  //! i-ème arête de la maille
  EdgeLocalId edgeId(Int32 i) const { return _edgeId(i); }

  //! Liste des arêtes de la maille
  EdgeConnectedListViewType edges() const { return _edgeList(); }

  //! Liste des arêtes de la maille
  EdgeLocalIdView edgeIds() const { return _edgeIds(); }

  //! AMR
  //! ATT: la notion de parent est utilisé à la fois dans le concept sous-maillages et AMR.
  //! La première implémentation AMR sépare les deux concepts pour des raisons de consistances.
  //! Une fusion des deux notions est envisageable dans un deuxième temps
  //! dans un premier temps, les appelations, pour l'amr, sont en français i.e. parent -> pere et child -> enfant
  //! un seul parent
  Cell hParent() const { return Cell(_hParentBase(0)); }

  //! Nombre de parent pour l'AMR
  Int32 nbHParent() const { return _nbHParent(); }

  //! Nombre d'enfants pour l'AMR
  Int32 nbHChildren() const { return _nbHChildren(); }

  //! i-ème enfant AMR
  Cell hChild(Int32 i) const { return Cell(_hChildBase(i)); }

  //! parent de niveau 0 pour l'AMR
  Cell topHParent() const { return Cell(_toItemBase().topHParentBase()); }

  /*!
   * \returns \p true si l'item est actif (i.e. n'a pas de
   * descendants actifs), \p false  sinon. Notez qu'il suffit de vérifier
   * le premier enfant seulement. Renvoie toujours \p true si l'AMR est désactivé.
   */
  bool isActive() const { return _toItemBase().isActive(); }

  bool isSubactive() const { return _toItemBase().isSubactive(); }

  /*!
   * \returns \p true si l'item est un ancetre (i.e. a un
   * enfant actif ou un enfant ancetre), \p false sinon.
   * Renvoie toujours \p false si l'AMR est désactivé.
   */
  bool isAncestor() const { return _toItemBase().isAncestor(); }

  /*!
   * \returns \p true si l'item a des enfants (actifs ou non),
   * \p false  sinon. Renvoie toujours \p false si l'AMR est désactivé.
   */
  bool hasHChildren() const { return _toItemBase().hasHChildren(); }

  /*!
   * \returns le niveau de raffinement de l'item courant.  Si l'item
   * parent est \p NULL donc par convention il est au niveau 0,
   * sinon il est simplement au niveau superieur que celui de son parent.
   */
  Int32 level() const
  {
    //! si je n'ai pas de parent donc j'ai été crée
    //! directement à partir d'un fichier ou par l'utilisateur,
    //! donc je suis un item de niveau 0
    if (this->_nbHParent() == 0)
      return 0;
    //! sinon je suis au niveau supérieur que celui de mon parent
    return (this->_hParentBase(0).level() + 1);
  }

  /*!
   * \returns le rang de l'enfant \p (iitem).
   * exemple: si rank = m_internal->whichChildAmI(iitem); donc
   * m_internal->hChild(rank) serait iitem;
   */
  Int32 whichChildAmI(const ItemInternal* iitem) const
  {
    return _toItemBase().whichChildAmI(iitem->localId());
  }

  /*!
   * \returns le rang de l'enfant avec \p (iitem).
   * exemple: si rank = m_internal->whichChildAmI(iitem); donc
   * m_internal->hChild(rank) serait iitem;
   */
  Int32 whichChildAmI(CellLocalId local_id) const
  {
    return _toItemBase().whichChildAmI(local_id);
  }

  ARCANE_DEPRECATED_REASON("Y2022: Do not use this operator. Use operator '.' instead")
  Cell* operator->() { return this; }

  ARCANE_DEPRECATED_REASON("Y2022: Do not use this operator. Use operator '.' instead")
  const Cell* operator->() const { return this; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline Cell Item::
_cell(Int32 index) const
{
  return Cell(_connectivity()->cellBase(m_local_id,index));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Particule.
 * \ingroup Mesh
 */
class Particle
: public Item
{
  using ThatClass = Particle;
  // Pour accéder aux constructeurs privés
  friend class ItemEnumeratorBaseT<ThatClass>;
  friend class ItemConnectedEnumeratorBaseT<ThatClass>;
  friend class ItemVectorT<ThatClass>;
  friend class ItemVectorViewT<ThatClass>;
  friend class ItemConnectedListViewT<ThatClass>;
  friend class ItemVectorViewConstIteratorT<ThatClass>;
  friend class ItemConnectedListViewConstIteratorT<ThatClass>;
  friend class SimdItemT<ThatClass>;
  friend class ItemInfoListViewT<ThatClass>;
  friend class ItemLocalIdToItemConverterT<ThatClass>;

 private:

  //! Constructeur réservé pour les énumérateurs
  Particle(Int32 local_id,ItemSharedInfo* shared_info)
  : Item(local_id,shared_info) {}

 public:
  
  //! Type du localId()
  typedef ParticleLocalId LocalIdType;

  //! Constructeur d'une particule nulle
  Particle() = default;

  //! Construit une référence à l'entité \a internal
  Particle(ItemInternal* ainternal) : Item(ainternal)
  { ARCANE_CHECK_KIND(isParticle); }

  //! Construit une référence à l'entité \a abase
  Particle(const ItemBase& abase) : Item(abase)
  { ARCANE_CHECK_KIND(isParticle); }

  //! Construit une référence à l'entité \a aitem
  explicit Particle(const Item& aitem) : Item(aitem)
  { ARCANE_CHECK_KIND(isParticle); }

  //! Construit une référence à l'entité \a internal
  Particle(const ItemInternalPtr* internals,Int32 local_id)
  : Item(internals,local_id)
  { ARCANE_CHECK_KIND(isParticle); }

  //! Opérateur de copie
  Particle& operator=(ItemInternal* ainternal)
  {
    _set(ainternal);
    return (*this);
  }

 public:

  //! Identifiant local de l'entité dans le sous-domaine du processeur
  ParticleLocalId itemLocalId() const { return ParticleLocalId{ m_local_id }; }

  /*!
   * \brief Maille à laquelle appartient la particule.
   * Il faut appeler setCell() avant d'appeler cette fonction.
   * \precondition hasCell() doit être vrai.
   */
  Cell cell() const { return _cell(0); }

  //! Maille connectée à la particule
  CellLocalId cellId() const { return _cellId(0); }

  //! Vrai si la particule est dans une maille du maillage
  bool hasCell() const { return (_cellId(0).localId()!=NULL_ITEM_LOCAL_ID); }

  /*!
   * \brief Maille à laquelle appartient la particule ou maille nulle.
   * Retourne cell() si la particule est dans une maille ou la
   * maille nulle si la particule n'est dans aucune maille.
   */
  Cell cellOrNull() const
  {
    Int32 cell_local_id = _cellId(0).localId();
    if (cell_local_id==NULL_ITEM_LOCAL_ID)
      return Cell();
    return _cell(0);
  }

  ARCANE_DEPRECATED_REASON("Y2022: Do not use this operator. Use operator '.' instead")
  Particle* operator->() { return this; }

  ARCANE_DEPRECATED_REASON("Y2022: Do not use this operator. Use operator '.' instead")
  const Particle* operator->() const { return this; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 *
 * \brief classe degré de liberté.
 *
 * \ingroup Mesh
 *
 * Ce nouvel item DoF introduit une nouvelle gestion de la connectivité, déportée
 * dans des propriétés et non plus stockées dans l'ItemSharedInfo afin de pouvoir créer
 * de nouvelles connectivités en fonction des besoins de l'utilisateur. Par défaut aucune
 * connectivité n'est associée au DoF. Les connectivités nécessaires seront ajoutées par l'utilisateur.
 *
 */
class DoF
: public Item
{
  using ThatClass = DoF;
  // Pour accéder aux constructeurs privés
  friend class ItemEnumeratorBaseT<ThatClass>;
  friend class ItemConnectedEnumeratorBaseT<ThatClass>;
  friend class ItemVectorT<ThatClass>;
  friend class ItemVectorViewT<ThatClass>;
  friend class ItemConnectedListViewT<ThatClass>;
  friend class ItemVectorViewConstIteratorT<ThatClass>;
  friend class ItemConnectedListViewConstIteratorT<ThatClass>;
  friend class SimdItemT<ThatClass>;
  friend class ItemInfoListViewT<ThatClass>;
  friend class ItemLocalIdToItemConverterT<ThatClass>;

 private:

  //! Constructeur réservé pour les énumérateurs
  DoF(Int32 local_id,ItemSharedInfo* shared_info)
  : Item(local_id,shared_info) {}


 public:

  using LocalIdType = DoFLocalId;

  //! Constructeur d'une maille non connectée
  DoF() = default;

  //! Construit une référence à l'entité \a internal
  DoF(ItemInternal* ainternal) : Item(ainternal)
  { ARCANE_CHECK_KIND(isDoF); }

  //! Construit une référence à l'entité \a abase
  DoF(const ItemBase& abase) : Item(abase)
  { ARCANE_CHECK_KIND(isDoF); }

  //! Construit une référence à l'entité \a abase
  explicit DoF(const Item& aitem) : Item(aitem)
  { ARCANE_CHECK_KIND(isDoF); }

  //! Construit une référence à l'entité \a internal
  DoF(const ItemInternalPtr* internals,Int32 local_id)
  : Item(internals,local_id)
  { ARCANE_CHECK_KIND(isDoF); }

  //! Opérateur de copie
  DoF& operator=(ItemInternal* ainternal)
  {
    _set(ainternal);
    return (*this);
  }

  ARCANE_DEPRECATED_REASON("Y2022: Do not use this operator. Use operator '.' instead")
  DoF* operator->() { return this; }

  ARCANE_DEPRECATED_REASON("Y2022: Do not use this operator. Use operator '.' instead")
  const DoF* operator->() const { return this; }

  //! Identifiant local de l'entité dans le sous-domaine du processeur
  DoFLocalId itemLocalId() const { return DoFLocalId{ m_local_id }; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline Edge Node::
edge(Int32 i) const
{
  return _edge(i);
}

inline Face Node::
face(Int32 i) const
{
  return _face(i);
}

inline Cell Node::
cell(Int32 i) const
{
  return _cell(i);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline Face Edge::
face(Int32 i) const
{
  return _face(i);
}

inline Cell Edge::
cell(Int32 i) const
{
  return _cell(i);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline Cell Face::
boundaryCell() const
{
  return Cell(_toItemBase().boundaryCell());
}

inline Cell Face::
backCell() const
{
  return Cell(_toItemBase().backCell());
}

inline Cell Face::
frontCell() const
{
  return Cell(_toItemBase().frontCell());
}

inline Cell Face::
oppositeCell(Cell cell) const
{
  ARCANE_ASSERT((backCell()==cell || frontCell()==cell),("cell is not connected to the face"));
  return (backCell()==cell) ? frontCell() : backCell();
}

inline Cell Face::
cell(Int32 i) const
{
  return _cell(i);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline ItemWithNodes Item::
toItemWithNodes() const
{
  ARCANE_CHECK_KIND(isItemWithNodes);
  return ItemWithNodes(*this);
}

inline Node Item::
toNode() const
{
  ARCANE_CHECK_KIND(isNode);
  return Node(*this);
}

inline Edge Item::
toEdge() const
{
  ARCANE_CHECK_KIND(isEdge);
  return Edge(*this);
}

inline Face Item::
toFace() const
{
  ARCANE_CHECK_KIND(isFace);
  return Face(*this);
}

inline Cell Item::
toCell() const
{
  ARCANE_CHECK_KIND(isCell);
  return Cell(*this);
}

inline Particle Item::
toParticle() const
{
  ARCANE_CHECK_KIND(isParticle);
  return Particle(*this);
}

inline DoF Item::
toDoF() const
{
  ARCANE_CHECK_KIND(isDoF);
  return DoF(*this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline ItemLocalId::
ItemLocalId(Item item)
: m_local_id(item.localId())
{
}

template<typename ItemType> inline ItemLocalIdT<ItemType>::
ItemLocalIdT(ItemType item)
: ItemLocalId(item.localId())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline Item ItemInfoListView::
operator[](ItemLocalId local_id) const
{
  return Item(local_id.localId(), m_item_shared_info);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline Item ItemInfoListView::
operator[](Int32 local_id) const
{
  return Item(local_id, m_item_shared_info);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename ItemType> inline ItemType ItemInfoListViewT<ItemType>::
operator[](ItemLocalId local_id) const
{
  return ItemType(local_id.localId(), m_item_shared_info);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename ItemType> inline ItemType ItemInfoListViewT<ItemType>::
operator[](Int32 local_id) const
{
  return ItemType(local_id, m_item_shared_info);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline constexpr ARCCORE_HOST_DEVICE Item ItemLocalIdToItemConverter::
operator[](ItemLocalId local_id) const
{
  return Item(local_id.localId(), m_item_shared_info);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline constexpr ARCCORE_HOST_DEVICE Item ItemLocalIdToItemConverter::
operator[](Int32 local_id) const
{
  return Item(local_id, m_item_shared_info);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename ItemType_> inline constexpr ARCCORE_HOST_DEVICE ItemType_
ItemLocalIdToItemConverterT<ItemType_>::
operator[](ItemLocalIdType local_id) const
{
  return ItemType(local_id.localId(), m_item_shared_info);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename ItemType_> inline constexpr ARCCORE_HOST_DEVICE ItemType_
ItemLocalIdToItemConverterT<ItemType_>::
operator[](Int32 local_id) const
{
  return ItemType(local_id, m_item_shared_info);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ItemCompatibility.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
