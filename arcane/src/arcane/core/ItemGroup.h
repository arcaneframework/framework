// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemGroup.h                                                 (C) 2000-2025 */
/*                                                                           */
/* Groupes d'entités du maillage.                                            */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ITEMGROUP_H
#define ARCANE_ITEMGROUP_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/AutoRef.h"
#include "arcane/utils/Iterator.h"

#include "arcane/core/ItemGroupImpl.h"
#include "arcane/core/ItemTypes.h"
#include "arcane/core/ItemEnumerator.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ItemVectorView;
class IVariableSynchronizer;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
//NOTE: la documentation plus complète est dans ItemGroup.cc
/*!
 * \ingroup Mesh
 * \brief Groupe d'entités de maillage.
 *
 * Un groupe d'entité contient un ensemble d'entité d'une famille
 * donnée. Un groupe se créé via la famille correspondante par
 * IItemFamily::createGroup(), IItemFamily::findGroup().
 *
 */
class ARCANE_CORE_EXPORT ItemGroup
{
 public:
	
  //! Construit un groupe nul.
  ItemGroup();
  //! Construit un groupe à partir de la représentation interne \a prv
  // TODO: ce contructeur devrait être explicite pour éviter des conversions
  // implicites mais on le met pas encore pour des raisons de compatibilité
  /*explicit*/ ItemGroup(ItemGroupImpl* prv);
  //! Construit une référence au groupe \a from.
  ItemGroup(const ItemGroup& from) : m_impl(from.m_impl) {}

  //! Affecte à cette instance une référence au groupe \a from.
  ItemGroup& operator=(const ItemGroup& from) = default;

  //! Type de l'intervalle d'itération (à supprimer)
  typedef ItemEnumerator const_iter;

 public:

  //! \a true is le groupe est le groupe nul
  inline bool null() const
  {
    return m_impl->null();
  }

  //! Nom du groupe
  inline const String& name() const
  {
    return m_impl->name();
  }

  //! Nom du groupe
  inline const String& fullName() const
  {
    return m_impl->fullName();
  }
		
  //! Nombre d'éléments du groupe
  inline Integer size() const
  {
    m_impl->_checkNeedUpdateNoPadding();
    return m_impl->size();
  }

  /*!
   * \brief Teste si le groupe est vide.
   *
   * Un groupe est vide s'il est nul (null() retourne \c true)
   * ou s'il n'a pas d'éléments (size() retourne \c 0).
   * \retval true si le groupe est vide,
   * \retval false sinon.
   */
  inline bool empty() const
  {
    m_impl->_checkNeedUpdateNoPadding();
    return m_impl->empty();
  }

  //! Genre du groupe. Il s'agit du genre de ses éléments
  inline eItemKind itemKind() const { return m_impl->itemKind(); }

 public:

  /*!
   * \internal
   * \brief Retourne l'implémentation du groupe.
   * \warning Cette méthode retourne un pointeur sur la représentation
   * interne du groupe et ne doit pas être utilisée
   * en dehors d'Arcane.
   */
  ItemGroupImpl* internal() const { return m_impl.get(); }

  //! Famille d'entité à laquelle appartient ce groupe (0 pour le group nul)
  IItemFamily* itemFamily() const { return m_impl->itemFamily(); }

  //! Maillage auquel appartient ce groupe (0 pour le group nul)
  IMesh* mesh() const { return m_impl->mesh(); }

 public:

  // Items in the group owned by the subdomain
  ItemGroup own() const;

  // Items in the group not owned by the subdomain
  ItemGroup ghost() const;

  //! Retourne si le groupe contient uniquement des éléments propres au sous-domaine
  bool isOwn() const;

  //! Positionne la propriété de groupe local ou non.
  void setOwn(bool v);

  // Items in the group lying on the boundary between two subdomains
  // Implemented for faces only
  ItemGroup interface() const;

  //! Groupe des noeuds des éléments de ce groupe
  NodeGroup nodeGroup() const;

  //! Groupe des arêtes des éléments de ce groupe
  EdgeGroup edgeGroup() const;

  //! Groupe des faces des éléments de ce groupe
  FaceGroup faceGroup() const;

  //! Groupe des mailles des éléments de ce groupe
  CellGroup cellGroup() const;

  /*!
   * \brief Groupe des faces internes des éléments de ce groupe
   *
   * Ce groupe n'existe que pour un groupe de maille (itemKind()==IK_Cell).
   * Une face est interne si elle connectée à deux mailles de ce groupe.
   */
  FaceGroup innerFaceGroup() const;

  /*!
   * \brief Groupe des faces externes des éléments de ce groupe
   *
   * Ce groupe n'existe que pour un groupe de maille (itemKind()==IK_Cell).
   * Une face est externe si elle n'est connectée qu'à une maille de ce groupe.
   */
  FaceGroup outerFaceGroup() const;

  //! AMR
  //! Groupe des mailles actives des éléments de ce groupe
  CellGroup activeCellGroup() const;

  //! Groupe des mailles propres actives des éléments de ce groupe
  CellGroup ownActiveCellGroup() const;

  //! Groupe des mailles de niveau l des éléments de ce groupe
  CellGroup levelCellGroup(const Integer& level) const;

  //! Groupe des mailles propres de niveau l des éléments de ce groupe
  CellGroup ownLevelCellGroup(const Integer& level) const;

  /*!
   *  \brief Groupe des faces actives
   *
   * Ce groupe n'existe que pour un groupe de maille (itemKind()==IK_Cell).
   */
  FaceGroup activeFaceGroup() const;

  /*!
   * \brief Groupe des faces actives propres au domaine des éléments de ce groupe
   *
   * Ce groupe n'existe que pour un groupe de maille (itemKind()==IK_Cell).
   */
  FaceGroup ownActiveFaceGroup() const;

  /*!
   * \brief Groupe des faces internes des éléments de ce groupe
   *
   * Ce groupe n'existe que pour un groupe de maille (itemKind()==IK_Cell).
   * Une face est interne si elle connectée à deux mailles actives de ce groupe.
   */
  FaceGroup innerActiveFaceGroup() const;

  /*!
   * \brief Groupe des faces externes actives des éléments de ce groupe
   *
   * Ce groupe n'existe que pour un groupe de maille (itemKind()==IK_Cell).
   * Une face est externe si elle n'est connectée qu'à une maille de ce groupe et est active.
   */
  FaceGroup outerActiveFaceGroup() const;

  //! Crée un sous-groupe calculé
  /*! Le gestion mémoire du functor est alors délégué au groupe */
  ItemGroup createSubGroup(const String& suffix, IItemFamily* family, ItemGroupComputeFunctor* functor) const;

  //! Accès à un sous-groupe
  ItemGroup findSubGroup(const String& suffix) const;

  //! Vrai si le groupe est local au sous-domaine
  bool isLocalToSubDomain() const
  {
    return m_impl->isLocalToSubDomain();
  }

  /*! \brief Positionne le booléen indiquant si le groupe est local au sous-domaine.
    
    Par défaut lors de sa création, un groupe est commun à tous les sous-domaines,
    ce qui signifie que chaque sous-domaine doit posséder une instance de ce groupe,
    même si cette instance est vide.
    
    Un groupe local au sous-domaine n'est pas transféré lors d'un rééquilibrage.
  */
  void setLocalToSubDomain(bool v)
  {
    m_impl->setLocalToSubDomain(v);
  }

  /*! \brief Invalide le groupe.
    
    Pour un groupe calculé dynamiquement (comme le groupe des entités propres
    au sous-domaine), cela signifie qu'il doit se recalculer.
    
    Si \a force_recompute est faux, le groupe est juste invalidé et sera
    recalculé la première fois qu'on y accédera. Sinon, il est immédiatement
    recalculé.
  */
  void invalidate(bool force_recompute=false) { m_impl->invalidate(force_recompute); }

  //! Ajoute des entités.
  void addItems(Int32ConstArrayView items_local_id,bool check_if_present=true);

  //! Supprime des entités.
  void removeItems(Int32ConstArrayView items_local_id,bool check_if_present=true);

  //! Positionne les entités du groupe.
  void setItems(Int32ConstArrayView items_local_id);

  /*!
   * \brief Positionne les entités du groupe.
   *
   * Si \a do_sort est vrai, les entités sont triées par uniqueId croissant.
   */
  void setItems(Int32ConstArrayView items_local_id,bool do_sort);

  //! Vérification interne de la validité du groupe
  void checkValid();

  //! Supprime les entités du groupe
  void clear();

  //! Applique l'opération \a operation sur les entités du groupe.
  void applyOperation(IItemOperationByBasicType* operation) const;

  //! Vue sur les entités du groupe.
  ItemVectorView view() const;

  //! Indique si le groupe est celui de toutes les entités
  bool isAllItems() const;

  /*!
   * Retourne le temps de dernière modification du groupe.
   *
   * Ce temps est incrémenté automatiquement après chaque modification.
   * Il est possible de l'incrémenter manuellement via l'appel
   * à incrementTimestamp().
   */
  Int64 timestamp() const
  {
    return m_impl->timestamp();
  }
  
  /*!
   * \brief Incrément le temps de dernière modification du groupe.
   *
   * Normalement ce temps est incrémenté automatiquemnt. Il est néanmmoins
   * possible de le faire manuellement en cas de modification externe des
   * informations du groupe.
   */
  void incrementTimestamp() const;

  //! Table des local ids vers une position pour toutes les entités du groupe
  SharedPtrT<GroupIndexTable> localIdToIndex() const
  {
    return m_impl->localIdToIndex();
  }
 
  //! Synchronizer du groupe
  IVariableSynchronizer* synchronizer() const;

  //! Vrai s'il s'agit d'un groupe calculé automatiquement.
  bool isAutoComputed() const;

  //! Indique si le groupe possède un synchroniser actif
  bool hasSynchronizer() const;

  //! Vérifie et retourne si le groupe est trié par uniqueId() croissants.
  bool checkIsSorted() const;

  //! Vue sur les entités du groupe avec padding pour la vectorisation
  ItemVectorView _paddedView() const;

  /*!
   * \brief Vue sur les entités du groupe sans padding pour la vectorisation.
   *
   * La vue retournée NE doit PAS être utilisée dans les macros de vectorisation
   * telles ENUMERATE_SIMD_CELL().
   */
  ItemVectorView _unpaddedView() const;

 public:

  //! API interne à Arcane
 ItemGroupImplInternal* _internalApi() const;

 public:

  //! Enumérateur sur les entités du groupe.
  ItemEnumerator enumerator() const;

 private:

  template <typename T>
  friend class SimdItemEnumeratorContainerTraits;
  //! Enumérateur sur les entités du groupe pour la vectorisation
  ItemEnumerator _simdEnumerator() const;

 protected:

  //! Représentation interne du groupe.
  AutoRefT<ItemGroupImpl> m_impl;

 protected:

  //! Retourne le groupe \a impl s'il est du genre \a kt, le groupe nul sinon
  static ItemGroupImpl* _check(ItemGroupImpl* impl,eItemKind ik)
  {
    return impl->itemKind()==ik ? impl : ItemGroupImpl::checkSharedNull();
  }

  ItemVectorView _view(bool do_padding) const;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Compare les références de deux groupes.
 * \retval true si \a g1 et \a g2 réfèrent au même groupe,
 * \retval false sinon.
 */
inline bool
operator==(const ItemGroup& g1,const ItemGroup& g2)
{
  return g1.internal()==g2.internal();
}

/*!
 * \brief Compare deux groupes.
 * L'ordre utilisé est quelconque et ne sert que pour faire un tri
 * éventuel pour les containers de la STL.
 * \retval true si \a g1 est inférieur à \a g2,
 * \retval false sinon.
 */
inline bool
operator<(const ItemGroup& g1,const ItemGroup& g2)
{
  return g1.internal()<g2.internal();
}

/*!
 * \brief Compare les références de deux groupes.
 * \retval true si \a g1 et \a g2 ne réfèrent pas au même groupe,
 * \retval false sinon.
 */
inline bool
operator!=(const ItemGroup& g1,const ItemGroup& g2)
{
  return g1.internal()!=g2.internal();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Référence à un groupe d'un genre donné.
 */
template<typename T>
class ItemGroupT
: public ItemGroup
{
 public:

  //! Type de cette classe
  typedef ItemGroupT<T> ThatClass;
  //! Type de la classe contenant les caractéristiques de l'entité
  typedef ItemTraitsT<T> TraitsType;

  typedef typename TraitsType::ItemType ItemType;

  typedef const ItemType* const_iterator;
  typedef ItemType* iterator;
  typedef ItemType value_type;
  typedef const ItemType& const_reference;

 public:

  inline ItemGroupT() = default;
  inline explicit ItemGroupT(ItemGroupImpl* from)
  : ItemGroup(_check(from,TraitsType::kind())){}
  inline ItemGroupT(const ItemGroup& from)
  : ItemGroup(_check(from.internal(),TraitsType::kind())) {}
  inline ItemGroupT(const ItemGroupT<T>& from)
  : ItemGroup(from) {}
  inline const ItemGroupT<T>& operator=(const ItemGroupT<T>& from)
  { m_impl = from.internal(); return (*this); }
  inline const ItemGroupT<T>& operator=(const ItemGroup& from)
  { _assign(from); return (*this); }

 public:
  
  ThatClass own() const
  {
    return ThatClass(ItemGroup::own());
  }

  ItemEnumeratorT<T> enumerator() const
  {
    return ItemEnumeratorT<T>::fromItemEnumerator(ItemGroup::enumerator());
  }

 protected:

  void _assign(const ItemGroup& from)
  {
    m_impl = _check(from.internal(),TraitsType::kind());
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
