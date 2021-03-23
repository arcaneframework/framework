// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemVector.h                                                (C) 2000-2018 */
/*                                                                           */
/* Vue sur un vecteur (tableau indirect) d'entités.                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ITEMVECTOR_H
#define ARCANE_ITEMVECTOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ItemEnumerator.h"
#include "arcane/ItemVectorView.h"
#include "arcane/IItemFamily.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vecteur d'entités.
 *
 * Le fonctionnement du vecteur d'entité est similaire à celui
 * du groupe d'entité ItemGroup à la différence suivante:
 * - le vecteur est local au sous-domaine
 * - le vecteur est invalidé si la famille (IItemFamily) associée évolue
 * (après appel à IItemFamily::endUpdate() si le compactage ou le tri est
 * actif)
 * - le vecteur peut contenir plusieurs fois la même entité.
 *
 * Un vecteur est intéressant pour construire une liste temporaire
 * d'entités. Il reprend les fonctionnalités similaires à la classe
 * Array et il est donc possible par exemple d'ajouter des éléments un par un,
 * soit via un localId(), soit via une entité.
 *
 * Un vecteur doit nécessairement être associée à une famille (setFamily())
 * avant d'être utilisé.
 */
class ItemVector
{
 public:

  typedef Item ItemType;

 public:

  //! Créé un vecteur vide associé à la famille \a family.
  ItemVector(IItemFamily* afamily)
  : m_items(afamily->itemsInternal()), m_family(afamily) {}
  //! Créé un vecteur associé à la famille \a family et contenant les entités \a local_ids.
  ItemVector(IItemFamily* afamily,Int32ConstArrayView local_ids)
  : m_items(afamily->itemsInternal()), m_local_ids(local_ids), m_family(afamily) {}
  /*!
   * \brief Constructeur par référence sur ItemInternals et liste de localId.
   *
   * Ce constructeur est obsolète et l'argument do_clone n'est plus utilisé. Les
   * entités de \a lids sont toujours copiées.
   *
   * \deprecated Utiliser ItemVector(IItemFamily*,Int32ConstArrayView) à la place.
   */
  ARCANE_DEPRECATED_280 ItemVector(IItemFamily* afamily, const SharedArray<Int32>& lids, bool do_clone)
  : m_items(afamily->itemsInternal()), m_local_ids(lids.clone()), m_family(afamily)
  {
    ARCANE_UNUSED(do_clone);
  }

  //! Créé un vecteur pour \a size éléments associé à la famille \a family.
  ItemVector(IItemFamily* afamily,Integer asize)
  : m_items(afamily->itemsInternal()), m_family(afamily)
  {
    m_local_ids.resize(asize);
  }

  //! Operateur de cast vers ItemVectorView
  operator ItemVectorView() const { return ItemVectorView(m_items,m_local_ids); }

  //! Créé un vecteur nul. Il faudra ensuite appeler setFamily() pour l'utiliser
  ItemVector() : m_family(nullptr) { }

 public:

  /*!
   * \brief Positionne la famille associée
   *
   * Le vecteur est vidé de ses éléments
   */
  void setFamily(IItemFamily* afamily)
  {
    m_local_ids.clear();
    m_family = afamily;
    m_items = afamily->itemsInternal();
  }

  //! Ajoute une entité de numéro local \a local_id à la fin du vecteur
  void add(Int32 local_id)
  {
    m_local_ids.add(local_id);
  }

  //! Ajoute une liste d'entité de numéros locaux \a local_ids à la fin du vecteur
  void add(Int32ConstArrayView local_ids)
  {
    m_local_ids.addRange(local_ids);
  }

  //! Ajoute une entité à la fin du vecteur
  void addItem(Item item)
  {
    m_local_ids.add(item.localId());
  }
  //! Nombre d'éléments du vecteur
  Integer size() const
  {
    return m_local_ids.size();
  }

  //! Réserve la mémoire pour \a capacity entités
  void reserve(Integer capacity)
  {
    m_local_ids.reserve(capacity);
  }

  //! Supprime toutes les entités du vecteur.
  void clear()
  {
    m_local_ids.clear();
  }

  //! Vue sur le tableau entier
  ItemVectorView view()
  {
    return ItemVectorView(m_items,m_local_ids);
  }

  //! Vue sur le tableau entier
  ItemVectorView view() const
  {
    return ItemVectorView(m_items,m_local_ids);
  }

  Int32ArrayView viewAsArray()
  {
    return m_local_ids.view();
  }

  Int32ConstArrayView viewAsArray() const
  {
    return m_local_ids.constView();
  }

  //! Conteneur de localIds
  /*! Peut etre directement utilisé pour construire un ItemGroup 
   *  Redondance avec viewAsArray car fusion impl cea et ifp
   */
  ARCANE_DEPRECATED Int32ConstArrayView localIds() const 
  { 
    return m_local_ids; 
  }

  /*!
   * \brief Supprime l'entité à l'index \a index.
   * \deprecated Utiliser removeAt() à la place.
   */
  ARCANE_DEPRECATED_240 void remoteAt(Int32 index)
  {
    m_local_ids.remove(index);
  }

  //! Supprime l'entité à l'index \a index
  void removeAt(Int32 index)
  {
    m_local_ids.remove(index);
  }

  /*!
   * \brief Positionne le nombre d'éléments du tableau.
   *
   * Si la nouvelle taille est supérieure à l'ancienne, les
   * éléments ajoutés sont indéfinis.
   */
  void resize(Integer new_size)
  {
    m_local_ids.resize(new_size);
  }

  //! Clone ce vecteur
  ItemVector clone()
  {
    return ItemVector(m_family,m_local_ids.constView());
  }

  //! Entité à la position \a index du vecteur
  Item operator[](Integer index) const
  {
    return m_items[m_local_ids[index]];
  }

  //! Famille associée au vecteur
  IItemFamily* family() const
  {
    return m_family;
  }

  //! Enumérateur
  inline ItemEnumerator enumerator() const
  {
    return ItemEnumerator(m_items.unguardedBasePointer(),m_local_ids.unguardedBasePointer(),
                          m_local_ids.size());
  }

 protected:

  ItemInternalList m_items;
  SharedArray<Int32> m_local_ids;
  IItemFamily* m_family;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vecteur typé d'entité.
 *
 * Pour plus d'infos, voir ItemVector.
 */
template<typename VectorItemType>
class ItemVectorT
: public ItemVector
{
 public:

  typedef VectorItemType ItemType;

 public:

  //! Constructeur vide
  ItemVectorT()
  : ItemVector() {}
  //! Constructeur vide avec famille
  ItemVectorT(IItemFamily* afamily)
  : ItemVector(afamily) {}
  //! Constructeur par copie
  ItemVectorT(const ItemVectorT<ItemType>& rhs)
  : ItemVector(rhs) {}
  //! Créé un vecteur associé à la famille \a afamily et contenant les entités \a local_ids.
  ItemVectorT(IItemFamily* afamily,Int32ConstArrayView local_ids)
  : ItemVector(afamily, local_ids) {}
  /*!
   * \brief Constructeur par référence sur ItemInternals et liste de localId.
   *
   * Ce constructeur est obsolète et l'argument do_clone n'est plus utilisé. Les
   * entités de \a lids sont toujours copiées.
   *
   * \deprecated Utiliser ItemVectorT(IItemFamily*,Int32ConstArrayView) à la place.
   */
  ARCANE_DEPRECATED_280 ItemVectorT(IItemFamily* afamily, const SharedArray<Int32>& lids, bool do_clone)
  : ItemVector(afamily,lids,do_clone){}
  //! Constructeur par copie
  ItemVectorT(const ItemVector &rhs)
  : ItemVector(rhs) {}
  //! Constructeur pour \a asize élément pour la familly \a afamily
  ItemVectorT(IItemFamily* afamily,Integer asize)
  : ItemVector(afamily,asize) {}

 public:

  //! Operateur de cast vers ItemVectorView
  operator ItemVectorViewT<VectorItemType>() const { return ItemVectorViewT<VectorItemType>(m_items,m_local_ids); }

 public:

  //! Entité à la position \a index du vecteur
  ItemType operator[](Integer index) const
  {
    return m_items[m_local_ids[index]];
    // return ItemType(m_items.begin(),m_local_ids[index]); // version IFP
  }

  //! Ajoute une entité à la fin du vecteur
  void addItem(ItemType item)
  {
    m_local_ids.add(item.localId());
  }

  //! Vue sur le tableau entier
  ItemVectorViewT<ItemType> view()
  {
    return ItemVectorViewT<ItemType>(m_items,m_local_ids);
  }

  //! Vue sur le tableau entier
  ItemVectorViewT<ItemType> view() const
  {
    return ItemVectorViewT<ItemType>(m_items,m_local_ids);
  }

  //! Enumérateur
  ItemEnumeratorT<ItemType>	enumerator() const
  {
    return ItemEnumeratorT<ItemType>(m_items.data(),m_local_ids.data(),
                                     m_local_ids.size());
  }

 public:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename ItemType> inline ItemVectorViewT<ItemType>::
ItemVectorViewT(const ItemVectorT<ItemType>& rhs)
: ItemVectorView(rhs.view())
{  
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
