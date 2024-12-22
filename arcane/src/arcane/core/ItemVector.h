// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemVector.h                                                (C) 2000-2024 */
/*                                                                           */
/* Vecteur d'entités de même genre.                                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ITEMVECTOR_H
#define ARCANE_CORE_ITEMVECTOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ItemEnumerator.h"
#include "arcane/core/ItemVectorView.h"
#include "arcane/core/IItemFamily.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vecteur d'entités.
 *
 * La classe ItemVector utilise une sémantique par référence.
 *
 * \note Cette classe n'est pas thread-safe et ne doit pas être utilisée par
 * différents threads en même temps.
 *
 * \warning Un vecteur doit nécessairement être associée à une famille d'entité
 * (ItemFamily*) avant d'être utilisé. Il est possible de le faire soit via
 * l'appel à setFamily(), soit via un constructeur qui prend une famille
 * en argument.
 *
 * \a ItemVector est la classe générique. Il est possible d'avoir une
 * version spécialisée par genre d'entité via ItemVectorT.
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
 */
class ARCANE_CORE_EXPORT ItemVector
{
 public:

  using ItemType = Item;

 public:

  //! Créé un vecteur vide associé à la famille \a family.
  explicit ItemVector(IItemFamily* afamily);

  //! Créé un vecteur associé à la famille \a family et contenant les entités \a local_ids.
  ItemVector(IItemFamily* afamily, Int32ConstArrayView local_ids);

  //! Créé un vecteur pour \a size éléments associé à la famille \a family.
  ItemVector(IItemFamily* afamily, Integer asize);

  //! Créé un vecteur nul. Il faudra ensuite appeler setFamily() pour l'utiliser
  ItemVector();

 public:

  //! Operateur de cast vers ItemVectorView
  operator ItemVectorView() const { return view(); }

 public:

  /*!
   * \brief Positionne la famille associée
   *
   * Le vecteur est vidé de ses éléments
   */
  void setFamily(IItemFamily* afamily);

  //! Ajoute une entité de numéro local \a local_id à la fin du vecteur
  void add(Int32 local_id) { m_local_ids.add(local_id); }

  //! Ajoute une liste d'entité de numéros locaux \a local_ids à la fin du vecteur
  void add(ConstArrayView<Int32> local_ids) { m_local_ids.addRange(local_ids); }

  //! Ajoute une entité de numéro local \a local_id à la fin du vecteur
  void addItem(ItemLocalId local_id) { m_local_ids.add(local_id); }

  //! Ajoute une entité à la fin du vecteur
  void addItem(Item item) { m_local_ids.add(item.localId()); }

  //! Nombre d'éléments du vecteur
  Int32 size() const { return m_local_ids.size(); }

  //! Réserve la mémoire pour \a capacity entités
  void reserve(Integer capacity) { m_local_ids.reserve(capacity); }

  //! Supprime toutes les entités du vecteur.
  void clear() { m_local_ids.clear(); }

  //! Vue sur le vecteur
  ItemVectorView view() const { return ItemVectorView(m_shared_info, m_local_ids, 0); }

  //! Vue sur les numéros locaux
  ArrayView<Int32> viewAsArray() { return m_local_ids.view(); }

  //! Vue constante sur les numéros locaux
  ConstArrayView<Int32> viewAsArray() const { return m_local_ids.constView(); }

  //! Supprime l'entité à l'index \a index
  void removeAt(Int32 index) { m_local_ids.remove(index); }

  /*!
   * \brief Positionne le nombre d'éléments du tableau.
   *
   * Si la nouvelle taille est supérieure à l'ancienne, les
   * éléments ajoutés sont indéfinis.
   */
  void resize(Integer new_size) { m_local_ids.resize(new_size); }

  //! Clone ce vecteur
  ItemVector clone() { return ItemVector(m_family, m_local_ids.constView()); }

  //! Entité à la position \a index du vecteur
  Item operator[](Int32 index) const { return Item(m_local_ids[index], m_shared_info); }

  //! Famille associée au vecteur
  IItemFamily* family() const { return m_family; }

  //! Enumérateur
  ItemEnumerator enumerator() const { return ItemEnumerator(m_shared_info, m_local_ids); }

 protected:

  SharedArray<Int32> m_local_ids;
  IItemFamily* m_family = nullptr;
  ItemSharedInfo* m_shared_info = ItemSharedInfo::nullInstance();

 private:

  void _init();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vecteur typé d'entité.
 *
 * Pour plus d'infos, voir ItemVector.
 */
template <typename VectorItemType>
class ItemVectorT
: public ItemVector
{
 public:

  using ItemType = VectorItemType;

 public:

  //! Constructeur vide
  ItemVectorT() = default;

  //! Constructeur vide avec famille
  explicit ItemVectorT(IItemFamily* afamily)
  : ItemVector(afamily)
  {}

  //! Créé un vecteur associé à la famille \a afamily et contenant les entités \a local_ids.
  ItemVectorT(IItemFamily* afamily, ConstArrayView<Int32> local_ids)
  : ItemVector(afamily, local_ids)
  {}

  //! Constructeur par copie
  ItemVectorT(const ItemVector& rhs)
  : ItemVector(rhs)
  {}

  //! Constructeur pour \a asize élément pour la familly \a afamily
  ItemVectorT(IItemFamily* afamily, Integer asize)
  : ItemVector(afamily, asize)
  {}

 public:

  //! Operateur de cast vers ItemVectorView
  operator ItemVectorViewT<VectorItemType>() const { return view(); }

 public:

  //! Entité à la position \a index du vecteur
  ItemType operator[](Int32 index) const
  {
    return ItemType(m_local_ids[index], m_shared_info);
  }

  //! Ajoute une entité à la fin du vecteur
  void addItem(ItemType item) { m_local_ids.add(item.localId()); }

  //! Ajoute une entité à la fin du vecteur
  void addItem(ItemLocalIdT<ItemType> local_id) { m_local_ids.add(local_id); }

  //! Vue sur le tableau entier
  ItemVectorViewT<ItemType> view() const
  {
    return ItemVectorViewT<ItemType>(m_shared_info, m_local_ids.constView(), 0);
  }

  //! Enumérateur
  ItemEnumeratorT<ItemType> enumerator() const
  {
    return ItemEnumeratorT<ItemType>(m_shared_info, m_local_ids);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename ItemType> inline ItemVectorViewT<ItemType>::
ItemVectorViewT(const ItemVectorT<ItemType>& rhs)
: ItemVectorView(rhs.view())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
