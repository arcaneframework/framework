// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemInfoListView.h                                          (C) 2000-2024 */
/*                                                                           */
/* Vue sur une liste pour obtenir des informations sur les entités.          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ITEMINFOLISTVIEW_H
#define ARCANE_ITEMINFOLISTVIEW_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ItemGenericInfoListView.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue sur une liste pour obtenir des informations sur les entités.
 *
 * Comme toutes les vues, ces instances sont temporaires et ne doivent pas être
 * conservées entre deux modifications de la famille associée.
 *
 * Les méthodes de cette classe ne sont valides que si l'instance a été initialisée
 * avec une famille (IItemFamily) non nulle.
 *
 * Via cette classe, il est possible de récupérer une instance de Item à partir
 * d'un numéro local ItemLocalId.
 */
class ARCANE_CORE_EXPORT ItemInfoListView
: public ItemGenericInfoListView
{
  using BaseClass = ItemGenericInfoListView;
  friend class mesh::ItemFamily;
  friend ItemVector;
  friend ItemPairEnumerator;
  friend ItemGenericInfoListView;
  template <int Extent> friend class ItemConnectedListView;
  template <typename ItemType> friend class ItemEnumeratorBaseT;

  // A supprimer lorqu'on n'aura plus besoin de _itemsInternal()
  friend ItemVectorView;

 public:

  ItemInfoListView() = default;

  /*!
   * \brief Construit une vue associée à la famille \a family.
   *
   * \a family peut valoir \a nullptr auquel cas l'instance n'est
   * pas utilisable pour récupérer des informations sur les entités
   */
  explicit ItemInfoListView(IItemFamily* family);

 public:

  //! Famille associée
  IItemFamily* itemFamily() const { return m_item_shared_info->itemFamily(); }

  // NOTE: Les définitions des deux méthodes operator[] sont dans Item.h

  //! Entité associée du numéro local \a local_id
  inline Item operator[](ItemLocalId local_id) const;

  //! Entité associée du numéro local \a local_id
  inline Item operator[](Int32 local_id) const;

 private:

  // Seule ItemFamily peut créer des instances via ce constructeur
  explicit ItemInfoListView(ItemSharedInfo* shared_info)
  : ItemGenericInfoListView(shared_info)
  {}

 protected:

  using BaseClass::m_flags;
  using BaseClass::m_item_shared_info;
  void _checkValid(eItemKind expected_kind);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe de base des vues spécialisées des informations sur les entités.
 */
template <typename ItemType>
class ItemInfoListViewT
: public ItemInfoListView
{
 public:

  ItemInfoListViewT() = default;

  //! Construit une vue associée à la famille \a family.
  explicit ItemInfoListViewT(IItemFamily* family)
  : ItemInfoListView(family)
  {
    _checkValid(ItemTraitsT<ItemType>::kind());
  }

 public:

  // NOTE: Les définitions des deux méthodes operator[] sont dans Item.h

  //! Entité associée du numéro local \a local_id
  inline ItemType operator[](ItemLocalId local_id) const;

  //! Entité associée du numéro local \a local_id
  inline ItemType operator[](Int32 local_id) const;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue sur les informations des noeuds.
 */
class NodeInfoListView
: public ItemInfoListViewT<Node>
{
 public:

  using BaseClass = ItemInfoListViewT<Node>;

 public:

  NodeInfoListView() = default;

  //! Construit une vue associée à la famille \a family.
  explicit NodeInfoListView(IItemFamily* family)
  : BaseClass(family)
  {}
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue sur les informations des arêtes.
 */
class EdgeInfoListView
: public ItemInfoListViewT<Edge>
{
 public:

  using BaseClass = ItemInfoListViewT<Edge>;

 public:

  EdgeInfoListView() = default;

  //! Construit une vue associée à la famille \a family.
  explicit EdgeInfoListView(IItemFamily* family)
  : BaseClass(family)
  {}
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue sur les informations des faces.
 */
class FaceInfoListView
: public ItemInfoListViewT<Face>
{
 public:

  using BaseClass = ItemInfoListViewT<Face>;

 public:

  FaceInfoListView() = default;

  //! Construit une vue associée à la famille \a family.
  explicit FaceInfoListView(IItemFamily* family)
  : BaseClass(family)
  {}

 public:

  constexpr ARCCORE_HOST_DEVICE bool isSubDomainBoundary(FaceLocalId local_id) const
  {
    return ItemFlags::isSubDomainBoundary(m_flags[local_id]);
  }
  constexpr ARCCORE_HOST_DEVICE bool isSubDomainBoundaryOutside(FaceLocalId local_id) const
  {
    return ItemFlags::isSubDomainBoundaryOutside(m_flags[local_id]);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue sur les informations des mailles.
 */
class CellInfoListView
: public ItemInfoListViewT<Cell>
{
 public:

  using BaseClass = ItemInfoListViewT<Cell>;

 public:

  CellInfoListView() = default;

  //! Construit une vue associée à la famille \a family.
  explicit CellInfoListView(IItemFamily* family)
  : BaseClass(family)
  {}
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue sur les informations des particules.
 */
class ParticleInfoListView
: public ItemInfoListViewT<Particle>
{
 public:

  using BaseClass = ItemInfoListViewT<Particle>;

 public:

  ParticleInfoListView() = default;

  //! Construit une vue associée à la famille \a family.
  explicit ParticleInfoListView(IItemFamily* family)
  : BaseClass(family)
  {}
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue sur les informations des DoFs.
 */
class DoFInfoListView
: public ItemInfoListViewT<DoF>
{
 public:

  using BaseClass = ItemInfoListViewT<DoF>;

 public:

  DoFInfoListView() = default;

  //! Construit une vue associée à la famille \a family.
  explicit DoFInfoListView(IItemFamily* family)
  : BaseClass(family)
  {}
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
