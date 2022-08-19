// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemInfoListView.h                                          (C) 2000-2022 */
/*                                                                           */
/* Vue sur une liste pour obtenir des informations sur les entités.          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ITEMINFOLISTVIEW_H
#define ARCANE_ITEMINFOLISTVIEW_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArrayView.h"

#include "arcane/ItemTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{
class ItemFamily;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class ItemSharedInfo;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue sur une liste pour obtenir des informations sur les entités.
 *
 * Comme toutes les vues, ces instances sont temporaires et ne doivent pas être
 * conservées entre deux modifications de la famille associée.
 *
 * Via cette classe, il est possible de récupérer une instance de Item à partir
 * d'un numéro local ItemLocalId.
 */
class ARCANE_CORE_EXPORT ItemInfoListView
{
  friend class mesh::ItemFamily;
  // A supprimer lorqu'on n'aura plus besoin de _itemsInternal()
  friend class ItemVectorView;

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
  IItemFamily* itemFamily() const { return m_family; }

  //! Entité associé du numéro local \a local_id
  inline Item operator[](ItemLocalId local_id) const;

  //! Entité associé du numéro local \a local_id
  inline Item operator[](Int32 local_id) const;

 private:

  // Seule ItemFamily peut créer des instances via ce constructeur
  ItemInfoListView(IItemFamily* family, ItemSharedInfo* shared_info, ItemInternalArrayView items_internal)
  : m_family(family)
  , m_item_shared_info(shared_info)
  , m_item_internal_list(items_internal)
  {}

  ItemInternalArrayView _itemsInternal() const { return m_item_internal_list; }

 private:

  IItemFamily* m_family = nullptr;

 protected:

  ItemSharedInfo* m_item_shared_info = nullptr;
  ItemInternalArrayView m_item_internal_list;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename ItemType>
class ARCANE_CORE_EXPORT ItemInfoListViewT
: public ItemInfoListView
{
 public:

  //! Construit une vue associée à la famille \a family.
  explicit ItemInfoListViewT(IItemFamily* family)
  : ItemInfoListView(family)
  {}

 public:

  //! Entité associé du numéro local \a local_id
  inline ItemType operator[](ItemLocalId local_id) const;

  //! Entité associé du numéro local \a local_id
  inline ItemType operator[](Int32 local_id) const;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_CORE_EXPORT NodeInfoListView
: public ItemInfoListViewT<Node>
{
 public:

  using BaseClass = ItemInfoListViewT<Node>;

 public:

  //! Construit une vue associée à la famille \a family.
  explicit NodeInfoListView(IItemFamily* family)
  : BaseClass(family)
  {}
};

class ARCANE_CORE_EXPORT EdgeInfoListView
: public ItemInfoListViewT<Edge>
{
 public:

  using BaseClass = ItemInfoListViewT<Edge>;

 public:

  //! Construit une vue associée à la famille \a family.
  explicit EdgeInfoListView(IItemFamily* family)
  : BaseClass(family)
  {}
};

class ARCANE_CORE_EXPORT FaceInfoListView
: public ItemInfoListViewT<Face>
{
 public:

  using BaseClass = ItemInfoListViewT<Face>;

 public:

  //! Construit une vue associée à la famille \a family.
  explicit FaceInfoListView(IItemFamily* family)
  : BaseClass(family)
  {}
};

class ARCANE_CORE_EXPORT CellInfoListView
: public ItemInfoListViewT<Cell>
{
 public:

  using BaseClass = ItemInfoListViewT<Cell>;

 public:

  //! Construit une vue associée à la famille \a family.
  explicit CellInfoListView(IItemFamily* family)
  : BaseClass(family)
  {}
};

class ARCANE_CORE_EXPORT ParticleInfoListView
: public ItemInfoListViewT<Particle>
{
 public:

  using BaseClass = ItemInfoListViewT<Particle>;

 public:

  //! Construit une vue associée à la famille \a family.
  explicit ParticleInfoListView(IItemFamily* family)
  : BaseClass(family)
  {}
};

class ARCANE_CORE_EXPORT DoFInfoListView
: public ItemInfoListViewT<DoF>
{
 public:

  using BaseClass = ItemInfoListViewT<DoF>;

 public:

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
