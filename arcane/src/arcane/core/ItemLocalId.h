// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemLocalId.h                                               (C) 2000-2025 */
/*                                                                           */
/* Numéro local d'une entité du maillage.                                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ITEMLOCALID_H
#define ARCANE_CORE_ITEMLOCALID_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArrayView.h"

#include "arcane/core/ItemTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class ItemSharedInfo;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// TODO: rendre obsolète les constructeurs qui prennent un argument
// un ItemEnumerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Mesh
 * \brief Index d'un Item dans une variable.
 */
class ARCANE_CORE_EXPORT ItemLocalId
{
 public:

  ItemLocalId() = default;
  constexpr ARCCORE_HOST_DEVICE explicit ItemLocalId(Int32 id)
  : m_local_id(id)
  {}
  // La définition de ce constructeur est dans ItemInternal.h
  inline ItemLocalId(ItemInternal* item);
  inline ItemLocalId(ItemConnectedEnumerator enumerator);
  template <typename ItemType> inline ItemLocalId(ItemEnumeratorT<ItemType> enumerator);
  template <typename ItemType> inline ItemLocalId(ItemConnectedEnumeratorT<ItemType> enumerator);
  inline ItemLocalId(Item item);
  constexpr ARCCORE_HOST_DEVICE operator Int32() const { return m_local_id; }
  constexpr ARCCORE_HOST_DEVICE Int32 asInt32() const { return m_local_id; }
  constexpr ARCCORE_HOST_DEVICE Int32 asInteger() const { return m_local_id; }

 public:

  constexpr ARCCORE_HOST_DEVICE Int32 localId() const { return m_local_id; }
  constexpr ARCCORE_HOST_DEVICE bool isNull() const { return m_local_id == NULL_ITEM_LOCAL_ID; }

 public:

  static SmallSpan<const ItemLocalId> fromSpanInt32(SmallSpan<const Int32> v)
  {
    auto* ptr = reinterpret_cast<const ItemLocalId*>(v.data());
    return { ptr, v.size() };
  }
  static SmallSpan<const Int32> toSpanInt32(SmallSpan<const ItemLocalId> v)
  {
    auto* ptr = reinterpret_cast<const Int32*>(v.data());
    return { ptr, v.size() };
  }

 private:

  Int32 m_local_id = NULL_ITEM_LOCAL_ID;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Mesh
 * \brief Index d'une entité \a ItemType dans une variable.
 */
template <typename ItemType_>
class ItemLocalIdT
: public ItemLocalId
{
 public:

  using ItemType = ItemType_;
  using ThatClass = ItemLocalIdT<ItemType>;

 public:

  ItemLocalIdT() = default;
  constexpr ARCCORE_HOST_DEVICE explicit ItemLocalIdT(Int32 id)
  : ItemLocalId(id)
  {}
  inline ItemLocalIdT(ItemInternal* item);
  inline ItemLocalIdT(ItemConnectedEnumeratorT<ItemType> enumerator);
  inline ItemLocalIdT(ItemType item);

 public:

  static SmallSpan<const ItemLocalId> fromSpanInt32(SmallSpan<const Int32> v)
  {
    auto* ptr = reinterpret_cast<const ThatClass*>(v.data());
    return { ptr, v.size() };
  }

  static SmallSpan<const Int32> toSpanInt32(SmallSpan<const ThatClass> v)
  {
    auto* ptr = reinterpret_cast<const Int32*>(v.data());
    return { ptr, v.size() };
  }

 public:

  ARCANE_DEPRECATED_REASON("Y2022: Use strongly typed 'ItemEnumeratorT<ItemType>' or 'ItemType'")
  inline ItemLocalIdT(ItemEnumerator enumerator);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe pour convertir un ItemLocalId vers une entité (Item).
 *
 * Les instances de cette classe restent valides durant toute la durée
 * de vie de la famille associée.
 */
class ARCANE_CORE_EXPORT ItemLocalIdToItemConverter
{
  template <typename ItemType_> friend class ItemLocalIdToItemConverterT;

 public:

  explicit ItemLocalIdToItemConverter(IItemFamily* family);
  /*!
   * \brief Constructeur par défaut.
   *
   * L'instance ne sera pas valide tant qu'elle n'aura pas été recopiée
   * depuis une instance valide (en utilisant le constructeur qui
   * prend un IItemFamily en argument.
   */
  ItemLocalIdToItemConverter() = default;

 public:

  //! Entité de numéro local \a local_id
  inline constexpr ARCCORE_HOST_DEVICE Item operator[](ItemLocalId local_id) const;
  //! Entité de numéro local \a local_id
  inline constexpr ARCCORE_HOST_DEVICE Item operator[](Int32 local_id) const;

 private:

  ItemSharedInfo* m_item_shared_info = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe pour convertir un ItemLocalId vers une entité (Item).
 *
 * Les instances de cette classe restent valides durant toute la durée
 * de vie de la famille associée.
 */
template <typename ItemType_> class ItemLocalIdToItemConverterT
: public ItemLocalIdToItemConverter
{
 public:

  using ItemType = ItemType_;
  using ItemLocalIdType = ItemLocalIdT<ItemType>;

 public:

  using ItemLocalIdToItemConverter::ItemLocalIdToItemConverter;

 public:

  //! Entité de numéro local \a local_id
  inline constexpr ARCCORE_HOST_DEVICE ItemType operator[](ItemLocalIdType local_id) const;
  //! Entité de numéro local \a local_id
  inline constexpr ARCCORE_HOST_DEVICE ItemType operator[](Int32 local_id) const;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Classe pour convertir un NodeLocalId vers une arête.
class NodeLocalIdToNodeConverter
: public ItemLocalIdToItemConverterT<Node>
{
 public:

  using ItemLocalIdToItemConverterT<Node>::ItemLocalIdToItemConverterT;
};

//! Classe pour convertir un EdgeLocalId vers une arête.
class EdgeLocalIdToEdgeConverter
: public ItemLocalIdToItemConverterT<Edge>
{
 public:

  using ItemLocalIdToItemConverterT<Edge>::ItemLocalIdToItemConverterT;
};

//! Classe pour convertir un FaceLocalId vers une face.
class FaceLocalIdToFaceConverter
: public ItemLocalIdToItemConverterT<Face>
{
 public:

  using ItemLocalIdToItemConverterT<Face>::ItemLocalIdToItemConverterT;
};

//! Classe pour convertir un CellLocalId vers une maille.
class CellLocalIdToCellConverter
: public ItemLocalIdToItemConverterT<Cell>
{
 public:

  using ItemLocalIdToItemConverterT<Cell>::ItemLocalIdToItemConverterT;
};

//! Classe pour convertir un ParticleLocalId vers une particule.
class ParticleLocalIdToParticleConverter
: public ItemLocalIdToItemConverterT<Particle>
{
 public:

  using ItemLocalIdToItemConverterT<Particle>::ItemLocalIdToItemConverterT;
};

//! Classe pour convertir un DoFLocalId vers un degré de liberté.
class DoFLocalIdToDoFConverter
: public ItemLocalIdToItemConverterT<DoF>
{
 public:

  using ItemLocalIdToItemConverterT<DoF>::ItemLocalIdToItemConverterT;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
