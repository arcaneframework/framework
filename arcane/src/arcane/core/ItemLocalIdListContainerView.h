// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemLocalIdListContainerView.h                              (C) 2000-2023 */
/*                                                                           */
/* Vue sur le conteneur d'une liste de ItemLocalId.                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ITEMLOCALIDLISTCONTAINERVIEW_H
#define ARCANE_CORE_ITEMLOCALIDLISTCONTAINERVIEW_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ItemTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Vue sur le conteneur d'une liste de ItemLocalId.
 *
 * Cette classe sert uniquement à conserver les informations nécessaires
 * pour une liste de 'ItemLocalId' et n'est utilisé que pour passer des
 * informations entre les vues sur les entités (par exemple ItemVectorView)
 * et les itérateurs associés.
 *
 * La classe utilisateur associée est ItemLocalIdListView.
 */
class ARCANE_CORE_EXPORT ItemLocalIdListContainerView
{
  template <typename ItemType> friend class ::Arcane::ItemLocalIdViewT;
  template <int Extent> friend class ::Arcane::impl::ItemIndexedListView;
  friend ItemVectorView;
  friend ItemLocalIdListView;
  friend ItemInternalConnectivityList;
  friend ItemInternalVectorView;
  friend ItemEnumeratorBase;
  friend SimdItemEnumeratorBase;
  friend ItemIndexArrayView;

 private:

  ItemLocalIdListContainerView() = default;
  constexpr ARCCORE_HOST_DEVICE ItemLocalIdListContainerView(const Int32* ids, Int32 s, Int32 local_id_offset)
  : m_local_ids(ids)
  , m_local_id_offset(local_id_offset)
  , m_size(s)
  {}

  constexpr ARCCORE_HOST_DEVICE ItemLocalIdListContainerView(SmallSpan<const Int32> ids, Int32 local_id_offset)
  : m_local_ids(ids.data())
  , m_local_id_offset(local_id_offset)
  , m_size(ids.size())
  {}

 public:

  ARCCORE_HOST_DEVICE Int32 operator[](Int32 index) const
  {
    ARCANE_CHECK_AT(index, m_size);
    return m_local_ids[index] + m_local_id_offset;
  }
  ARCCORE_HOST_DEVICE Int32 localId(Int32 index) const
  {
    ARCANE_CHECK_AT(index, m_size);
    return m_local_ids[index] + m_local_id_offset;
  }
  constexpr ARCCORE_HOST_DEVICE Int32 size() const { return m_size; }

  void fillLocalIds(Array<Int32>& ids) const;

  friend ARCANE_CORE_EXPORT std::ostream&
  operator<<(std::ostream& o, const ItemLocalIdListContainerView& lhs);

 private:

  ConstArrayView<Int32> _idsWithoutOffset() const { return { m_size, m_local_ids }; }

 private:

  const Int32* m_local_ids = nullptr;
  Int32 m_local_id_offset = 0;
  Int32 m_size = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
