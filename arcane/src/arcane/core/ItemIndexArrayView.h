// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemIndexArrayView.h                                        (C) 2000-2025 */
/*                                                                           */
/* Vue sur un tableau d'index (localIds()) d'entités.                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ITEMINDEXARRAYVIEW_H
#define ARCANE_CORE_ITEMINDEXARRAYVIEW_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArrayView.h"
#include "arcane/utils/UtilsTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Vue sur un tableau d'index (localIds()) d'entités.
 *
 * \warning La vue n'est valide que tant que le tableau associé n'est
 * pas modifié. Les instances de cette classe sont en général temporaires
 * et ne doivent pas être conservées.
 *
 * En plus de la liste des entités, cette classe permet d'avoir des
 * informations supplémentaires, comme par exemple si la liste est contigüe.
 */
class ARCANE_CORE_EXPORT ItemIndexArrayView
{
  // NOTE: Cette classe est mappée en C# et si on change sa structure il
  // faut mettre à jour la version C# correspondante.
  friend ItemVectorView;
  friend ItemGroup;
  template <int Extent> friend class ItemConnectedListView;
  template <typename ItemType, int Extent> friend class ItemConnectedListViewT;
  template <typename ItemType> friend class ItemVectorViewT;

 public:

  // NOTE: Si on ajoute des valeurs ici, il faut vérifier s'il faut les
  // propager dans les méthodes telles que subView().
  enum
  {
    F_Contiguous = 1 << 1, //!< Les numéros locaux sont contigüs.
    F_Contigous = F_Contiguous
  };

 public:

  //! Construit une vue vide
  ItemIndexArrayView() = default;

  // TODO: A supprimer
  //! Construit une vue à partir des numéros locaux \a local_ids
  explicit ItemIndexArrayView(const Int32ConstArrayView local_ids)
  : m_view(local_ids, 0)
  {}

  explicit ItemIndexArrayView(const impl::ItemLocalIdListContainerView& view)
  : m_view(view)
  {
  }

 public:

  //! Accède au \a i-ème élément du vecteur
  inline Int32 operator[](Integer index) const
  {
    return m_view.localId(index);
  }

  //! Nombre d'éléments du vecteur
  Int32 size() const
  {
    return m_view.size();
  }

  //! Ajoute à \a ids la liste des localIds() du vecteur.
  void fillLocalIds(Array<Int32>& ids) const;

  //! Sous-vue à partir de l'élément \a abegin et contenant \a asize éléments
  inline ItemIndexArrayView subView(Integer abegin, Integer asize) const
  {
    // On propage le flag F_Contigous sur la sous-vue.
    // Pour les autres flags, il faudra vérifier s'il faut les propager.
    return ItemIndexArrayView(m_view._idsWithoutOffset().subView(abegin, asize), m_view.m_local_id_offset, m_flags);
  }

  Int32 flags() const
  {
    return m_flags;
  }

  bool isContigous() const { return isContiguous(); }

  //! Vrai si les localIds() sont contigüs
  bool isContiguous() const
  {
    return m_flags & F_Contigous;
  }

  friend std::ostream& operator<<(std::ostream& o, const ItemIndexArrayView& a)
  {
    o << a.m_view;
    return o;
  }

 public:

  // TODO Rendre obsolète (3.11+)
  //! Tableau des numéros locaux des entités
  Int32ConstArrayView localIds() const
  {
    return m_view._idsWithoutOffset();
  }

  ARCANE_DEPRECATED_REASON("Y2023: This method is internal to Arcane. Do not use it")
  operator Int32ConstArrayView() const
  {
    return _localIds();
  }

 private:

  ARCANE_DEPRECATED_REASON("Y2023: This method is internal to Arcane. Do not use it")
  const Int32* unguardedBasePointer() const
  {
    return _data();
  }

  ARCANE_DEPRECATED_REASON("Y2023: This method is internal to Arcane. Do not use it")
  const Int32* data() const
  {
    return _data();
  }

 protected:

  impl::ItemLocalIdListContainerView m_view;
  Int32 m_flags = 0;

 private:

  ItemIndexArrayView(SmallSpan<const Int32> local_ids, Int32 local_id_offset, Int32 aflags)
  : m_view(local_ids, local_id_offset)
  , m_flags(aflags)
  {}

  const Int32* _data() const
  {
    return m_view.m_local_ids;
  }

  Int32ConstArrayView _localIds() const
  {
    return m_view._idsWithoutOffset();
  }
  Int32 _localIdOffset() const
  {
    return m_view.m_local_id_offset;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
