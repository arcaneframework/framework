// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CartesianMeshPatchListView.h                                (C) 2000-2023 */
/*                                                                           */
/* Vue sur une liste de patchs.                                              */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CARTESIANMESH_CARTESIANMESPATCHLISTVIEW_H
#define ARCANE_CARTESIANMESH_CARTESIANMESPATCHLISTVIEW_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArrayView.h"

#include "arcane/core/ArcaneTypes.h"

#include "arcane/cartesianmesh/CartesianPatch.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class CartesianPatchGroup;

/*!
 * \brief Vue sur une liste de patchs.
 *
 * Les instances de cette classe sont invalidées si la liste des patchs évolue.
 */
class ARCANE_CARTESIANMESH_EXPORT CartesianMeshPatchListView
{
  friend CartesianMeshImpl;
  friend CartesianPatchGroup;

 public:

  //! Sentinelle pour l'itération pour une liste de patchs.
  class Sentinel
  {};

  //! Itérateur pour une liste de patchs.
  class Iterator
  {
    friend CartesianMeshPatchListView;

   public:

    typedef std::forward_iterator_tag iterator_category;
    typedef ICartesianMeshPatch value_type;
    typedef Int32 size_type;

   public:

    Iterator() = default;

   public:

    CartesianPatch operator*() const { return CartesianPatch(m_patches[m_index]); }
    Iterator& operator++()
    {
      ++m_index;
      return (*this);
    }
    friend bool operator==(const Iterator& a, const Iterator& b)
    {
      return (a.m_patches.data() + a.m_index) == (b.m_patches.data() + b.m_index);
    }
    friend bool operator!=(const Iterator& a, const Iterator& b)
    {
      return !(operator==(a, b));
    }
    friend bool operator==(const Iterator& a, const Sentinel&)
    {
      return a.m_patches.size() != a.m_index;
    }
    friend bool operator!=(const Iterator& a, const Sentinel& b)
    {
      return !(operator==(a, b));
    }

   private:

    Iterator(ConstArrayView<ICartesianMeshPatch*> v, Int32 index)
    : m_index(index)
    , m_patches(v)
    {}

   private:

    Int32 m_index = 0;
    ConstArrayView<ICartesianMeshPatch*> m_patches;
  };

 public:

  CartesianMeshPatchListView() = default;

 private:

  // Constructeur pour 'CartesiaMeshImpl'
  explicit CartesianMeshPatchListView(ConstArrayView<ICartesianMeshPatch*> v)
  : m_patches(v)
  {}

 public:

  Iterator begin() const { return Iterator(m_patches, 0); }
  Sentinel end() const { return {}; }
  Iterator endIterator() const { return Iterator(m_patches, m_patches.size()); }
  Int32 size() const { return m_patches.size(); }

 private:

  ConstArrayView<ICartesianMeshPatch*> m_patches;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

