// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ConnectivityItemVector.h                                    (C) 2000-2025 */
/*                                                                           */
/* Interface for entity connectivity accessors.                              */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_CONNECTIVITYITEMVECTOR_H
#define ARCANE_CORE_CONNECTIVITYITEMVECTOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArrayView.h"
#include "arcane/utils/String.h"

#include "arcane/core/ItemTypes.h"
#include "arcane/core/ItemVector.h"
#include "arcane/core/IItemConnectivity.h"
#include "arcane/core/IIncrementalItemConnectivity.h"

#include <functional>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Temporary type automatically cast to ConnectivityItemVector
 */
struct ConnectivityItemVectorCatalyst
{
  std::function<void(ConnectivityItemVector&)> set;
  std::function<void(ConnectivityItemVector&)> apply; // When C++14 available, use a template type (will avoid std::function object and allow more genericity).
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Manages the retrieval of connectivity information.
 *
 * \sa IItemConnectivity
 * \sa IIncrementalItemConnectivity
 * \sa IItemConnectivityAccessor.
 */
class ARCANE_CORE_EXPORT ConnectivityItemVector
: public ItemVector
// SDC new API: user handles ConnectivityItemVector directly and iterates on it... needs
// public inheritance.
// the Use of views as in first version is confusing for user that doesn't understand
// where the view comes from and easily invalidates it...
{
 public:

  ConnectivityItemVector(IItemConnectivity* c)
  : ItemVector(c->targetFamily())
  , m_connectivity_accessor(c)
  {
    c->_initializeStorage(this);
  }
  ConnectivityItemVector(IItemConnectivity& c)
  : ItemVector(c.targetFamily())
  , m_connectivity_accessor(&c)
  {
    c._initializeStorage(this);
  }
  ConnectivityItemVector(IIncrementalItemConnectivity* c)
  : ItemVector(c->targetFamily())
  , m_connectivity_accessor(c)
  {
    c->_initializeStorage(this);
  }
  ConnectivityItemVector(IIncrementalItemConnectivity& c)
  : ItemVector(c.targetFamily())
  , m_connectivity_accessor(&c)
  {
    c._initializeStorage(this);
  }
  ConnectivityItemVector(const ConnectivityItemVectorCatalyst& to_c)
  : ItemVector()
  , m_connectivity_accessor(nullptr)
  {
    to_c.set(*this);
    to_c.apply(*this);
  }

 public:

  //! Associated connectivity
  IItemConnectivityAccessor* accessor() const { return m_connectivity_accessor; }

  //! Returns the entities connected to \a item.
  ItemVectorView connectedItems(ItemLocalId item)
  {
    return m_connectivity_accessor->_connectedItems(item, *this);
  }

 public:

  /*!
   * \internal
   * \brief Positions the connectivity list with the entities
   * specified by \a ids.
   */
  ItemVectorView resizeAndCopy(Int32ConstArrayView ids)
  {
    this->resize(ids.size());
    this->viewAsArray().copy(ids);
    return (*this);
  }

  /*!
   * \internal
   * \brief Positions the connectivity list with the entity
   * of localId() \a id.
   */
  ItemVectorView setItem(Int32 id)
  {
    this->resize(1);
    this->viewAsArray()[0] = id;
    return (*this);
  }

  /*!
   * \internal
   * \brief Allows retrieving the ConnectivityItemVector
   */
  void operator=(const ConnectivityItemVectorCatalyst& to_con_vec)
  {
    to_con_vec.apply(*this);
  }

 private:

  IItemConnectivityAccessor* m_connectivity_accessor = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
