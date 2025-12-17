// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ConnectivityItemVector.h                                    (C) 2000-2025 */
/*                                                                           */
/* Interface des accesseurs des connectivités des entités.                   */
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
 * \brief Type temporaire automatiquement casté en ConnectivityItemVector
 */
struct ConnectivityItemVectorCatalyst
{
  std::function<void(ConnectivityItemVector&)> set;
  std::function<void(ConnectivityItemVector&)> apply; // When C++14 available, use a template type (will avoid std::function object and allow more genericity).
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Gère la récupération des informations de connectivité.
 *
 * \sa IItemConnectivity
 * \sa IIncrementalItemConnectivity
 * \sa IItemConnectivityAccessor.
 */
class ARCANE_CORE_EXPORT ConnectivityItemVector
: public ItemVector
// SDC new API : user handles directly the ConnectivityItemVector and iterates on it...need a public inheritance.
// the Use of views as in first version is confusing for user that doesn't understand where the view comes from and easily invalidates it...
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

  //! Connectivité associée
  IItemConnectivityAccessor* accessor() const { return m_connectivity_accessor; }

  //! Retourne les entités connectées à \a item.
  ItemVectorView connectedItems(ItemLocalId item)
  {
    return m_connectivity_accessor->_connectedItems(item, *this);
  }

 public:

  /*!
   * \internal
   * \brief Positionne la liste de connectivité avec les entités
   * spécifiées par \a ids.
   */
  ItemVectorView resizeAndCopy(Int32ConstArrayView ids)
  {
    this->resize(ids.size());
    this->viewAsArray().copy(ids);
    return (*this);
  }

  /*!
   * \internal
   * \brief Positionne la liste de connectivité avec l'entité
   * de localId() \a id.
   */
  ItemVectorView setItem(Int32 id)
  {
    this->resize(1);
    this->viewAsArray()[0] = id;
    return (*this);
  }

  /*!
   * \internal
   * \brief Permet de récupérer le ConnectivityItemVector
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
