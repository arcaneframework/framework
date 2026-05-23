// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemGroupsSynchronize.h                                     (C) 2000-2016 */
/*                                                                           */
/* Group synchronizations.                                                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESH_ITEMGROUPSSYNCHRONIZE_H
#define ARCANE_MESH_ITEMGROUPSSYNCHRONIZE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/List.h"
#include "arcane/utils/TraceAccessor.h"

#include "arcane/MeshVariable.h"

#include "arcane/mesh/MeshGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IMesh;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_MESH_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Information to synchronize groups between sub-domains.

 Synchronizing a group means that every sub-domain that possesses a
 entity type sends the group information to the others.

 To be able to use this class, it is necessary to ensure that the synchronization information
 is up to date (IParallelMng::computeSynchronizeInfos()).

 After creating an instance, it is enough to call the method
 synchronize() to synchronize the group. For example, to synchronize
 the face groups:
 \code
 ItemGroupsSynchronize igs(m_mesh->faceFamily());
 igs.synchronize();
 \endcode
 */
class ItemGroupsSynchronize
: public TraceAccessor
{
 public:

  /*!
   * \brief Create an instance to synchronize all groups
   * of the family \a item_family.
   */
  ItemGroupsSynchronize(IItemFamily* item_family);
  /*!
   * \brief Create an instance to synchronize the groups \a groups
   * of the family \a item_family.
   */
  ItemGroupsSynchronize(IItemFamily* item_family,ItemGroupCollection groups);
  ~ItemGroupsSynchronize();

 public:

  //! Synchronizes the groups
  void synchronize();
  /*!
   * \brief Checks if the groups are synchronized.
   *
   * \retval the number of unsynchronized entities.
   */
  Integer checkSynchronize();

 public:

  IItemFamily* m_item_family;
  typedef Int32 IntAggregator; //!< Type used for aggregating group communications
  ItemVariableScalarRefT<IntAggregator> m_var;
  ItemGroupList m_groups;

 private:
  void _setGroups();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_MESH_END_NAMESPACE
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
