// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IGatherGroup.h                                              (C) 2000-2026 */
/*                                                                           */
/* Interface for managing groupings across a limited number of subdomains.   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifndef ARCANE_CORE_INTERNAL_IGATHERGROUP_H
#define ARCANE_CORE_INTERNAL_IGATHERGROUP_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Interface class allowing the grouping of data from certain
 * subdomains onto other subdomains.
 */
class ARCANE_CORE_EXPORT IGatherGroup
{

 public:

  virtual ~IGatherGroup() = default;

 public:

  /*!
   * \brief Method allowing determination of whether the grouping needs to be performed or if the data can be written directly.
   *
   * Non-collective call, but the returned value will be the same for all callers.
   *
   * The call to gatherToMasterIO() can still be made; the array \a in will simply be copied into the array \a out.
   */
  virtual bool isNeedGather() = 0;

  /*!
   * \brief Method allowing the grouping of data from multiple
   * subdomains onto one or more subdomains.
   *
   * Collective call.
   *
   * \param sizeof_elem The size of an element.
   * \param in Our array that we wish to group.
   * \param out The grouped array. If we are not a writer, there will be no modification.
   */
  virtual void gatherToMasterIO(Int64 sizeof_elem, Span<const Byte> in, Span<Byte> out) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Interface class allowing the calculation and storage of grouping
 * information.
 */
class ARCANE_CORE_EXPORT IGatherGroupInfo
{
 public:

  virtual ~IGatherGroupInfo() = default;

 public:

  /*!
   * \brief Method allowing the calculation of grouping information.
   *
   * Collective call.
   *
   * A second call to this method will have no effect, unless the method
   * \a needRecompute() was called beforehand.
   *
   * \param nb_elem_in The number of elements that our subdomain wishes to
   * send to the master.
   */
  virtual void computeSize(Int32 nb_elem_in) = 0;

  /*!
   * \brief Method allowing a request for recalculation of grouping information.
   * To do this, the method \a computeSize() must be called again.
   */
  virtual void setNeedRecompute() = 0;

  /*!
   * \brief Method allowing determination of whether the method \a computeSize()
   * has already been called.
   */
  virtual bool isComputed() = 0;

  /*!
   * \brief Method allowing knowledge of the number of elements that our
   * subdomain must process after reception.
   */
  virtual Int32 nbElemOutput() = 0;

  /*!
   * \brief Method allowing knowledge of the size, in bytes, of the set of
   * elements that our subdomain must process after reception.
   *
   * \param sizeof_type The size of an element.
   */
  virtual Int32 sizeOfOutput(Int32 sizeof_type) = 0;

  /*!
   * \brief Method allowing knowledge of the number of elements that each
   * third-party subdomain will send to us.
   */
  virtual SmallSpan<Int32> nbElemRecvGatherToMasterIO() = 0;

  /*!
   * \brief Method allowing knowledge of the number of writing subdomains.
   */
  virtual Int32 nbWriterGlobal() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
