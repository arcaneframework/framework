// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ISharedReference.h                                          (C) 2000-2025 */
/*                                                                           */
/* Interface of the reference counter class.                                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ISHAREDREFERENCE_H
#define ARCANE_CORE_ISHAREDREFERENCE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Ptr.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup Core
 * \brief Interface of a reference counter.
 *
 * The reference counter allows a class instance to know the number
 * of references on it. When this number reaches zero, it means
 * that the instance is no longer used. This system is used primarily
 * to automatically free memory when the number of references
 * drops to zero.
 *
 * This class is used through classes like AutoRefT which
 * allow automatically incrementing or decrementing the counter
 * of the objects they point to.
 */
class ARCANE_CORE_EXPORT ISharedReference
{
 public:

  //! Releases resources
  virtual ~ISharedReference() = default;

 public:

  //! Increments the reference counter
  virtual void addRef() = 0;

  //! Decrements the reference counter
  virtual void removeRef() = 0;

  //! Returns the value of the reference counter
  virtual Int32 refCount() const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
