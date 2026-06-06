// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SharedReference.h                                           (C) 2000-2025 */
/*                                                                           */
/* Base class of a reference counter.                                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_SHAREDREFERENCE_H
#define ARCANE_CORE_SHAREDREFERENCE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ISharedReference.h"

#include <atomic>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup Core
 * \brief Implementation of a reference counter using std::atomic.
 */
class ARCANE_CORE_EXPORT SharedReference
: public ISharedReference
{
 public:

  SharedReference()
  : m_ref_count(0)
  {}

 public:

  void addRef() override;
  void removeRef() override;
  Int32 refCount() const override { return m_ref_count; }

  //! Destroys the referenced object
  virtual void deleteMe() = 0;

 private:

  std::atomic<Int32> m_ref_count; //!< Number of references on the object.
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
