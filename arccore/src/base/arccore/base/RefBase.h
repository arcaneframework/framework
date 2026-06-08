// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RefBase.h                                                   (C) 2000-2025 */
/*                                                                           */
/* Base class for reference management on an instance.                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_REFBASE_H
#define ARCCORE_BASE_REFBASE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ExternalRef.h"
#include "arccore/base/RefDeclarations.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Base class for reference management.
 */
class ARCCORE_BASE_EXPORT RefBase
{
  friend class ReferenceCounterImpl;
  template <typename InstanceType> friend class impl::ReferenceCounterWrapper;

 protected:

  class ARCCORE_BASE_EXPORT BasicDeleterBase
  {
  };

  class ARCCORE_BASE_EXPORT DeleterBase
  {
    using ExternalRef = Internal::ExternalRef;
    friend class ReferenceCounterImpl;

   public:

    bool hasExternal() const { return m_handle.isValid(); }
    void setNoDestroy(bool x) { m_no_destroy = x; }

   protected:

    bool _destroyHandle(const void* instance, ExternalRef& handle);
    bool _destroyHandle(void* instance, ExternalRef& handle);

   private:

    bool _destroyHandleTrue(const void* instance, ExternalRef& handle);

   protected:

    DeleterBase() = default;
    DeleterBase(ExternalRef h)
    : m_handle(std::move(h))
    {}
    DeleterBase(ExternalRef h, bool no_destroy)
    : m_handle(std::move(h))
    , m_no_destroy(no_destroy)
    {}

   protected:

    //! External handle responsible for the destruction of the instance
    Internal::ExternalRef m_handle;
    /*!
     * \brief Indicates whether the instance destructor should be called
     * when there are no more references to it.
     */
    bool m_no_destroy = false;
  };
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
