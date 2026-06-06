// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ActionWrapperService.h                                      (C) 2000-2025 */
/*                                                                           */
/* Service wrapping an action.                                               */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ACTIONWRAPPER_H
#define ARCANE_CORE_ACTIONWRAPPER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/IActionWrapperService.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 *
 * \brief Wrapper around an action.
 */
class ActionWrapperService
: public IActionWrapperService
{
 public:

  explicit ActionWrapperService(const ServiceBuildInfo& sbi);
  ~ActionWrapperService() override;

 public:

  //! Parent of this service
  IBase* serviceParent() const override;

  //! Service information
  IServiceInfo* serviceInfo() const override { return m_service_info; }

 private:

  IApplication* m_application = nullptr;
  IServiceInfo* m_service_info = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
