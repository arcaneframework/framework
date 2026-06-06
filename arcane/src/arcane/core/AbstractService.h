// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AbstractService.h                                           (C) 2000-2025 */
/*                                                                           */
/* Base class of a service.                                                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ABSTRACTSERVICE_H
#define ARCANE_CORE_ABSTRACTSERVICE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"

#include "arcane/core/ArcaneTypes.h"
#include "arcane/core/IService.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Base class of a service.
 *
 * This class is THE low-level implementation class for the \a IService interface.
 *
 * \ingroup Service
 */
class ARCANE_CORE_EXPORT AbstractService
: public TraceAccessor
, public IService
{
 protected:

  //! Constructor from a \a ServiceBuildInfo
  explicit AbstractService(const ServiceBuildInfo&);

 public:

  //! Destructor
  ~AbstractService() override;

 public:

  /*!
   * \brief Build-level construction of the service.
   *
   * This method is called right after the constructor.
   */
  virtual void build() {}

 public:

  //! Access to service information. See \a IServiceInfo for details
  IServiceInfo* serviceInfo() const override { return m_service_info; }

  //! Access to the base interface of main Arcane objects
  IBase* serviceParent() const override { return m_parent; }

  //! Returns the low-level \a IService interface of the service
  IService* serviceInterface() override { return this; }

 private:

  IServiceInfo* m_service_info = nullptr;
  IBase* m_parent = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
