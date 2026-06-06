// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IFactoryService.h                                           (C) 2000-2025 */
/*                                                                           */
/* Factory service interface.                                                */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IFACTORYSERVICE_H
#define ARCANE_CORE_IFACTORYSERVICE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/IService.h"
#include "arcane/core/ServiceBuildInfo.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Factory interface for a class implementing \a InterfaceType.
 */
template <typename InterfaceType>
class IFactoryServiceT
: public IService
{
 protected:

  IFactoryServiceT() = default;

 public:

  /*!
   * \brief Create an instance.
   *
   * The returned object is guaranteed not to be null.
   */
  virtual InterfaceType* createInstance() = 0;

  /*!
   * \brief Create an instance for the mesh \a mesh.
   *
   * Only subdomain services support this type of creation.
   *
   * The returned object is guaranteed not to be null.
   */
  virtual InterfaceType* createInstance(IMesh* mesh) = 0;

  /*! \brief Create a singleton instance.
   *
   * The returned object is guaranteed not to be null. The returned instance
   * is always the same.
   */
  virtual InterfaceType* singletonInstance() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
