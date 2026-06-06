// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IDataStorageFactory.h                                       (C) 2000-2025 */
/*                                                                           */
/* Interface of a data container factory.                                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IDATASTORAGEFACTORY_H
#define ARCANE_CORE_IDATASTORAGEFACTORY_H
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
 * \internal
 * \brief Interface of a data container factory.
 */
class ARCANE_CORE_EXPORT IDataStorageFactory
{
 public:

  virtual ~IDataStorageFactory() = default;

 public:

  //! Creates a simple data type.
  virtual Ref<IData> createSimpleDataRef(const DataStorageBuildInfo& dsbi) = 0;

  //! Information about the created container type
  virtual DataStorageTypeInfo storageTypeInfo() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
