// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IDataStorageFactory.h                                       (C) 2000-2025 */
/*                                                                           */
/* Interface d'une fabrique de conteneur d'une donnée.                       */
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
 * \brief Interface d'une fabrique de conteneur d'une donnée.
 */
class ARCANE_CORE_EXPORT IDataStorageFactory
{
 public:
  
  virtual ~IDataStorageFactory() = default;

 public:

  //! Créé une donnée d'un type simple.
  virtual Ref<IData> createSimpleDataRef(const DataStorageBuildInfo& dsbi) =0;

  //! Informations sur le type de conteneur créé
  virtual DataStorageTypeInfo storageTypeInfo() =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
