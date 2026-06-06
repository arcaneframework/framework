// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IDataFactory.h                                              (C) 2000-2025 */
/*                                                                           */
/* Interface of a data factory.                                              */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IDATAFACTORY_H
#define ARCANE_CORE_IDATAFACTORY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"
#include "arcane/core/Parallel.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Interface of a data factory.
 * \todo Rename to 'IDataFactoryMng'.
 * \warning This class is obsolete and should no longer be used outside of Arcane.
 */
class IDataFactory
{
 public:

  virtual ~IDataFactory() = default;

 public:

  //! Builds the instance
  virtual void build() = 0;

  //! Application
  virtual IApplication* application() = 0;

  /*!
   * \brief Creates an operation performing a reduction of type \a rt.
   * \todo put in another interface.
   */
  ARCCORE_DEPRECATED_2021("Do not use deprecated interface 'IDataFactory'. Use 'IDataFactoryMng' instead")
  virtual IDataOperation* createDataOperation(Parallel::eReduceType rt) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
