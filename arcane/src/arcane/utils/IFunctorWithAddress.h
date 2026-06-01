// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IFunctorWithAddress.h                                       (C) 2000-2012 */
/*                                                                           */
/* Interface of a functor.                                                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_IFUNCTOR_WITH_ADDRESS_H
#define ARCANE_UTILS_IFUNCTOR_WITH_ADDRESS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/IFunctor.h"
#include "arcane/utils/ArcaneGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Interface of a functor.
 * \ingroup Core
 */
class ARCANE_UTILS_EXPORT IFunctorWithAddress
: public IFunctor
{
 public:

  //! Frees resources
  virtual ~IFunctorWithAddress() {}

 public:

  /*!
   * \internal
   * \brief Returns the address of the associated method.
   * \warning This method must only be called by HYODA
   * and is not valid on all platforms.
   */
  virtual void* functorAddress() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
