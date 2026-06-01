// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* FloatingPointExceptionSentry.h                              (C) 2000-2023 */
/*                                                                           */
/* Temporary activation/deactivation of floating-point exceptions            */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_FLOATINGPOINTEXCEPTIONSENTRY_H
#define ARCANE_UTILS_FLOATINGPOINTEXCEPTIONSENTRY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Class allowing temporary activation/deactivation of exceptions
 * of the processor.
 *
 * This class allows temporary modification of the state of the exceptions
 * floating-point. The previous state is reactivated when the destructor is called.
 */
class ARCANE_UTILS_EXPORT FloatingPointExceptionSentry
{
 public:

  explicit FloatingPointExceptionSentry(bool want_active);
  ~FloatingPointExceptionSentry();

 private:

  bool m_want_active = false;
  bool m_is_active = false;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
