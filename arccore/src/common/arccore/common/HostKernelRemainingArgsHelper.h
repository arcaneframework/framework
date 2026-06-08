// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* HostKernelRemainingArgsHelper.h                             (C) 2000-2025 */
/*                                                                           */
/* Class to execute a method at the beginning and end of the kernel.         */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COMMON_HOSTKERNELREMAININGARGSHELPER_H
#define ARCCORE_COMMON_HOSTKERNELREMAININGARGSHELPER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/CommonGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Class to apply methods of additional arguments at the beginning and end of the kernel.
 */
class HostKernelRemainingArgsHelper
{
 public:

  //! Applies the functors of additional arguments at the beginning of the iteration.
  template <typename... RemainingArgs> static void
  applyAtBegin(RemainingArgs&... remaining_args)
  {
    (_doOneAtBegin(remaining_args), ...);
  }

  //! Applies the functors of additional arguments at the end of the iteration.
  template <typename... RemainingArgs> static void
  applyAtEnd(RemainingArgs&... remaining_args)
  {
    (_doOneAtEnd(remaining_args), ...);
  }

 private:

  template <typename OneArg> static void _doOneAtBegin(OneArg& one_arg)
  {
    using HandlerType = OneArg::RemainingArgHandlerType;
    HandlerType::execWorkItemAtBeginForHost(one_arg);
  }
  template <typename OneArg> static void _doOneAtEnd(OneArg& one_arg)
  {
    using HandlerType = OneArg::RemainingArgHandlerType;
    HandlerType::execWorkItemAtEndForHost(one_arg);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
