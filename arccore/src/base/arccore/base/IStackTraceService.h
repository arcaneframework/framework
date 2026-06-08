// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IStackTraceService.h                                        (C) 2000-2025 */
/*                                                                           */
/* Interface of a function call tracing service.                             */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_ISTACKTRACESERVICE_H
#define ARCCORE_BASE_ISTACKTRACESERVICE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/BaseTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Interface of a function call tracing service.
 */
class ARCCORE_BASE_EXPORT IStackTraceService
{
 public:

  virtual ~IStackTraceService() {} //<! Releases resources

 public:

  virtual void build() =0;

 public:

  /*!
   * \brief Character string indicating the call stack.
   *
   * \a first_function indicates the number in the stack of the first function
   * displayed in the trace.
   */
  virtual StackTrace stackTrace(int first_function=0) =0;

  /*!
   * \brief Name of a function in the call stack.
   *
   * \a function_index indicates the position of the function to return in the
   * call stack. For example, 0 indicates the current function, 1 the previous one
   * (i.e., the one calling this method).
   */
  virtual StackTrace stackTraceFunction(int function_index) =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
