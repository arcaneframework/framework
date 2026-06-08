// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ISymbolizerService.h                                        (C) 2000-2026 */
/*                                                                           */
/* Interface of a source code symbol retrieval service.                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_ISYMBOLIZERSERVICE_H
#define ARCCORE_BASE_ISYMBOLIZERSERVICE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ArccoreGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Interface of a source code symbol retrieval service.
 *
 * This service allows retrieving certain information from the source code
 * from a memory address. Among the recoverable information
 * are the file name, the method name, and the line numbers.
 *
 * \warning UNSTABLE API
 */
class ARCCORE_BASE_EXPORT ISymbolizerService
{
 public:

  virtual ~ISymbolizerService() {} //<! Releases resources

 public:

  //! Information for the call stack \a frames.
  // TODO TODO RENAME THIS METHOD
  virtual String stackTrace(ConstArrayView<StackFrame> frames) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
