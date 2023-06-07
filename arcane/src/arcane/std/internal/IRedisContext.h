// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IRedisContext.h                                             (C) 2000-2023 */
/*                                                                           */
/* Interface d'un contexte pour Redis.                                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_STD_INTERNAL_IREDISCONTEXT_H
#define ARCANE_STD_INTERNAL_IREDISCONTEXT_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Ref.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_STD_EXPORT IRedisContext
{
 public:

  virtual ~IRedisContext() = default;

 public:

  virtual void open(const String& machine, Int32 port) = 0;
  virtual void sendBuffer(const String& key, Span<const std::byte> bytes) = 0;
  virtual void getBuffer(const String& key, Array<std::byte>& bytes) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_STD_EXPORT Ref<IRedisContext>
createRedisContext(ITraceMng* tm);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
