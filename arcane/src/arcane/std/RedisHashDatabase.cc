// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RedisHashDatabase.cc                                        (C) 2000-2023 */
/*                                                                           */
/* Base de données de hash gérée par le système de fichier.                  */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/Array.h"
#include "arcane/utils/FatalErrorException.h"

#include "arcane/std/internal/IHashDatabase.h"
#include "arcane/std/internal/IRedisContext.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class RedisHashDatabase
: public TraceAccessor
, public IHashDatabase
{
 public:

  RedisHashDatabase(ITraceMng* tm, const String& machine_name, Int32 port)
  : TraceAccessor(tm)
  , m_context(createRedisContext(tm))
  {
    ARCANE_CHECK_POINTER(m_context.get());
    m_context->open(machine_name, port);
  }
  ~RedisHashDatabase()
  {
  }

 public:

  void writeValues(const HashDatabaseWriteArgs& args, HashDatabaseWriteResult& xresult) override
  {
    // TODO: ne faire le write que si la clé n'est pas présente
    m_context->sendBuffer(args.hashValue(), args.values());
    xresult.setHashValueAsString(args.hashValue());
  }

  void readValues(const HashDatabaseReadArgs& args) override
  {
    UniqueArray<std::byte> bytes;
    m_context->getBuffer(args.hashValueAsString(), bytes);
    if (bytes.size() != args.values().size())
      ARCANE_FATAL("Bad size expected={0} actual={1}", args.values().size(), bytes.size());
    args.values().copy(bytes);
  }

 private:

  Ref<IRedisContext> m_context;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" Ref<IHashDatabase>
createRedisHashDatabase(ITraceMng* tm, const String& machine_ip, Int32 port)
{
  return makeRef<IHashDatabase>(new RedisHashDatabase(tm, machine_ip, port));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
