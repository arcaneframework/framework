// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#ifndef ARCCORE_ALINA_TESTS_TESTMAINMPI_H
#define ARCCORE_ALINA_TESTS_TESTMAINMPI_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/alina/AlinaGlobal.h"
#include "arccore/message_passing/MessagePassingGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace AlinaTest
{

// Global communicator (initialized in TestMain.cc)
extern Arcane::Ref<Arcane::MessagePassing::IMessagePassingMng> global_message_passing_mng;

} // namespace AlinaTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
