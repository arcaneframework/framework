// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#pragma once

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include <alien/AlienConfig.h>
#include <alien/AlienExport.h>

#ifdef WIN32
#include <iso646.h>
#endif

#include <assert.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/***********************************
* Common defines.
*/

#define ALIEN_ASSERT(a, b) assert((a))

#ifdef ALIEN_CHECK
#define ALIEN_CHECK_AT((a), (b)) assert((((a) >= 0) && ((a) < (b))))
#else
#define ALIEN_CHECK_AT(a, b)
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include <arccore/base/ArgumentException.h>
#include <arccore/base/BaseTypes.h>
#include <arccore/base/FatalErrorException.h>
#include <arccore/base/NotImplementedException.h>
#include <arccore/base/TraceInfo.h>
#include <arccore/trace/ITraceMng.h>

#include <arccore/collections/Array.h>
#include <arccore/collections/Array2.h>

#include <arccore/message_passing/Messages.h>
#include <arccore/message_passing/Request.h>
#include <arccore/message_passing/ISerializeMessage.h>
#include <arccore/message_passing/ISerializeMessageList.h>
#include <arccore/serialize/ISerializer.h>

#include <arccore/trace/ITraceMng.h>

#include <alien/utils/arccore/AlienTypes.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
