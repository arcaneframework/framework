/*
* Copyright 2020 IFPEN-CEA
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*
* SPDX-License-Identifier: Apache-2.0
*/

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

#include <arccore/message_passing/BasicSerializeMessage.h>
#include <arccore/message_passing/ISerializeMessage.h>
#include <arccore/message_passing/ISerializeMessageList.h>
#include <arccore/serialize/BasicSerializer.h>
#include <arccore/serialize/ISerializer.h>

#include <arccore/trace/ITraceMng.h>

#include <alien/utils/arccore/AlienTypes.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
