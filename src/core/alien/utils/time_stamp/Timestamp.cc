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

#include "Timestamp.h"

#include <arccore/base/FatalErrorException.h>
#include <arccore/base/TraceInfo.h>

#include <alien/utils/time_stamp/TimestampMng.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

using namespace Arccore;
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Timestamp::Timestamp(const TimestampMng* manager)
: m_timestamp(0)
, m_manager(manager)
{}

/*---------------------------------------------------------------------------*/

Int64 Timestamp::timestamp() const
{
  return m_timestamp;
}

/*---------------------------------------------------------------------------*/

void Timestamp::updateTimestamp()
{
  if (m_manager)
    m_manager->updateTimestamp(this);
}

/*---------------------------------------------------------------------------*/

void Timestamp::copyTimestamp(const Timestamp& v)
{
  m_timestamp = v.m_timestamp;
}

/*---------------------------------------------------------------------------*/

void Timestamp::setTimestamp(const TimestampMng* manager, const Int64 timestamp)
{
  if (manager != m_manager)
    throw FatalErrorException(
    A_FUNCINFO, "Illegal TimestampMng used for updating timestamp");
  m_timestamp = timestamp;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
