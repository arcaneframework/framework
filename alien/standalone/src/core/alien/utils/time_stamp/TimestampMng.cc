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

#include "TimestampMng.h"

#include <alien/utils/Trace.h>

#include "ITimestampObserver.h"
#include "Timestamp.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

using namespace Arccore;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TimestampMng::TimestampMng()
: m_timestamp(0)
{}

/*---------------------------------------------------------------------------*/

TimestampMng::TimestampMng(const TimestampMng& tm)
: m_timestamp(tm.m_timestamp)
{}

/*---------------------------------------------------------------------------*/

Int64 TimestampMng::timestamp() const
{
  return m_timestamp;
}

/*---------------------------------------------------------------------------*/

void TimestampMng::updateTimestamp(Timestamp* ts) const
{
  alien_debug([&] {
    cout() << "Udpate Timestamp " << ts << " (" << m_observers.size()
           << " Observers) by TimestampMng " << this;
  });

  ts->setTimestamp(this, ++m_timestamp);

  for (auto& o : m_observers)
    o->updateTimestamp();
}

/*---------------------------------------------------------------------------*/

void TimestampMng::addObserver(std::shared_ptr<ITimestampObserver> observer)
{
  alien_debug([&] {
    cout() << "Add Timestamp Observer " << observer.get() << " to TimestampMng " << this;
  });

  m_observers.add(observer);
}

/*---------------------------------------------------------------------------*/

void TimestampMng::clearObservers()
{
  m_observers.clear();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
