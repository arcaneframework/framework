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

/*!
 * \file Universe.cc
 * \brief Universe.cc
 */

#include "Universe.h"

#include <arccore/trace/ITraceMng.h>

#include "UniverseDataBase.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

using namespace Arccore;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Universe internal structure
 */
struct Universe::Internal
{
  //! Constructor
  Internal();

  //! Trace manager
  ITraceMng* m_trace_mng;
  //! Verbosity level
  Verbosity::Level m_level;
  //! Universe data base
  UniverseDataBase m_data_base;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Universe::Internal::Internal()
: m_trace_mng(nullptr)
, m_level(Verbosity::Info)
, m_data_base()
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

std::shared_ptr<Universe::Internal> Universe::m_internal = nullptr;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Universe::Universe()
{
  bigBang();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

UniverseDataBase&
Universe::dataBase()
{
  return m_internal->m_data_base;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Universe::bigBang()
{
  if (!m_internal) {
    m_internal.reset(new Internal());
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Universe::setTraceMng(ITraceMng* traceMng)
{
  m_internal->m_trace_mng = traceMng;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ITraceMng*
Universe::traceMng() const
{
  return m_internal->m_trace_mng;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Universe::setVerbosityLevel(Verbosity::Level level)
{
  m_internal->m_level = level;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Verbosity::Level
Universe::verbosityLevel() const
{
  return m_internal->m_level;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Universe::reset()
{
  m_internal.reset(new Internal());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
