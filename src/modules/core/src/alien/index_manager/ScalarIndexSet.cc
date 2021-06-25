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

#include "ScalarIndexSet.h"

#include <list>

#include <alien/index_manager/IndexManager.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

struct ScalarIndexSet::Internal
{
  const String m_name;
  const Integer m_kind;
  const Integer m_uid;
  const IndexManager* m_manager;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ScalarIndexSet::ScalarIndexSet() = default;

/*---------------------------------------------------------------------------*/

ScalarIndexSet::ScalarIndexSet(const ScalarIndexSet& en) = default;

/*---------------------------------------------------------------------------*/

ScalarIndexSet::ScalarIndexSet(ScalarIndexSet&& en) noexcept
: m_internal(std::move(en.m_internal))
{}

/*---------------------------------------------------------------------------*/

ScalarIndexSet::ScalarIndexSet(
const String& name, const Integer uid, const IndexManager* manager, Integer kind)
: m_internal(new Internal{ name, kind, uid, manager })
{}

/*---------------------------------------------------------------------------*/

ScalarIndexSet& ScalarIndexSet::operator=(const ScalarIndexSet& en) = default;

/*---------------------------------------------------------------------------*/

//! Opérateur de copie
ScalarIndexSet&
ScalarIndexSet::operator=(ScalarIndexSet&& en) noexcept
{
  m_internal = std::move(en.m_internal);
  return *this;
}

/*---------------------------------------------------------------------------*/

//! Opérateur de comparaison
bool ScalarIndexSet::operator==(const ScalarIndexSet& en) const
{
  return m_internal == en.m_internal;
}

/*---------------------------------------------------------------------------*/

ConstArrayView<Integer>
ScalarIndexSet::getOwnIndexes() const
{
  return manager()->getOwnIndexes(*this);
}

/*---------------------------------------------------------------------------*/

ConstArrayView<Integer>
ScalarIndexSet::getAllIndexes() const
{
  return manager()->getAllIndexes(*this);
}

/*---------------------------------------------------------------------------*/

ConstArrayView<Integer>
ScalarIndexSet::getOwnLocalIds() const
{
  return manager()->getOwnLocalIds(*this);
}

/*---------------------------------------------------------------------------*/

ConstArrayView<Integer>
ScalarIndexSet::getAllLocalIds() const
{
  return manager()->getAllLocalIds(*this);
}

/*---------------------------------------------------------------------------*/

String
ScalarIndexSet::getName() const
{
  return m_internal->m_name;
}

/*---------------------------------------------------------------------------*/

Integer
ScalarIndexSet::getKind() const
{
  return m_internal->m_kind;
}

/*---------------------------------------------------------------------------*/

const IAbstractFamily&
ScalarIndexSet::getFamily() const
{
  return manager()->getFamily(*this); //*m_internal->m_family;
}

/*---------------------------------------------------------------------------*/

const IndexManager*
ScalarIndexSet::manager() const
{
  return m_internal->m_manager;
}

/*---------------------------------------------------------------------------*/

Integer
ScalarIndexSet::getUid() const
{
  return m_internal->m_uid;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
