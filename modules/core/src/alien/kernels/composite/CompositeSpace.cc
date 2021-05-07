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

#include "CompositeSpace.h"

#include <map>

#include <arccore/base/FatalErrorException.h>
#include <arccore/base/String.h>
#include <arccore/base/TraceInfo.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

using namespace Arccore;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace CompositeKernel
{

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  // Structure interne étant partagée
  struct Space::Internal
  {
    String m_name;

    // Mapping label -> indices des champs
    std::map<String, UniqueArray<Integer>> m_fields;
    // Pour accès direct
    UniqueArray<String> m_labels;

    UniqueArray<std::shared_ptr<ISpace>> m_sub_spaces;
  };

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  Space::Space()
  : m_internal(new Space::Internal())
  {}

  /*---------------------------------------------------------------------------*/

  Space::Space(const Space& space)
  : ISpace()
  , m_internal(space.m_internal)
  {}

  /*---------------------------------------------------------------------------*/

  Space::Space(Space&& space)
  : m_internal(space.m_internal)
  {}

  /*---------------------------------------------------------------------------*/

  Space& Space::operator=(const Space& src)
  {
    m_internal = src.m_internal;
    return *this;
  }

  /*---------------------------------------------------------------------------*/

  Space& Space::operator=(Space&& src)
  {
    m_internal = src.m_internal;
    return *this;
  }

  /*---------------------------------------------------------------------------*/

  Integer Space::size() const
  {
    if (subSpaceSize() == 0)
      return 0;
    Integer size = 0;
    for (auto& s : m_internal->m_sub_spaces)
      size += s->size();
    return size;
  }

  /*---------------------------------------------------------------------------*/

  bool Space::operator==(const ISpace& space) const
  {
    return size() == space.size() && space.name().empty();
  }

  /*---------------------------------------------------------------------------*/

  bool Space::operator!=(const ISpace& space) const { return not operator==(space); }

  /*---------------------------------------------------------------------------*/

  const String& Space::name() const { return m_internal->m_name; }

  /*---------------------------------------------------------------------------*/

  void Space::setField(String label, const UniqueArray<Integer>& indices)
  {
    auto& fields = m_internal->m_fields;
    if (fields.find(label) != fields.end())
      throw Alien::FatalErrorException(A_FUNCINFO, "Field already defined");
    fields[label] = indices;
    m_internal->m_labels.add(label);
  }

  /*---------------------------------------------------------------------------*/

  Integer Space::nbField() const { return m_internal->m_fields.size(); }

  /*---------------------------------------------------------------------------*/

  const UniqueArray<Integer>& Space::field(String label) const
  {
    auto& fields = m_internal->m_fields;
    auto field = fields.find(label);
    if (field == fields.end())
      throw Alien::FatalErrorException(A_FUNCINFO, "Field not defined");
    return field->second;
  }

  /*---------------------------------------------------------------------------*/

  const UniqueArray<Integer>& Space::field(Integer i) const
  {
    return field(m_internal->m_labels[i]);
  }

  /*---------------------------------------------------------------------------*/

  String Space::fieldLabel(Integer i) const { return m_internal->m_labels[i]; }

  /*---------------------------------------------------------------------------*/

  std::shared_ptr<ISpace> Space::clone() const { return std::make_shared<Space>(*this); }

  /*---------------------------------------------------------------------------*/

  void Space::resizeSubSpace(Integer size)
  {
    m_internal->m_sub_spaces.resize(size);
    for (auto& s : m_internal->m_sub_spaces)
      s.reset(new Space());
  }

  /*---------------------------------------------------------------------------*/

  std::shared_ptr<ISpace>& Space::operator[](Integer i)
  {
    ALIEN_ASSERT(i < subSpaceSize(), "Bound error");
    return m_internal->m_sub_spaces[i];
  }

  /*---------------------------------------------------------------------------*/

  const std::shared_ptr<ISpace>& Space::operator[](Integer i) const
  {
    ALIEN_ASSERT(i < subSpaceSize(), "Bound error");
    return m_internal->m_sub_spaces[i];
  }

  /*---------------------------------------------------------------------------*/

  Integer Space::subSpaceSize() const { return m_internal->m_sub_spaces.size(); }

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

} // namespace CompositeKernel

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
