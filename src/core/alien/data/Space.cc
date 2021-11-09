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
 * \file Space.cc
 * \brief Space.cc
 */

#include "Space.h"

#include <map>

#include <arccore/base/BaseTypes.h>
#include <arccore/base/FatalErrorException.h>
#include <arccore/base/String.h>

#include <arccore/collections/Array.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

using namespace Arccore;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Internal structure of Space object
 */
class Space::Internal final
{
 public:
  /*!
   * \brief Full constructor
   * \param[in] size The size of the space
   * \param[in] name The name of the space
   */
  Internal(Integer size, String name)
  : m_size(size)
  , m_name(name)
  , m_fields()
  , m_labels()
  {}

  /*!
   * \brief Size constructor
   * \param[in] size The size of the space
   */
  Internal(Integer size)
  : m_size(size)
  , m_name()
  , m_fields()
  , m_labels()
  {}

  /*!
   * \brief Get space size
   * \returns The size of the space
   */
  Integer size() const { return m_size; }

  /*!
   * \brief Get space name
   * \returns The name of the space
   */
  const String& name() const { return m_name; }

  /*!
   * \brief Set label on matrix entries
   * \param[in] label The name of the label
   * \param[in] indices The indices to which label is set
   */
  void setField(String label, const UniqueArray<Arccore::Integer>& indices)
  {
    if (m_fields.find(label) != m_fields.end())
      throw Alien::FatalErrorException("Field already defined");
    m_fields[label] = indices;
    m_labels.add(label);
  }

  /*!
   * \brief Get the number of fields
   * \returns The number of fields (labels)
   */
  Integer nbField() const { return static_cast<Integer>(m_fields.size()); }

  /*!
   * \brief Get the indices associated to a label
   * \param[in] label The requested label
   * \returns The indices associated to the field
   */
  const UniqueArray<Arccore::Integer>& field(String label) const
  {
    auto field = m_fields.find(label);
    if (field == m_fields.end())
      throw Alien::FatalErrorException("Field not defined");
    return field->second;
  }

  /*!
   * \brief Get indices associated to the i-th field
   * \para[in] i The requested field
   * \returns The indices associated to the field
   */
  const UniqueArray<Arccore::Integer>& field(Integer i) const
  {
    return field(m_labels[i]);
  }

  /*!
   * \brief Get the label of the i-th field
   * \param[in] i The requested field
   * \returns The associated label
   */
  String fieldLabel(Integer i) const { return m_labels[i]; }

 private:
  //! The size of the space
  Integer m_size;
  //! The name of the space
  String m_name;
  //! Mapping label to indices
  std::map<String, UniqueArray<Arccore::Integer>> m_fields;
  //! Direct acces for labels
  UniqueArray<String> m_labels;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Space::Space()
: Space(0)
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Space::Space(Integer size)
: m_internal(new Internal(size))
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Space::Space(Integer size, String name)
: m_internal(new Internal(size, name))
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Space::Space(const Space& space)
: ISpace()
, m_internal(space.m_internal)
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Space::Space(Space&& space)
: m_internal(space.m_internal)
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Space::~Space() {}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Space&
Space::operator=(const Space& src)
{
  m_internal = src.m_internal;
  return *this;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Space&
Space::operator=(Space&& src)
{
  m_internal = src.m_internal;
  return *this;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer
Space::size() const
{
  return m_internal->size();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool Space::operator==(const ISpace& space) const
{
  return (size() == space.size()) && ((name() == space.name()) || (name().empty() || space.name().empty()));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool Space::operator!=(const ISpace& space) const
{
  return not operator==(space);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const String&
Space::name() const
{
  return m_internal->name();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Space::setField(String label, const UniqueArray<Arccore::Integer>& indices)
{
  m_internal->setField(label, indices);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer
Space::nbField() const
{
  return m_internal->nbField();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const UniqueArray<Arccore::Integer>&
Space::field(String label) const
{
  return m_internal->field(label);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const UniqueArray<Arccore::Integer>&
Space::field(Integer i) const
{
  return m_internal->field(i);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String
Space::fieldLabel(Integer i) const
{
  return m_internal->fieldLabel(i);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

std::shared_ptr<ISpace>
Space::clone() const
{
  return std::make_shared<Space>(*this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
