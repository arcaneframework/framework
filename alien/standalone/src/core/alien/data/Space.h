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
 * \file Space.h
 * \brief Space.h
 */

#pragma once

#include <alien/data/ISpace.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup data
 * \brief Implementation of an algebraic space
 *
 * An algebraic space is composed of a name and size, on which can be added some labels
 * over some elements on this space. The purpose is to be able to differentiate different
 * spaces which have the same size, although operations between elements on those spaces
 * should not be permitted. It is close to the algebraic definition of a space
 */
class ALIEN_EXPORT Space final : public ISpace
{
 public:
  //! Constructor
  Space();

  /*!
   * \brief Anonymous constructor. The space have no name
   * \param[in] size The size of the space
   */
  Space(Arccore::Integer size);

  /*!
   * \brief Full constructor
   * \param[in] size The size of the space
   * \param[in] name The name of the space
   */
  Space(Arccore::Integer size, Arccore::String name);

  /*!
   * \brief Copy constructor
   * \param[in] s The space to copy
   */
  Space(const Space& s);

  /*!
   * \brief Rvalue constructor
   * \param[in] s The space to take
   */
  Space(Space&& s);

  //! Free resources
  ~Space();

  /*!
   * \brief Equal operator
   * \param[in] src The space to copy
   * \returns The space copied
   */
  Space& operator=(const Space& src);

  /*!
   * \brief Rvalue equal operator
   * \param[in] src The space to copy
   * \returns The space copied
   */
  Space& operator=(Space&& src);

  /*!
   * \brief Comparison operator
   * \param[in] space The space to compare to
   * \returns Wheteher the spaces are the same
   */
  bool operator==(const ISpace& space) const;

  /*!
   * \brief Comparison operator
   * \param[in] space The space to compare to
   * \returns Wheteher the spaces are different
   */
  bool operator!=(const ISpace& space) const;

  /*!
   * \brief Get space size
   * \returns The size of the space
   */
  Arccore::Integer size() const;

  /*!
   * \brief Get space name
   * \returns The name of the space
   */
  const Arccore::String& name() const;

  /*!
   * \brief Set label on matrix entries
   * \param[in] label The name of the label
   * \param[in] indices The indices to which label is set
   */
  void setField(
  Arccore::String label, const Arccore::UniqueArray<Arccore::Integer>& indices);

  /*!
   * \brief Get the number of fields
   * \returns The number of fields (labels)
   */
  Arccore::Integer nbField() const;

  /*!
   * \brief Get the label of the i-th field
   * \param[in] i The requested field
   * \returns The associated label
   */
  Arccore::String fieldLabel(Arccore::Integer i) const;

  /*!
   * \brief Get indices associated to the i-th field
   * \para[in] i The requested field
   * \returns The indices associated to the field
   */
  const Arccore::UniqueArray<Arccore::Integer>& field(Arccore::Integer i) const;

  /*!
   * \brief Get the indices associated to a label
   * \param[in] label The requested label
   * \returns The indices associated to the field
   */
  const Arccore::UniqueArray<Arccore::Integer>& field(Arccore::String label) const;

  /*!
   * \brief Clone this object
   * \returns A clone of this object
   */
  std::shared_ptr<ISpace> clone() const;

 private:
  class Internal;
  //! Internal implementation of a space
  std::shared_ptr<Internal> m_internal;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
