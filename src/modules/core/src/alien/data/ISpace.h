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
 * \file ISpace.h
 * \brief ISpace.h
 */

#pragma once

#include <alien/utils/Precomp.h>

#include <arccore/collections/Array.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup data
 * \brief Interface for algebraic space objects
 */
class ALIEN_EXPORT ISpace
{
 protected:
  //! Constructor
  ISpace() {}

 private:
  /* Forbid use of copy and move constructors for implementations. */
  ISpace(const ISpace&) = delete;

  ISpace(ISpace&&) = delete;

  void operator=(const ISpace&) = delete;

  void operator=(ISpace&&) = delete;

 public:
  //! Free resources
  virtual ~ISpace() {}

  /*!
   * \brief Comparison operator
   * \param[in] space The space to compare to
   * \returns Wheteher the spaces are the same
   */
  virtual bool operator==(const ISpace& space) const = 0;

  /*!
   * \brief Comparison operator
   * \param[in] space The space to compare to
   * \returns Wheteher the spaces are different
   */
  virtual bool operator!=(const ISpace& space) const = 0;

  /*!
   * \brief Get space size
   * \returns The size of the space
   */
  virtual Arccore::Integer size() const = 0;

  /*!
   * \brief Get space name
   * \returns The name of the space
   */
  virtual const Arccore::String& name() const = 0;

  /*!
   * \brief Set label on matrix entries
   * \param[in] label The name of the label
   * \param[in] indices The indices to which label is set
   */
  virtual void setField(
  Arccore::String label, const Arccore::UniqueArray<Arccore::Integer>& indices) = 0;

  /*!
   * \brief Get the number of fields
   * \returns The number of fields (labels)
   */
  virtual Arccore::Integer nbField() const = 0;

  /*!
   * \brief Get the label of the i-th field
   * \param[in] i The requested field
   * \returns The associated label
   */
  virtual Arccore::String fieldLabel(Arccore::Integer i) const = 0;

  /*!
   * \brief Get indices associated to the i-th field
   * \para[in] i The requested field
   * \returns The indices associated to the field
   */
  virtual const Arccore::UniqueArray<Arccore::Integer>& field(
  Arccore::Integer i) const = 0;

  /*!
   * \brief Get the indices associated to a label
   * \param[in] label The requested label
   * \returns The indices associated to the field
   */
  virtual const Arccore::UniqueArray<Arccore::Integer>& field(
  Arccore::String label) const = 0;

  /*!
   * \brief Clone this object
   * \returns A clone of this object
   */
  virtual std::shared_ptr<ISpace> clone() const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
