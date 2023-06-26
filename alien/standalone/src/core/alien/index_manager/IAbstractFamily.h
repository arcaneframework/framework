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

#include <alien/utils/SafeConstArrayView.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Interface for abstract families of items.
class IAbstractFamily
{
 public:
  class Item
  {
   public:
    Item(Int64 uniqueId, Integer owner)
    : m_unique_id(uniqueId)
    , m_owner(owner)
    {}

    Int64 uniqueId() const { return m_unique_id; }

    Integer owner() const { return m_owner; }

   private:
    Int64 m_unique_id;
    Integer m_owner;
  };

 protected:
  IAbstractFamily() = default;

 public:
  virtual ~IAbstractFamily() = default;

  IAbstractFamily(const IAbstractFamily&) = delete;
  IAbstractFamily(IAbstractFamily&&) = delete;
  void operator=(const IAbstractFamily&) = delete;
  void operator=(IAbstractFamily&&) = delete;

  virtual IAbstractFamily* clone() const = 0;

  //! Max local Id for this family
  virtual Int32 maxLocalId() const = 0;

  /*! Convert unique ids to local ids
   * @throw FatalError if an item is not found.
   */
  virtual void uniqueIdToLocalId(
  ArrayView<Integer> localIds, ConstArrayView<Int64> uniqueIds) const = 0;

  //! Give back an Item from its local id.
  virtual Item item(Int32 localId) const = 0;

  //! Owners of the given local ids.
  virtual SafeConstArrayView<Integer> owners(ConstArrayView<Integer> localIds) const = 0;

  //! unique ids of the given local ids.
  virtual SafeConstArrayView<Int64> uids(ConstArrayView<Integer> localIds) const = 0;

  //! Local ids of this family members.
  virtual SafeConstArrayView<Int32> allLocalIds() const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
