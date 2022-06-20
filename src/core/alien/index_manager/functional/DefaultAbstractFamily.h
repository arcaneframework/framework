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

#include <alien/index_manager/IAbstractFamily.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*! This module relies on the fact that local_id of the item i is also i.
 */
class ALIEN_EXPORT DefaultAbstractFamily : public IAbstractFamily
{
 public:
  DefaultAbstractFamily(const DefaultAbstractFamily& family);

  /*! Build a family for locally known unique ids.
   *
   * @param uniqueIds Array of locally known uniqueIds.
   * @param owners Array of item owners
   * @param parallel_mng Parallel Manager.
   */
  DefaultAbstractFamily(ConstArrayView<Int64> uniqueIds,
                        ConstArrayView<Integer> owners,
                        IMessagePassingMng* parallel_mng);

  /*! Build a family for locally owned unique ids.
   *
   * @param uniqueIds Array of locally owned uniqueIds.
   * @param parallel_mng Parallel Manager.
   */
  DefaultAbstractFamily(ConstArrayView<Int64> uniqueIds,
                        IMessagePassingMng* parallel_mng);

  ~DefaultAbstractFamily() override = default;

 public:
  IAbstractFamily* clone() const override { return new DefaultAbstractFamily(*this); }

 public:
  Int32 maxLocalId() const override { return m_unique_ids.size(); }

  void uniqueIdToLocalId(
  ArrayView<Int32> localIds, ConstArrayView<Int64> uniqueIds) const override;

  IAbstractFamily::Item item(Int32 localId) const override;

  SafeConstArrayView<Integer> owners(ConstArrayView<Int32> localIds) const override;

  SafeConstArrayView<Int64> uids(ConstArrayView<Int32> localIds) const override;

  SafeConstArrayView<Int32> allLocalIds() const override;

 private:
  IMessagePassingMng* m_parallel_mng;
  UniqueArray<Int64> m_unique_ids;
  UniqueArray<Integer> m_owners;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
