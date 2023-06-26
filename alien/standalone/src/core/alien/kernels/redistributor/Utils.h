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

#include <alien/utils/Precomp.h>

#include <arccore/message_passing/Messages.h>

namespace Alien
{

namespace RedistributionTools
{

  template <typename T>
  static void computeCounts(ConstArrayView<T> offsets, ArrayView<T> counts)
  {
    if ((offsets.size() - 1) != counts.size())
      FatalErrorException("Redistributor communication error");
    for (Int32 p = 0; p < counts.size(); ++p) {
      counts[p] = offsets[p + 1] - offsets[p];
    }
  }

  template <typename T>
  static void exchange(IMessagePassingMng* pm, ConstArrayView<T> snd,
                       ConstArrayView<Int32> snd_offset, ArrayView<T> rcv,
                       ConstArrayView<Int32> rcv_offset)
  {
    Int32 comm_size = pm->commSize();
    UniqueArray<Int32> snd_count(comm_size);
    UniqueArray<Int32> rcv_count(comm_size);
    Alien::RedistributionTools::computeCounts(snd_offset, snd_count.view());
    Alien::RedistributionTools::computeCounts(rcv_offset, rcv_count.view());
    Arccore::MessagePassing::mpAllToAllVariable(
    pm, snd, snd_count, snd_offset, rcv, rcv_count, rcv_offset);
  }

} // namespace RedistributionTools

} // namespace Alien
