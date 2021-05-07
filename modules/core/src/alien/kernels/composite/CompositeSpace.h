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

#include <memory>

#include <alien/data/ISpace.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace CompositeKernel
{

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  class ALIEN_EXPORT Space : public ISpace
  {
   public:
    Space();

    Space(const Space&);
    Space(Space&&);

    Space& operator=(const Space& src);
    Space& operator=(Space&&);

   public:
    bool operator==(const ISpace& space) const;
    bool operator!=(const ISpace& space) const;

    Integer size() const;

    const String& name() const;

    void setField(String label, const UniqueArray<Integer>& indices);

    Integer nbField() const;

    String fieldLabel(Integer i) const;

    const UniqueArray<Integer>& field(Integer i) const;
    const UniqueArray<Integer>& field(String label) const;

    std::shared_ptr<ISpace> clone() const;

   public:
    void resizeSubSpace(Integer size);

    std::shared_ptr<ISpace>& operator[](Integer i);

    const std::shared_ptr<ISpace>& operator[](Integer i) const;

    Integer subSpaceSize() const;

   private:
    struct Internal;

    std::shared_ptr<Internal> m_internal;
  };

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

} // namespace CompositeKernel

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
