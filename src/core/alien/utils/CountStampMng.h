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

namespace Alien
{

class CountStampMng
{
 public:
  CountStampMng()
  : m_count_stamp(-1)
  {}

  virtual ~CountStampMng() {}

  int getCountStamp() const { return m_count_stamp; }

  int getNewCountStamp() const { return ++m_count_stamp; }

 protected:
  mutable int m_count_stamp;
};

class CountStampObject
{
 public:
  CountStampObject()
  : m_count_stamp(-1)
  {}

  virtual ~CountStampObject() {}

  bool isUpdated(int count) const { return m_count_stamp >= count; }

  void setUpdated(int count) { m_count_stamp = count; }

  int getCountStamp() const { return m_count_stamp; }

 protected:
  int m_count_stamp;
};

} // namespace Alien
