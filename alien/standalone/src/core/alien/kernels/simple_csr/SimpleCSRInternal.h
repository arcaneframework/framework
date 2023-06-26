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

#include <alien/kernels/simple_csr/CSRStructInfo.h>
#include <alien/kernels/simple_csr/SimpleCSRPrecomp.h>

/*---------------------------------------------------------------------------*/

namespace Alien::SimpleCSRInternal
{

/*---------------------------------------------------------------------------*/

template <typename ValueT = Real>
class MatrixInternal
{
 public:
  typedef ValueT ValueType;
  typedef MatrixInternal<ValueType> ThisType;
  typedef CSRStructInfo ProfileType;

 public:
  MatrixInternal(bool is_variable_block = false)
  : m_profile(new ProfileType(is_variable_block))
  {}

  ~MatrixInternal() {}

  ConstArrayView<ValueType> getValues() const { return m_values; }

  UniqueArray<ValueType>& getValues() { return m_values; }

  ValueType* getDataPtr() { return m_values.data(); }

  ValueType const* getDataPtr() const { return m_values.data(); }

  // Remark: once a profile is associated to a matrix he should not allow profile change
  CSRStructInfo& getCSRProfile() { return *m_profile; }

  const CSRStructInfo& getCSRProfile() const { return *m_profile; }

  Integer getRowSize(Integer row) const { return m_profile->getRowSize(row); }

  void clear() { m_values.resize(0); }

  MatrixInternal<ValueT>* clone() const { return new MatrixInternal<ValueT>(*this); }

  template <typename T>
  void copy(const MatrixInternal<T>& internal)
  {
    m_values.copy(internal.getValues());
    m_profile->copy(internal.getCSRProfile());
  }

  bool needUpdate()
  {
    return m_is_update != true;
  }

  void notifyChanges()
  {
    m_is_update = false;
  }

  void endUpdate()
  {
    m_is_update = true;
  }

  bool m_is_update = false;
  UniqueArray<ValueType> m_values;
  std::shared_ptr<CSRStructInfo> m_profile;
};

/*---------------------------------------------------------------------------*/

} // namespace Alien::SimpleCSRInternal

/*---------------------------------------------------------------------------*/
