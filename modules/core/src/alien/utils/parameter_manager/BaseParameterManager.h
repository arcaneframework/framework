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

#include <map>
#include <sstream>
#include <vector>

#include <alien/utils/Precomp.h>
#include <string>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class BaseParameterMng
{
 public:
  BaseParameterMng()
  : m_child_mng(nullptr)
  , m_nb_child(0)
  , m_parameters_have_changed(false)
  {}

  virtual ~BaseParameterMng()
  {
    if (m_nb_child == 1)
      delete m_child_mng;
    else
      delete[] m_child_mng;
  }

  BaseParameterMng* getChild()
  {
    if (!m_child_mng) {
      m_nb_child = 1;
      m_child_mng = new BaseParameterMng();
    }
    return m_child_mng;
  }

  [[nodiscard]] BaseParameterMng const* getChild() const { return m_child_mng; }

  [[nodiscard]] bool hasChild() const { return (m_nb_child != 0); }

  [[nodiscard]] const BaseParameterMng* getChild(Arccore::Integer child) const
  {
    ALIEN_ASSERT(m_nb_child > 0 && child < m_nb_child, "Inconsistent child values");
    return &m_child_mng[child];
  }

  BaseParameterMng* getChild(Integer child)
  {
    ALIEN_ASSERT(m_nb_child > 0 && child < m_nb_child, "Inconsistent child values");
    return &m_child_mng[child];
  }

  void setNbChild(Integer nbChild)
  {
    if (m_nb_child == 1 || m_nb_child == 0)
      delete m_child_mng;
    else
      delete[] m_child_mng;
    m_nb_child = nbChild;
    m_child_mng = new BaseParameterMng[nbChild];
  }

  template <typename ValueT>
  std::map<std::string, ValueT> const& getParams() const;

  template <typename ValueT>
  std::map<std::string, ValueT>& getParams();

  template <typename ValueT>
  void setParameter(std::string const& key, ValueT const& value)
  {
    getParams<ValueT>()[key] = value;
  }

  template <typename ValueT>
  ValueT getParameter(std::string const& key, ValueT default_value) const
  {
    auto iter = getParams<ValueT>().find(key);
    if (iter == getParams<ValueT>().end())
      return default_value;
    else
      return iter->second;
  }

  template <typename ValueT>
  void addOptions(
  std::vector<std::string>& options, std::map<std::string, ValueT> const& params)
  {
    for (auto iter = params.begin(); iter != params.end(); ++iter) {
      options.push_back(iter->first);
      std::stringstream value_str;
      value_str << iter->second;
      options.push_back(value_str.str());
    }
  }

  void addOptions(std::vector<std::string>& options)
  {
    addOptions(options, m_int_params);
    addOptions(options, m_double_params);
    addOptions(options, m_string_params);
  }

  void notifyParamChangesObserver() { m_parameters_have_changed = true; }

 public:
  void resetParamChangesObserver() { m_parameters_have_changed = false; }

  bool needUpdate() { return m_parameters_have_changed; }

 protected:
  std::map<std::string, int> m_int_params;
  std::map<std::string, double> m_double_params;
  std::map<std::string, std::string> m_string_params;
  BaseParameterMng* m_child_mng;
  Integer m_nb_child;
  bool m_parameters_have_changed;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <>
[[nodiscard]] ALIEN_EXPORT std::map<std::string, int> const&
BaseParameterMng::getParams() const;

template <>
ALIEN_EXPORT std::map<std::string, int>& BaseParameterMng::getParams();

template <>
[[nodiscard]] ALIEN_EXPORT std::map<std::string, double> const&
BaseParameterMng::getParams() const;

template <>
ALIEN_EXPORT std::map<std::string, double>& BaseParameterMng::getParams();

template <>
[[nodiscard]] ALIEN_EXPORT std::map<std::string, std::string> const&
BaseParameterMng::getParams() const;

template <>
ALIEN_EXPORT std::map<std::string, std::string>& BaseParameterMng::getParams();

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
