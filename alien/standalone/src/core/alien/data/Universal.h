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
 * \file Universal.h
 * \brief Universal.h
 */

#pragma once

#include <functional>

#include <alien/data/Universe.h>
#include <alien/data/UniverseDataBase.h>
#include <alien/utils/Trace.h>
#include <iostream>
#include <utility>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup data
 * \brief Universal object
 * \tparam U The type of the object
 */
template <typename U>
struct UniversalObject
{
  /*!
   * \brief Constructor
   * \tparam T The type of the further objects
   * \param[in] t The objects
   */
  template <typename... T>
  UniversalObject(T&... t)
  : m_value(Universe().dataBase().findOrCreate<U>(t...))
  {}
  //! The object with its instancied flag
  std::pair<std::shared_ptr<U>, bool> m_value;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Stores an universal object
 * \tparam U The type of the universal object
 */
template <typename U>
class Universal : private UniversalObject<U>
, private std::shared_ptr<U>
{
 public:
  /*!
   * \brief Constructor
   * \tparam T The type of the further objects
   * \param[in] t The objects
   */
  template <typename... T>
  Universal(T&... t)
  : UniversalObject<U>(t...)
  , std::shared_ptr<U>(this->m_value.first)
  {
    alien_debug([&] {
      if (this->m_value.second)
        cout() << "Create Universal Object";
      else
        cout() << "Find Universal Object";
    });
  }

  /*
   * \brief Initialize an object
   * \param[in] f The initialize function
   */
  void first_time(std::function<void(U&)> f)
  {
    if (this->m_value.second) {
      alien_debug([&] { cout() << "Initialize Universal Object"; });
      f(*(this->get()));
    }
  }

  using std::shared_ptr<U>::operator->;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
