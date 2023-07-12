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
 * \file Universe.h
 * \brief Universe.h
 */

#pragma once

#include <alien/utils/Precomp.h>

#include <memory>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{
class ITraceMng;
}

namespace Alien
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class UniverseDataBase;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Verbosity
{
  //! Verbosity level
  enum Level
  {
    None = 3,
    Info = 2,
    Warning = 1,
    Debug = 0
  };
} // namespace Verbosity

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Alien universe. Common structure to store shared objects between all elements of
 * the library Alien.
 */
class ALIEN_EXPORT Universe final
{
 public:
  //! Default constructor
  Universe();

 private:
  Universe(const Universe&) = delete;

  Universe(Universe&&) = delete;

  void operator=(const Universe&) = delete;

  void operator=(Universe&&) = delete;

 public:
  /*!
   * \brief Set trace manager
   * \param[in] traceMng The trace manager
   */
  void setTraceMng(Arccore::ITraceMng* traceMng);

  /*!
   * \brief Get the trace manager
   * \returns The trace manager
   */
  Arccore::ITraceMng* traceMng() const;

  /*!
   * \brief Set verbosity level
   * \param[in] level The verbosity level
   */
  void setVerbosityLevel(Verbosity::Level level);

  /*!
   * \brief Get the verbosity level
   * \returns The verbosity level
   */
  Verbosity::Level verbosityLevel() const;

  //! Reset the universe
  void reset();

  /*!
   * \brief Access the universe data base
   * \returns The universe data base
   */
  UniverseDataBase& dataBase();

 private:
  //! Creates the universe
  void bigBang();

 private:
  struct Internal;
  //! Internal structure of the universe
  static std::shared_ptr<Internal> m_internal;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline void
setTraceMng(ITraceMng* trace)
{
  Universe().setTraceMng(trace);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline void
setVerbosityLevel(Verbosity::Level level)
{
  Universe().setVerbosityLevel(level);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline Verbosity::Level
verbosityLevel()
{
  return Universe().verbosityLevel();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
