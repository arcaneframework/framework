// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*!
 * \file Universe.h
 * \brief Universe.h
 */

#pragma once

#include <alien/utils/Precomp.h>
#include <arccore/trace/TraceGlobal.h>

#include <memory>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

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
