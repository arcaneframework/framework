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
 * \file ISolverFabric.h
 * \brief ISolverFabric.h
 */

#pragma once

#include <alien/core/backend/BackEnd.h>
#include <alien/expression/solver/ILinearSolver.h>
#include <alien/utils/ObjectWithTrace.h>
#include <cstdlib>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/value_semantic.hpp>
#include <boost/program_options/variables_map.hpp>

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

namespace Alien
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename tag>
class SolverFabric;

/*!
 * \ingroup core
 * \brief Solver Fabric
 *
 *
 */
class ISolverFabric : public ObjectWithTrace
{
 public:
  typedef boost::program_options::options_description CmdLineOptionDescType;
  typedef boost::program_options::variables_map CmdLineOptionType;

  typedef boost::property_tree::ptree JsonOptionType;
  //! Free resources
  virtual ~ISolverFabric() {}

 public:
  /*!
   * \brief Get the source backend id
   * \returns The source backend id
   */
  virtual BackEndId backend() const = 0;

  virtual void add_options(CmdLineOptionDescType& cmdline_options) const = 0;

  /*!
   * \brief Convert a vector from one format to another
   * \param[in] sourceImpl Implementation of the source vector
   * \param[in,out] targetImpl Implementation of the target vector
   */
  virtual Alien::ILinearSolver* create(
  CmdLineOptionType const& options, Alien::IMessagePassingMng* pm) const = 0;

  virtual Alien::ILinearSolver* create(
  JsonOptionType const& options, Alien::IMessagePassingMng* pm) const = 0;

 protected:
  template <typename T>
  static T get(
  boost::program_options::variables_map const& options, std::string const& key)
  {
    return options[key].as<T>();
  }

  template <typename T>
  static T get(boost::property_tree::ptree const& options, std::string const& key)
  {
    return options.get<T>(key);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
