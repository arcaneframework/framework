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
 * \file SolverFabricRegisterer.h
 * \brief SolverFabricRegisterer.h
 */
#pragma once

#include <alien/core/backend/ISolverFabric.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup core
 * \brief Solver fabric registerer
 *
 * Allows to register a solver fabric to convert a solver from a format to another
 */
class ALIEN_EXPORT SolverFabricRegisterer
{
 public:
  //! Type of the solver fabric function
  typedef ISolverFabric* (*FabricCreateFunc)();

 public:
  /*!
   * \brief Creates a solver fabric registerer
   * \param[in] func solver fabric function
   */
  explicit SolverFabricRegisterer(FabricCreateFunc func);

  //! Free resources
  ~SolverFabricRegisterer() = default;

 public:
  /*!
   * \brief Get the fabric from one solver format to another one
   * \param[in] from Backend id of the source format
   * \param[in] to Backend id of the target format
   * \returns solver format fabric
   */
  static ISolverFabric* getSolverFabric(BackEndId back_end);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Macro to register a solver fabric
 */
#define REGISTER_SOLVER_FABRIC(fabric) \
  extern "C++" Alien::ISolverFabric* alienCreateSolverFabric_##fabric() \
  { \
    return new fabric(); \
  } \
  Alien::SolverFabricRegisterer globaliSolverFabricRegisterer_##fabric( \
  alienCreateSolverFabric_##fabric)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
