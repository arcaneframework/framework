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

#include <string>

namespace Alien
{

class MatrixDistribution;

struct SolutionInfo
{
 public:
  typedef enum
  {
    N2_ABS_RES,
    N2_RELATIVE2RHS_RES,
    N2_RELATIVE2X0_RES
  } eConvCrit;

  // N2_ABS_RES = ||Ax-b||2
  // N2_RELATIVE2RHS_RES = ||Ax-b||2 / ||b||2
  // N2_RELATIVE2X0_RES = ||Ax-b||2 / ||x0||2

  eConvCrit m_conv_crit;
  double m_conv_crit_value;
  std::string m_solver_comment;
  Alien::MatrixDistribution* m_dist;

  SolutionInfo(eConvCrit conv_crit, double conv_crit_value,
               const std::string& solver_comment, Alien::MatrixDistribution* dist = nullptr)
  : m_conv_crit(conv_crit)
  , m_conv_crit_value(conv_crit_value)
  , m_solver_comment(solver_comment)
  , m_dist(dist)
  {}
};

} // namespace Alien
