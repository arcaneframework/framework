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

//
// Created by chevalier on 15/03/19.
//

#include "Environment.h"

#include <arccore/message_passing_mpi/StandaloneMpiMessagePassingMng.h>

namespace AlienTest
{

void Environment::initialize(int argc, char** argv)
{
  MPI_Init(&argc, &argv);
}

void Environment::finalize()
{
  MPI_Finalize();
}

Arccore::MessagePassing::IMessagePassingMng*
Environment::parallelMng()
{
  return Arccore::MessagePassing::Mpi::StandaloneMpiMessagePassingMng::create(
  MPI_COMM_WORLD);
}

Arccore::ITraceMng*
Environment::traceMng()
{
  return Arccore::arccoreCreateDefaultTraceMng();
}

} // namespace AlienTest