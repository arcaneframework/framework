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
/*
 * AlienKrylov.h
 *
 *  Created on: Dec 22, 2021
 *      Author: gratienj
 */

#pragma once

#include <alien/expression/krylov/Iteration.h>
#include <alien/expression/krylov/CG.h>
#include <alien/expression/krylov/BiCGStab.h>
#include <alien/expression/krylov/DiagPreconditioner.h>
#include <alien/expression/krylov/ChebyshevPreconditioner.h>
#include <alien/expression/krylov/NeumannPolyPreconditioner.h>
#include <alien/expression/krylov/ILU0Preconditioner.h>
#include <alien/expression/krylov/FILU0Preconditioner.h>
