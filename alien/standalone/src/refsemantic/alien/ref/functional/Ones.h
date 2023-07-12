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

#include <alien/core/block/Block.h>
#include <alien/core/block/VBlock.h>
#include <alien/ref/AlienRefSemanticPrecomp.h>
#include <alien/utils/Precomp.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class Vector;
class BlockVector;
class VBlockVector;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ALIEN_REFSEMANTIC_EXPORT extern Vector ones(
Arccore::Integer size, IMessagePassingMng* pm);

ALIEN_REFSEMANTIC_EXPORT extern BlockVector ones(
Integer size, const Block& block, IMessagePassingMng* pm);

ALIEN_REFSEMANTIC_EXPORT extern VBlockVector ones(
Integer size, const VBlock& block, IMessagePassingMng* pm);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
