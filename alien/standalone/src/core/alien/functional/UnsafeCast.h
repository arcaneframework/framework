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

#include <type_traits>

#include <alien/data/IMatrix.h>
#include <alien/data/IVector.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename T>
T& unsafeCast(IMatrix& M)
{
  static_assert(std::is_base_of<IMatrix, T>::value, "IMatrix is not a valid base");
  return static_cast<T&>(M);
}

/*---------------------------------------------------------------------------*/

template <typename T>
const T&
unsafeCast(const IMatrix& M)
{
  static_assert(std::is_base_of<IMatrix, T>::value, "IMatrix is not a valid base");
  return static_cast<const T&>(M);
}

/*---------------------------------------------------------------------------*/

template <typename T>
T& unsafeCast(IVector& M)
{
  static_assert(std::is_base_of<IVector, T>::value, "IVector is not a valid base");
  return static_cast<T&>(M);
}

/*---------------------------------------------------------------------------*/

template <typename T>
const T&
unsafeCast(const IVector& M)
{
  static_assert(std::is_base_of<IVector, T>::value, "IVector is not a valid base");
  return static_cast<const T&>(M);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
