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

#include <alien/utils/ObjectWithTrace.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename T>
void alien_debug(T&& t)
{
#ifndef NDEBUG
  if (Universe().traceMng() == nullptr)
    return;
  ObjectWithTrace().alien_debug(std::move(t));
#endif
}

/*---------------------------------------------------------------------------*/

template <typename T>
void alien_debug(bool flag, T&& t)
{
#ifndef NDEBUG
  if (not flag)
    return;
  if (Universe().traceMng() == nullptr)
    return;
  ObjectWithTrace().alien_debug(std::move(t));
#endif
}

/*---------------------------------------------------------------------------*/

template <typename T>
void alien_info(T&& t)
{
  if (Universe().traceMng() == nullptr)
    return;
  ObjectWithTrace().alien_info(std::move(t));
}

/*---------------------------------------------------------------------------*/

template <typename T>
void alien_info(bool flag, T&& t)
{
  if (not flag)
    return;
  if (Universe().traceMng() == nullptr)
    return;
  ObjectWithTrace().alien_info(std::move(t));
}

/*---------------------------------------------------------------------------*/

template <typename T>
void alien_fatal(T&& t)
{
  if (Universe().traceMng() == nullptr)
    return;
  ObjectWithTrace().alien_fatal(std::move(t));
}

/*---------------------------------------------------------------------------*/

template <typename T>
void alien_fatal(bool flag, T&& t)
{
  if (not flag)
    return;
  if (Universe().traceMng() == nullptr)
    return;
  ObjectWithTrace().alien_fatal(std::move(t));
}

/*---------------------------------------------------------------------------*/

template <typename T>
void alien_warning(T&& t)
{
  if (Universe().traceMng() == nullptr)
    return;
  ObjectWithTrace().alien_warning(std::move(t));
}

/*---------------------------------------------------------------------------*/

template <typename T>
void alien_warning(bool flag, T&& t)
{
  if (not flag)
    return;
  if (Universe().traceMng() == nullptr)
    return;
  ObjectWithTrace().alien_warning(std::move(t));
}

/*---------------------------------------------------------------------------*/

inline TraceMessage
cout()
{
  return Universe().traceMng()->info();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
