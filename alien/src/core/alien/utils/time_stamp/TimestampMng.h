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

#include <alien/utils/Precomp.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class Timestamp;
class ITimestampObserver;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Gestionnaire de Timestamp
/*! Permet d'identifier les éléments à jour d'un point de vue global.
 *  Un TimestampMng est associé à un ensemble de Timestamp qui le connaisse et se réfère à
 *  lui pour la mise à jour de leur valeur 'à jour' */
class ALIEN_EXPORT TimestampMng
{
 public:
  TimestampMng();
  TimestampMng(const TimestampMng& tm);

 private:
  TimestampMng(TimestampMng&&) = delete;
  void operator=(const TimestampMng&) = delete;
  void operator=(TimestampMng&&) = delete;

 public:
  virtual ~TimestampMng() {}

 public:
  //! Valeur du timestamp de référence
  Int64 timestamp() const;

 public:
  //! Requête de mise à jour d'un Timestamp
  void updateTimestamp(Timestamp* ts) const;

  void addObserver(std::shared_ptr<ITimestampObserver> observer);

  void clearObservers();

 private:
  /*! Le modificateur mutable est une erreur de conception à corriger mais sans urgence */
  mutable Int64 m_timestamp;

  UniqueArray<std::shared_ptr<ITimestampObserver>> m_observers;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
