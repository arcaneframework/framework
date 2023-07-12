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

#include <alien/data/Universe.h>
#include <alien/utils/Precomp.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*
 * Objet permettant les traces utilisateurs.
 * Utilisation via section de trace definies par lambda
 * Le but est de limiter les problèmes et tests de traces si ITraceMng est
 * nullptr De plus, les actions d'écritures sont atomiques. Il est facile d'améliorer
 * l'implémentation pour les packer ou pour gérer le multithreading.
 *
 * Activation des traces d'Alien:
 *
 * Alien::ITraceMng* trace = ...
 * Alien::setTraceMng(trace);
 *
 * Niveau de verbosite:
 *
 * Alien::setVerbosity(Alien::Verbosity::Debug);
 *
 * Utilisation:
 *
 * alien_debug([&] {
 *   ... message à imprimer ...
 *   ... on peut utiliser debug() << ... ou info(), warning(), etc.
 *   ... En pratique, utiliser cout() de preference ...
 * });
 *
 *
 */

class ObjectWithTrace
{
 public:
  ObjectWithTrace()
  : m_is_locked(true)
  {}

  virtual ~ObjectWithTrace() {}

 public:
  template <typename T>
  void alien_info(T&& t) const
  {
    _print(Verbosity::Info, std::move(t));
  }

  template <typename T>
  void alien_debug(T&& t) const
  {
    _print(Verbosity::Debug, std::move(t));
  }

  template <typename T>
  void alien_warning(T&& t) const
  {
    _print(Verbosity::Warning, std::move(t));
  }

  template <typename T>
  void alien_fatal(T&& t) const
  {
    auto* trace = traceMng();
    if (trace != nullptr) {
      m_is_locked = false;
      t();
      m_is_locked = true;
    }
    throw FatalErrorException(
    A_FUNCINFO, "Fatal error in Alien - for more details, increase verbosity level");
  }

 protected:
  // Trace a utiliser dans les sections de trace
  TraceMessage cout() const
  {
    _checkLock();
    return traceMng()->info();
  }

 private:
  void _checkLock() const
  {
    if (m_is_locked) {
      throw FatalErrorException("Trace in Alien should be done using trace section");
    }
  }

  template <typename T>
  void _print(Verbosity::Level N, T&& t) const
  {
    auto* trace = traceMng();
    if (trace == nullptr)
      return;
    m_is_locked = false;
    if (Universe().verbosityLevel() <= N) {
      t();
    }
    m_is_locked = true;
  }

 public:
  ITraceMng* traceMng() const { return Universe().traceMng(); }

 private:
  // NB: on ne stocke pas le ITraceMng* car on ne sait pas s'il sera
  // positionne apres la construction de l'objet...

  // Permet de bloquer pour forcer l'affichage via les fonctions de print
  mutable bool m_is_locked;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
