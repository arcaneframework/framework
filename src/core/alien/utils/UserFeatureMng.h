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

#include <set>

#include <alien/utils/Precomp.h>
#include <arccore/base/String.h>

/*
 * Outil à destination des développeurs de noyaux externes
 *
 * Outillage utilisé pour ajouter de l'information utilisateur
 * à des objets Alien:
 * Typiquement, IMatrixImpl et IMatrixImpl possède un tel gestionnaire
 * Ainsi, si un utilisateur développant un noyau ou une api souhaite
 * ajouter des informations spécifiques pour un traitement dans un
 * solveur spécifique, c'est possible et Alien reste extensible
 * sans toucher au coeur.
 *
 * Passage d'information de type String, ie faible robustesse, peu d'aide
 * possible du noyau Alien
 *
 * !! Ne doit pas se substituer à la paramétrisation des solveurs d'Alien !!
 *
 * Si des features sont communes à plusieurs solveurs, il conviendra d'
 * intégrer les features dans l'API pour amener la robustesse nécessaire
 *
 */

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class UserFeatureMng
{
 public:
  UserFeatureMng() {}

  virtual ~UserFeatureMng() {}

  bool hasFeature(const String& feature) const
  {
    return m_features.find(feature) != m_features.end();
  }

  void setFeature(const String& feature) { m_features.insert(feature); }

 private:
  std::set<String> m_features;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
