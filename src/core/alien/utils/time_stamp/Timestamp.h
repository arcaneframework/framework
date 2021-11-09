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

class TimestampMng;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ALIEN_EXPORT Timestamp
{
 public:
  //! Constructeur de la classe
  /*! Si @name manager est nul, il n'y a pas de gestion de timestamp */
  Timestamp(const TimestampMng* manager);

  //! Destructeur de la classe
  virtual ~Timestamp() {}

 public:
  //! Donne la valeur courante du timestamp
  virtual Int64 timestamp() const;

  //! Met à jour le timestamp
  /*! La politique actuelle position ce timestamp comme le plus 'à jour' de ceux associés
   * aux même manager */
  void updateTimestamp();

  //! Copy un autre timestamp
  void copyTimestamp(const Timestamp& v);

 public:
  //! Méthode interne de changement de valeur par le manager
  /*! Le paramètre @name manager permet de garantir que l'identité du manager effectuant
   *  cette requête de modification et de garantir l'intégrité des objets */
  void setTimestamp(const TimestampMng* manager, const Int64 timestamp);

 private:
  Int64 m_timestamp;
  const TimestampMng* m_manager;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
