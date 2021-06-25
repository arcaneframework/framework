// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2020 IFPEN-CEA
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IStackTraceService.h                                        (C) 2000-2018 */
/*                                                                           */
/* Interface d'un service de trace des appels de fonctions.                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_ISTACKTRACESERVICE_H
#define ARCCORE_BASE_ISTACKTRACESERVICE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/BaseTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Interface d'un service de trace des appels de fonctions.
 */
class ARCCORE_BASE_EXPORT IStackTraceService
{
 public:

  virtual ~IStackTraceService() {} //<! Libère les ressources

 public:

  virtual void build() =0;

 public:

  /*! \brief Chaîne de caractère indiquant la pile d'appel.
   *
   * \a first_function indique le numéro dans la pile de la première fonction
   * affichée dans la trace.
   */
  virtual StackTrace stackTrace(int first_function=0) =0;

  /*!
   * \brief Nom d'une fonction dans la pile d'appel.
   *
   * \a function_index indique la position de la fonction à retourner dans la
   * pile d'appel. Par exemple, 0 indique la fonction courante, 1 celle
   * d'avant (donc celle qui appelle cette méthode).
   */
  virtual StackTrace stackTraceFunction(int function_index) =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

