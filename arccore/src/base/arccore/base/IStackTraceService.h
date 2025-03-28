// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IStackTraceService.h                                        (C) 2000-2025 */
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

namespace Arcane
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

