// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IItemEnumeratorTracer.h                                     (C) 2000-2025 */
/*                                                                           */
/* Interface de trace des appels aux énumérateur sur les entités.            */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IITEMENUMERATORTRACER_H
#define ARCANE_CORE_IITEMENUMERATORTRACER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ItemTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Interface d'un traceur d'énumérateur sur les entités.
 *
 * Cette interface fournit des méthodes qui sont appelées automatiquement
 * lors de l'utilisation des macros permettant d'itérer sur les entités
 * comme ENUMERATE_CELL ou ENUMERATE_SIMD_CELL. Pour des raisons de performance,
 * ces macros ne sont tracées  que si le fichier source qui les utilise est
 * compilé avec la macro ARCANE_TRACE_ENUMERATOR.
 *
 * La méthode singleton() permet de récupérer l'implémentation actuelle.
 *
 */
class ARCANE_CORE_EXPORT IItemEnumeratorTracer
{
 public:

  static IItemEnumeratorTracer* singleton();

 public:

  virtual ~IItemEnumeratorTracer() = default;

 public:

  //! Méthode appelée lors avant d'exécuter un ENUMERATE_
  virtual void enterEnumerator(const ItemEnumerator& e, EnumeratorTraceInfo& eti) = 0;

  //! Méthode appelée lors après l'exécution d'un ENUMERATE_
  virtual void exitEnumerator(const ItemEnumerator& e, EnumeratorTraceInfo& eti) = 0;

  //! Méthode appelée lors avant d'exécuter un ENUMERATE_SIMD_
  virtual void enterEnumerator(const SimdItemEnumeratorBase& e, EnumeratorTraceInfo& eti) = 0;

  //! Méthode appelée lors après l'exécution d'un ENUMERATE_SIMD_
  virtual void exitEnumerator(const SimdItemEnumeratorBase& e, EnumeratorTraceInfo& eti) = 0;

 public:

  virtual void dumpStats() = 0;
  virtual IPerformanceCounterService* perfCounter() = 0;
  virtual Ref<IPerformanceCounterService> perfCounterRef() = 0;

 public:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
