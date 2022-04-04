// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IReduceMemoryImpl.h                                         (C) 2000-2021 */
/*                                                                           */
/* Interface de la gestion mémoire pour les réductions.                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ACCELERATOR_IREDUCEMEMORYIMPL_H
#define ARCANE_ACCELERATOR_IREDUCEMEMORYIMPL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/accelerator/core/AcceleratorCoreGlobal.h"

#include <stack>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Interface de la gestion mémoire pour les réductions.
 * \warning API en cours de définition.
 */
class ARCANE_ACCELERATOR_CORE_EXPORT IReduceMemoryImpl
{
 public:
  virtual ~IReduceMemoryImpl() = default;
 public:
  virtual void* allocateMemory(Int64 size) = 0;
  virtual void release() =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename T> T*
allocateReduceMemory(IReduceMemoryImpl* p)
{
  return reinterpret_cast<T*>(p->allocateMemory(sizeof(T)));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Accelerator::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
