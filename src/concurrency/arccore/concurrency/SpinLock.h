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
/* SpinLock.h                                                  (C) 2000-2017 */
/*                                                                           */
/* SpinLock pour le multi-threading.                                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_CONCURRENCY_SPINLOCK_H
#define ARCCORE_CONCURRENCY_SPINLOCK_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/concurrency/IThreadImplementation.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class SpinLockImpl;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief SpinLock.
 */
class ARCCORE_CONCURRENCY_EXPORT SpinLock
{
 public:
  class ScopedLock
  {
   public:
    ScopedLock(SpinLock& sl)
    {
      spin_lock_addr = &sl.spin_lock;
      Concurrency::getThreadImplementation()->lockSpinLock(spin_lock_addr,scoped_spin_lock);
    }
    ~ScopedLock()
    {
      Concurrency::getThreadImplementation()->unlockSpinLock(spin_lock_addr,scoped_spin_lock);
    }
   private:
    Int64* spin_lock_addr;
    Int64 scoped_spin_lock[2];
  };

  class ManualLock
  {
   public:
    void lock(SpinLock& sl)
    {
      Concurrency::getThreadImplementation()->lockSpinLock(&sl.spin_lock,scoped_spin_lock);
    }
    void unlock(SpinLock& sl)
    {
      Concurrency::getThreadImplementation()->unlockSpinLock(&sl.spin_lock,scoped_spin_lock);
    }
   private:
    Int64 scoped_spin_lock[2];
  };

  friend class ScopedLock;
  friend class ManualLock;

 public:
  SpinLock()
  {
    Concurrency::getThreadImplementation()->createSpinLock(&spin_lock);
  }
 private:
  Int64 spin_lock;
 private:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
