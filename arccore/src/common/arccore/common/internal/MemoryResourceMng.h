// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MemoryResourceMng.h                                         (C) 2000-2025 */
/*                                                                           */
/* Gestion des ressources mémoire pour les CPU et accélérateurs.             */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COMMON_INTERNAL_MEMORYRESOURCEMNG_H
#define ARCCORE_COMMON_INTERNAL_MEMORYRESOURCEMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/FixedArray.h"

#include "arccore/common/IMemoryResourceMng.h"
#include "arccore/common/internal/IMemoryResourceMngInternal.h"

#include <memory>
#include <array>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Gestion des ressources mémoire pour les CPU et accélérateurs.
 */
class ARCCORE_COMMON_EXPORT MemoryResourceMng
: public IMemoryResourceMng
, public IMemoryResourceMngInternal
{
 public:

  MemoryResourceMng();

 public:

  IMemoryAllocator* getAllocator(eMemoryResource r) override;
  IMemoryAllocator* getAllocator(eMemoryResource r, bool throw_if_not_found) override;

 public:

  void copy(ConstMemoryView from, eMemoryResource from_mem,
            MutableMemoryView to, eMemoryResource to_mem, const RunQueue* queue) override;

 public:

  void setAllocator(eMemoryResource r, IMemoryAllocator* allocator) override;
  void setCopier(IMemoryCopier* copier) override { m_copier = copier; }
  void setIsAccelerator(bool v) override { m_is_accelerator = v; }

 public:

  //! Interface interne
  IMemoryResourceMngInternal* _internal() override { return this; }

 public:

  //! Copie générique utilisant platform::getDataMemoryRessourceMng()
  static void genericCopy(ConstMemoryView from, MutableMemoryView to);

 private:

  FixedArray<IMemoryAllocator*, ARCCORE_NB_MEMORY_RESOURCE> m_allocators;
  std::unique_ptr<IMemoryCopier> m_default_memory_copier;
  IMemoryCopier* m_copier = nullptr;
  bool m_is_accelerator = false;

 private:

  inline int _checkValidResource(eMemoryResource r);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
