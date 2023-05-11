// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* PointerAttribute.h                                          (C) 2000-2023 */
/*                                                                           */
/* Informations sur une adresse mémoire.                                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ACCELERATOR_CORE_POINTERATTRIBUTE_H
#define ARCANE_ACCELERATOR_CORE_POINTERATTRIBUTE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/String.h"
#include "arcane/accelerator/core/DeviceId.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Informations sur une adresse mémoire.
 *
 * Les informations de cette instance sont équivalentes à celles obtenues
 * par l'appel à cudaPointerGetAttributes()
 */
class ARCANE_ACCELERATOR_CORE_EXPORT PointerAttribute
{
  // Seule cette classe peut construire une instance
  friend impl::IRunnerRuntime;

 public:
 public:

  PointerAttribute() = default;

 public:

  ePointerMemoryType memoryType() const { return m_memory_type; }
  const void* originalPointer() const { return m_pointer; }
  const void* hostPointer() const { return m_host_pointer; }
  const void* devicePointer() const { return m_device_pointer; }
  int device() const { return m_device; }

 private:

  PointerAttribute(const void* pointer)
  : m_memory_type(ePointerMemoryType::Host)
  , m_pointer(pointer)
  , m_host_pointer(pointer)
  {
  }

  PointerAttribute(ePointerMemoryType mem_type, int device, const void* pointer,
                   const void* device_pointer, const void* host_pointer)
  : m_memory_type(mem_type)
  , m_device(device)
  , m_pointer(pointer)
  , m_device_pointer(device_pointer)
  , m_host_pointer(host_pointer)
  {}

 private:

  ePointerMemoryType m_memory_type = ePointerMemoryType::Unregistered;
  int m_device = (-1);
  const void* m_pointer = nullptr;
  const void* m_device_pointer = nullptr;
  const void* m_host_pointer = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
