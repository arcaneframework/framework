// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* PointerAttribute.h                                          (C) 2000-2025 */
/*                                                                           */
/* Memory address information.                                               */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COMMON_ACCELERATOR_POINTERATTRIBUTE_H
#define ARCCORE_COMMON_ACCELERATOR_POINTERATTRIBUTE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/String.h"
#include "arccore/common/accelerator/DeviceId.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Information about a memory address.
 *
 * The information in this instance is equivalent to that obtained
 * by calling cudaPointerGetAttributes().
 * The information is only valid if isValid() is true.
 */
class ARCCORE_COMMON_EXPORT PointerAttribute
{
  // Only this class can construct an instance
  friend Impl::IRunnerRuntime;

 public:

  PointerAttribute() = default;

 public:

  bool isValid() const { return m_is_valid; }
  ePointerMemoryType memoryType() const { return m_memory_type; }
  const void* originalPointer() const { return m_pointer; }
  const void* hostPointer() const { return m_host_pointer; }
  const void* devicePointer() const { return m_device_pointer; }
  int device() const { return m_device; }
  friend std::ostream& operator<<(std::ostream& o, const PointerAttribute& a);

 private:

  //! Constructor indicating that there is no information about the memory region
  PointerAttribute(const void* pointer)
  : m_pointer(pointer)
  , m_is_valid(false)
  {
  }

  PointerAttribute(ePointerMemoryType mem_type, int device, const void* pointer,
                   const void* device_pointer, const void* host_pointer)
  : m_memory_type(mem_type)
  , m_device(device)
  , m_pointer(pointer)
  , m_device_pointer(device_pointer)
  , m_host_pointer(host_pointer)
  , m_is_valid(true)
  {}

 private:

  ePointerMemoryType m_memory_type = ePointerMemoryType::Unregistered;
  int m_device = (-1);
  const void* m_pointer = nullptr;
  const void* m_device_pointer = nullptr;
  const void* m_host_pointer = nullptr;
  bool m_is_valid = false;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
