// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AllocatedMemoryInfo.h                                       (C) 2000-2025 */
/*                                                                           */
/* Informations sur une zone mémoire allouée.                                */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COMMON_ALLOCATEDMEMORYINFO_H
#define ARCCORE_COMMON_ALLOCATEDMEMORYINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/CommonGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Informations sur une zone mémoire allouée.
 */
class AllocatedMemoryInfo
{
 public:

  AllocatedMemoryInfo() = default;
  explicit AllocatedMemoryInfo(void* base_address)
  : m_base_address(base_address)
  {}
  AllocatedMemoryInfo(void* base_address, Int64 size)
  : m_base_address(base_address)
  , m_size(size)
  , m_capacity(size)
  {}
  AllocatedMemoryInfo(void* base_address, Int64 size, Int64 capacity)
  : m_base_address(base_address)
  , m_size(size)
  , m_capacity(capacity)
  {}

  //! Adresse du début de la zone allouée.
  void* baseAddress() const { return m_base_address; }
  //! Taille en octets de la zone mémoire utilisée. (-1) si inconnue
  Int64 size() const { return m_size; }
  //! Taille en octets de la zone mémoire allouée. (-1) si inconnue
  Int64 capacity() const { return m_capacity; }

 public:

  void* m_base_address = nullptr;
  Int64 m_size = -1;
  Int64 m_capacity = -1;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

