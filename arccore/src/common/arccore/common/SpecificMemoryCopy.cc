// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SpecificMemoryCopy.cc                                       (C) 2000-2025 */
/*                                                                           */
/* Classes pour gérer des fonctions spécialisées de copie mémoire.           */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/internal/SpecificMemoryCopyList.h"
#include "arccore/common/internal/HostSpecificMemoryCopy.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::impl
{

ISpecificMemoryCopyList* GlobalMemoryCopyList::default_global_copy_list = nullptr;
ISpecificMemoryCopyList* GlobalMemoryCopyList::accelerator_global_copy_list = nullptr;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class HostIndexedCopyTraits
{
 public:

  using InterfaceType = ISpecificMemoryCopy;
  template <typename DataType, typename Extent> using SpecificType = HostSpecificMemoryCopy<DataType, Extent>;
  using RefType = SpecificMemoryCopyRef<HostIndexedCopyTraits>;
};

namespace
{
  // Copier spécifique lorsqu'aucun runtime accélérateur n'est initialisé
  impl::SpecificMemoryCopyList<impl::HostIndexedCopyTraits> global_host_copy_list;
} // namespace

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void GlobalMemoryCopyList::
setAcceleratorInstance(ISpecificMemoryCopyList* ptr)
{
  if (!default_global_copy_list) {
    default_global_copy_list = ptr;
  }
  accelerator_global_copy_list = ptr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ISpecificMemoryCopyList* GlobalMemoryCopyList::
getDefault(const RunQueue* queue)
{
  if (queue && !default_global_copy_list)
    ARCCORE_FATAL("No instance of copier is available for RunQueue");
  if (default_global_copy_list && queue)
    return default_global_copy_list;
  return &global_host_copy_list;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
