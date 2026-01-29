// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SpecificMemoryCopy.cc                                       (C) 2000-2026 */
/*                                                                           */
/* Classes pour gérer des fonctions spécialisées de copie mémoire.           */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/internal/SpecificMemoryCopyList.h"
#include "arccore/common/internal/HostSpecificMemoryCopy.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Impl
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

//! Copieur spécifique lorsqu'aucun runtime accélérateur n'est initialisé
class HostSpecificMemoryCopyList
: public SpecificMemoryCopyList<Impl::HostIndexedCopyTraits>
{
 public:

  HostSpecificMemoryCopyList()
  {
    using impl::ExtentValue;
    //! Ajoute des implémentations spécifiques pour les tailles courantes
    addCopier<SpecificType<std::byte, ExtentValue<1>>>(); // 1
    addCopier<SpecificType<Int16, ExtentValue<1>>>(); // 2
    addCopier<SpecificType<std::byte, ExtentValue<3>>>(); // 3
    addCopier<SpecificType<Int32, ExtentValue<1>>>(); // 4
    addCopier<SpecificType<std::byte, ExtentValue<5>>>(); // 5
    addCopier<SpecificType<Int16, ExtentValue<3>>>(); // 6
    addCopier<SpecificType<std::byte, ExtentValue<7>>>(); // 7
    addCopier<SpecificType<Int64, ExtentValue<1>>>(); // 8
    addCopier<SpecificType<std::byte, ExtentValue<9>>>(); // 9
    addCopier<SpecificType<Int16, ExtentValue<5>>>(); // 10
    addCopier<SpecificType<Int32, ExtentValue<3>>>(); // 12

    addCopier<SpecificType<Int64, ExtentValue<2>>>(); // 16
    addCopier<SpecificType<Int64, ExtentValue<3>>>(); // 24
    addCopier<SpecificType<Int64, ExtentValue<4>>>(); // 32
    addCopier<SpecificType<Int64, ExtentValue<5>>>(); // 40
    addCopier<SpecificType<Int64, ExtentValue<6>>>(); // 48
    addCopier<SpecificType<Int64, ExtentValue<7>>>(); // 56
    addCopier<SpecificType<Int64, ExtentValue<8>>>(); // 64
    addCopier<SpecificType<Int64, ExtentValue<9>>>(); // 72
  }
};

namespace
{
  // Copieur spécifique lorsqu'aucun runtime accélérateur n'est initialisé
  HostSpecificMemoryCopyList global_host_copy_list;
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
