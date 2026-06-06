// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMeshComponentInternal.h                                    (C) 2000-2024 */
/*                                                                           */
/* Arcane internal API for 'IMeshComponent'.                                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_MATERIALS_INTERNAL_IMESHCOMPONENTINTERNAL_H
#define ARCANE_CORE_MATERIALS_INTERNAL_IMESHCOMPONENTINTERNAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ItemTypes.h"
#include "arcane/core/materials/MaterialsCoreGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Arcane internal API for 'IMeshComponent'.
 */
class ARCANE_CORE_EXPORT IMeshComponentInternal
{
 public:

  virtual ~IMeshComponentInternal() = default;

 public:

  //! Indexer to access partial variables.
  virtual MeshMaterialVariableIndexer* variableIndexer() const =0;

  //! View of the constituent meshes.
  virtual ConstituentItemLocalIdListView constituentItemListView() const =0;

  //! Index to access partial variables.
  virtual Int32 variableIndexerIndex() const =0;

  //! Create an instance of the 'ConstituentItemVectorImpl' implementation
  virtual Ref<IConstituentItemVectorImpl> createItemVectorImpl() const =0;

  //! Create an instance of the 'ConstituentItemVectorImpl' implementation
  virtual Ref<IConstituentItemVectorImpl> createItemVectorImpl(ComponentItemVectorView rhs) const =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
