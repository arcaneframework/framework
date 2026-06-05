// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMeshMaterialVariableSynchronizer.h                         (C) 2000-2023 */
/*                                                                           */
/* Interface of the material variable synchronizer.                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MATERIALS_INTERNAL_IMESHMATERIALVARIABLESYNCHRONIZER_H
#define ARCANE_MATERIALS_INTERNAL_IMESHMATERIALVARIABLESYNCHRONIZER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/materials/MaterialsGlobal.h"

#include "arcane/utils/ArrayView.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class IVariableSynchronizer;
}

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Interface of the material variable synchronizer.
 */
class ARCANE_MATERIALS_EXPORT IMeshMaterialVariableSynchronizer
{
 public:

  virtual ~IMeshMaterialVariableSynchronizer(){}

 public:

  //! Associated classical variable synchronizer.
  virtual IVariableSynchronizer* variableSynchronizer() =0;

  /*!
   * \brief List of shared MatVarIndex for index rank \a index
   * in the variableSynchronizer::communicatingRanks() array;
   */
  virtual ConstArrayView<MatVarIndex> sharedItems(Int32 index) =0;

  /*!
   * \brief List of ghost MatVarIndex for index rank \a index
   * in the variableSynchronizer::communicatingRanks() array;
   */
  virtual ConstArrayView<MatVarIndex> ghostItems(Int32 index) =0;

  //! Recalculates synchronization information.
  virtual void recompute() =0;

  //! Recalculates synchronization information if necessary.
  virtual void checkRecompute() =0;

  //! Common buffer for messages.
  virtual Ref<IMeshMaterialSynchronizeBuffer> commonBuffer() =0;

  //! Memory resource to use for communication buffers
  virtual eMemoryRessource bufferMemoryRessource() const =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
