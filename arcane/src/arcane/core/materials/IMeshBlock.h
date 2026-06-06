// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMeshBlock.h                                                (C) 2000-2025 */
/*                                                                           */
/* Interface of a mesh block.                                                */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_MATERIALS_IMESHBLOCK_H
#define ARCANE_CORE_MATERIALS_IMESHBLOCK_H
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
 * \ingroup ArcaneMaterials
 * \brief Interface of a mesh block.
 * 
 * Blocks are created via IMeshMaterialMng::createBlock().
 *
 * Blocks cannot be destroyed and must be created during
 * initialization.
 *
 * The concept of a block is optional and it is not necessary to have
 * blocks to use environments and materials.
 *
 * A block is characterized by a name (name()), a cell group (cells()),
 * and a list of environments (environments()).
 *
 * Note that theoretically the cell group (cells())
 * is independent of the list of environments, but for reasons of
 * consistency, it is preferable that this group corresponds to the union of
 * the block's environments. However, no verification of this consistency is performed.
 *
 * It is possible to use an instance of this class as an argument to
 * ENUMERATE_ENV or to ENUMERATE_ALLENVCELL.
 */
class ARCANE_CORE_EXPORT IMeshBlock
{
 public:

  virtual ~IMeshBlock() {}

 public:

  //! Associated manager.
  virtual IMeshMaterialMng* materialMng() = 0;

  //! Block name
  virtual const String& name() const = 0;

  /*!
   * \brief Cell group of this block.
   */
  virtual const CellGroup& cells() const = 0;

  //! List of environments in this block
  virtual ConstArrayView<IMeshEnvironment*> environments() = 0;

  //! Number of environments in the block
  virtual Integer nbEnvironment() const = 0;

  /*!
   * \brief Block identifier.
   * It is also the index (starting from 0) of this block
   * in the list of blocks.
   */
  virtual Int32 id() const = 0;

  //! View of the environments cells corresponding to this block.
  virtual AllEnvCellVectorView view() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
