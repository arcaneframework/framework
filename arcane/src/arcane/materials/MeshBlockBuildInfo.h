// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshBlockBuildInfo.h                                        (C) 2000-2013 */
/*                                                                           */
/* Information for the creation of a block.                                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MATERIALS_MESHBLOCKBUILDINFO_H
#define ARCANE_MATERIALS_MESHBLOCKBUILDINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/String.h"
#include "arcane/utils/Array.h"
#include "arcane/core/ItemGroup.h"
#include "arcane/materials/MaterialsGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IMeshMaterialMng;
class IMeshEnvironment;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup ArcaneMaterials
 * \brief Brief: Information for the creation of a block.
 *
 * This instance contains the necessary information for the creation of a block.
 *
 * For more information, refer to IMeshBlock.
 *

 * Once the specified creation information is provided, the block must be created
 * via IMeshMaterialMng::createBlock().
 */
class ARCANE_MATERIALS_EXPORT MeshBlockBuildInfo
{
 public:

  //! Creates the information for a block named \a name on the cells \a cells.
  MeshBlockBuildInfo(const String& name, const CellGroup& cells);
  ~MeshBlockBuildInfo();

 public:

  //! Block name
  const String& name() const { return m_name; }

  //! List of block entities
  const CellGroup& cells() const { return m_cells; }

  /*!
   * \brief Brief: Adds the environment \a env to the block
   *
   * The environment must already have been created via
   * IMeshMaterialMng::createEnvironment().
   */
  void addEnvironment(IMeshEnvironment* env);

 public:

  //! List of environments in the block.
  ConstArrayView<IMeshEnvironment*> environments() const
  {
    return m_environments;
  }

 private:

  String m_name;
  CellGroup m_cells;
  UniqueArray<IMeshEnvironment*> m_environments;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
