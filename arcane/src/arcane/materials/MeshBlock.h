// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshBlock.h                                                 (C) 2000-2016 */
/*                                                                           */
/* Block of a mesh.                                                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MATERIALS_MESHBLOCK_H
#define ARCANE_MATERIALS_MESHBLOCK_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"

#include "arcane/ItemGroup.h"

#include "arcane/materials/IMeshBlock.h"
#include "arcane/materials/MeshBlockBuildInfo.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MATERIALS_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Block of a mesh.
 *
 * This class is for internal use in Arcane and should not be used
 * explicitly. The IMeshBlock interface must be used to access
 * the materials.
 */
class MeshBlock
: public TraceAccessor
, public IMeshBlock
{
 public:

  MeshBlock(IMeshMaterialMng* mm,Int32 block_id,const MeshBlockBuildInfo& infos);
  virtual ~MeshBlock(){}

 public:

  virtual IMeshMaterialMng* materialMng() { return m_material_mng; }
  virtual const String& name() const { return m_name; }
  virtual const CellGroup& cells() const { return m_cells; }
  virtual ConstArrayView<IMeshEnvironment*> environments()
  {
    return m_environments;
  }
  virtual Integer nbEnvironment() const
  {
    return m_environments.size();
  }
  virtual Int32 id() const
  {
    return m_block_id;
  }

  virtual AllEnvCellVectorView view();

 public:

  //! Public functions but reserved for IMeshMaterialMng
  //@{
  void build();
  void addEnvironment(IMeshEnvironment* env);
  void removeEnvironment(IMeshEnvironment* env);
  //@}

 private:

  //! Material manager
  IMeshMaterialMng* m_material_mng;
  
  //! Material identifier (index of this material in the list of materials)
  Int32 m_block_id;

  //! Material name
  String m_name;

  //! List of meshes for this material
  CellGroup m_cells;

  //! List of materials/environments in this block.
  UniqueArray<IMeshEnvironment*> m_environments;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MATERIALS_END_NAMESPACE
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
