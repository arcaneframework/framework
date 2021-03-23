// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshBlock.h                                                 (C) 2000-2016 */
/*                                                                           */
/* Bloc d'un maillage.                                                       */
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
 * \brief Bloc d'un maillage.
 *
 * Cette classe est à usage interne à Arcane et ne doit pas être utilisée
 * explicitement. Il faut utiliser l'interface IMeshBlock pour accéder
 * aux milieux.
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

  //! Fonctions publiques mais réservées au IMeshMaterialMng
  //@{
  void build();
  void addEnvironment(IMeshEnvironment* env);
  void removeEnvironment(IMeshEnvironment* env);
  //@}

 private:

  //! Gestionnaire de matériaux
  IMeshMaterialMng* m_material_mng;
  
  //! Identifiant du milieu (indice de ce milieu dans la liste des milieux)
  Int32 m_block_id;

  //! Nom du milieu
  String m_name;

  //! Liste des mailles de ce milieu
  CellGroup m_cells;

  //! Liste des milieux de ce bloc.
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

