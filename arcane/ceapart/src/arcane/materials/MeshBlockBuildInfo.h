// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshBlockBuildInfo.h                                        (C) 2000-2013 */
/*                                                                           */
/* Informations pour la création d'un bloc.                                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MATERIALS_MESHBLOCKBUILDINFO_H
#define ARCANE_MATERIALS_MESHBLOCKBUILDINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/String.h"
#include "arcane/utils/Array.h"
#include "arcane/ItemGroup.h"
#include "arcane/materials/MaterialsGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MATERIALS_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IMeshMaterialMng;
class IMeshEnvironment;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneMaterials
 * \brief Informations pour la création d'un bloc.
 *
 * Cette instance contient les infos nécessaire à la création d'un bloc.
 *
 * Pour plus d'infos, se reporter à IMeshBlock.
 *

 * Une fois les informations spécifiées de création spécifiées, il faut créer le bloc
 * via IMeshMaterialMng::createBlock().
 */
class ARCANE_MATERIALS_EXPORT MeshBlockBuildInfo
{
 public:

  //! Créé les informations pour un bloc de nom \a name sur les mailles \a cells.
  MeshBlockBuildInfo(const String& name,const CellGroup& cells);
  ~MeshBlockBuildInfo();

 public:

  //! Nom du bloc
  const String& name() const { return m_name; }

  //! Liste des entités du bloc
  const CellGroup& cells() const { return m_cells; }

  /*!
   * \brief Ajoute le milieu \a env au bloc
   *
   * Le milieu doit déjà avoir été créé via
   * IMeshMaterialMng::createEnvironment().
   */
  void addEnvironment(IMeshEnvironment* env);

 public:

  //! Liste des milieux du bloc.
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

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MATERIALS_END_NAMESPACE
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

