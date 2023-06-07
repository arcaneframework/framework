// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshBuildInfo.h                                             (C) 2000-2023 */
/*                                                                           */
/* Informations pour construire un maillage.                                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESHBUILDINFO_H
#define ARCANE_MESHBUILDINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/String.h"

#include "arcane/core/IParallelMng.h"
#include "arcane/core/ItemGroup.h"
#include "arcane/core/MeshKind.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Paramètres nécessaires à la construction d'un maillage.
 *
 * Seul le nom (name()) est indispensable. Les autres paramètres sont
 * optionnels suivant le type de création qu'on souhaite. S'il ne sont pas
 * définit et utiles, une valeur par défaut sera utilisée.
 *
 * Il existe deux possibilités de création:
 *
 * - création d'un nouveau maillage. Dans ce cas il est nécessaire de spécifier
 *   le \a IParallelMng associé via addParallelMng().
 * - création d'un sous-maillage d'un maillage existant. Dans ce cas il
 *   est nécessaire de positionner le groupe via addItemGroup(). Le
 *   IParallelMng associé est nécessairement celui du maillage parent. Le
 *   sous-maillage créé comprendra les mailles de ce groupe.
 */
class ARCANE_CORE_EXPORT MeshBuildInfo
{
 public:

  /*!
   * \brief Construit un maillage par défaut avec pour nom \a name.
   */
  explicit MeshBuildInfo(const String& name);

 public:

  //! Positionne le nom de la fabrique pour créer ce maillage
  MeshBuildInfo& addFactoryName(const String& factory_name);
  //! Positionne le gestionnaire de parallélisme pour créér la maillage
  MeshBuildInfo& addParallelMng(Ref<IParallelMng> pm);
  //! Positionne le groupe de mailles pour un sous-maillage
  MeshBuildInfo& addParentGroup(const ItemGroup& parent_group);
  /*!
  * \brief Indique si le générateur nécessite d'appeler un partitionneur.
  *
  * C'est le cas par exemple si le lecteur ne sait générer que des maillages
  * séquentiels.
  */
  MeshBuildInfo& addNeedPartitioning(bool v);

  //! Positionne les caractéristiques du maillage
  MeshKind& addMeshKind(const MeshKind& v);

  //! Nom du nouveau maillage
  const String& name() const { return m_name; }
  //! Nom de la fabrique pour créer le maillage (via IMeshFactory)
  const String& factoryName() const { return m_factory_name; }
  //! Gestionnaire de parallélisme dans le cas d'un nouveau maillage.
  Ref<IParallelMng> parallelMngRef() const { return m_parallel_mng; }
  //! Groupe parent dans le cas d'un sous-maillage, null sinon.
  const ItemGroup& parentGroup() const { return m_parent_group; }
  //! Indique si le lecteur/générateur nécessite un partitionnement
  bool isNeedPartitioning() const { return m_is_need_partitioning; }
  //! Caractéristiques du maillage
  const MeshKind meshKind() const { return m_mesh_kind; }

 private:

  String m_name;
  String m_factory_name;
  Ref<IParallelMng> m_parallel_mng;
  ItemGroup m_parent_group;
  bool m_is_need_partitioning = false;
  MeshKind m_mesh_kind;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
