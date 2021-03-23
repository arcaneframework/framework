// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ModuleBuildInfo.h                                           (C) 2000-2019 */
/*                                                                           */
/* Paramètres pour construire un module.                                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MODULEBUILDINFO_H
#define ARCANE_MODULEBUILDINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/String.h"
#include "arcane/MeshHandle.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ISubDomain;
class IMesh;
class IGraph;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Informations pour construire un module.
 *
 * \a ModuleBuildInfo est usuellement utilisé via \a BasicModule
 * (module basique) et \a AbstractModule (tout module) pour la
 * création des différents modules.
 * 
 * \ingroup Module
 */
class ARCANE_CORE_EXPORT ModuleBuildInfo
{
 public:
  //! Constructeur à partir d'un sous-domaine, un maillage et un nom d'implémentation de module
  ModuleBuildInfo(ISubDomain* sd,IMesh* mesh,const String& name);
  //! Constructeur à partir d'un sous-domaine, un maillage et un nom d'implémentation de module
  ModuleBuildInfo(ISubDomain* sd,const MeshHandle& mesh_handle,const String& name);
  //! Constructeur à partir d'un sous-domaine et un nom d'implémentation de module
  /*! Le maillage considéré est alors le maillage par défaut \a defautMesh() */
  ModuleBuildInfo(ISubDomain* sd,const String& name);

  //! Destructeur
  virtual ~ModuleBuildInfo() { }

 public:
  //! Accès au sous-domaine associé
  ISubDomain* subDomain() const { return m_sub_domain; }
  /*!
   * \brief Accès au maillage associé.
   *
   * Le maillage n'existe pas toujours si le jeu de donnée n'a pas
   * encore été lu. Il est préférable d'utilise meshHandle() à la place
   */
  IMesh* mesh() const { return m_mesh_handle.mesh(); }
  //! Accès au maillage associé
  const MeshHandle& meshHandle() const { return m_mesh_handle; }
  //! Accès au graphe associé au maillage
  IGraph* graph() const;
  //! Nom de l'implémentation recherchée
  const String& name() const { return m_name; }

 public:
  //! Sous-domaine associé
  ISubDomain* m_sub_domain;
  //! Maillage associé
  MeshHandle m_mesh_handle;
  //! Graphe associé
  /*! \warning toujours defaultGraph() pour l'instant */
  IGraph* m_graph;
  //! Nom de l'implémentation recherchée
  String m_name;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

