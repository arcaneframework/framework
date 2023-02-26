// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* UnstructuredMeshAllocateBuildInfo.h                         (C) 2000-2023 */
/*                                                                           */
/* Informations pour allouer les entités d'un maillage non structuré.        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UNSTRUCTUREDMESHALLOCATEBUILDINFO_H
#define ARCANE_UNSTRUCTUREDMESHALLOCATEBUILDINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ItemTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Informations pour allouer les entités d'un maillage non structuré.
 *
 * Cette classe permet de spécifier les mailles qui seront ajoutées lors
 * de l'allocation initiale du maillage.
 * Il faut appeler setMeshDimension() pour spécifier la dimension du maillage
 * puis appeler addCell() pour chaque maille qu'on souhaite ajouter. Une fois
 * toute les mailles ajoutées il faut appeler allocateMesh().
 */
class ARCANE_CORE_EXPORT UnstructuredMeshAllocateBuildInfo
{
  class Impl;

 public:

  explicit UnstructuredMeshAllocateBuildInfo(IPrimaryMesh* mesh);
  ~UnstructuredMeshAllocateBuildInfo();

 public:

  UnstructuredMeshAllocateBuildInfo(UnstructuredMeshAllocateBuildInfo&& from) = delete;
  UnstructuredMeshAllocateBuildInfo(const UnstructuredMeshAllocateBuildInfo& from) = delete;
  UnstructuredMeshAllocateBuildInfo& operator=(UnstructuredMeshAllocateBuildInfo&& from) = delete;
  UnstructuredMeshAllocateBuildInfo& operator=(const UnstructuredMeshAllocateBuildInfo& from) = delete;

 public:

  /*!
   * \brief Pre-alloue la mémoire.
   *
   * Pré-alloue les tableaux contenant la connectivité pour contenir \a nb_cell
   * mailles et \a nb_connectivity_node pour la liste des noeuds
   * des mailles.
   *
   * Cette méthode est optionnelle et n'est utile que pour
   * optimiser la gestion mémoire.
   *
   * Par exemple, si on sait que notre maillage contiendra 300 quadrangles
   * alors on peut utiliser preAllocate(300,300*4).
   */
  void preAllocate(Int32 nb_cell,Int64 nb_connectivity_node);

  //! Positionne la dimension du maillage
  void setMeshDimension(Int32 v);

  //! Ajoute une maille au maillage
  void addCell(ItemTypeId type_id, Int64 cell_uid, SmallSpan<const Int64> nodes_uid);

  /*!
   * \brief Alloue le maillage avec les mailles ajoutées lors de l'appel à addCell().
   *
   * Cette méthode est collective.
   */
  void allocateMesh();

 private:

  Impl* m_p = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
