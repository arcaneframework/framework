﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ICartesianMesh.h                                            (C) 2000-2021 */
/*                                                                           */
/* Interface d'un maillage cartésien.                                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CARTESIANMESH_ICARTESIANMESH_H
#define ARCANE_CARTESIANMESH_ICARTESIANMESH_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ArcaneTypes.h"
#include "arcane/cartesianmesh/CartesianMeshGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class CartesianMeshRenumberingInfo;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneCartesianMesh
 * \brief Interface d'un maillage cartésien.
 */
class ARCANE_CARTESIANMESH_EXPORT ICartesianMesh
{
 public:
  virtual ~ICartesianMesh() {} //<! Libère les ressources

  /*!
   * \brief Récupère ou créé la référence associée à \a mesh.
   *
   * Si aucun gestionnaire de matériau n'est associé à \a mesh, il
   * sera créé lors de l'appel à cette méthode si \a create vaut \a true.
   * Si \a create vaut \a false est qu'aucune gestionnaire n'est associé
   * au maillage, un pointeur nul est retourné.
   * L'instance retournée reste valide tant que le maillage \a mesh existe.
   */
  static ICartesianMesh* getReference(IMesh* mesh, bool create = true);

 public:
  virtual void build() = 0;

 public:
  //! Maillage associé à ce maillage cartésien
  virtual IMesh* mesh() const = 0;

  //! Gestionnaire de trace associé.
  virtual ITraceMng* traceMng() const = 0;

  //! Liste des mailles dans la direction \a dir
  virtual CellDirectionMng cellDirection(eMeshDirection dir) = 0;

  //! Liste des mailles dans la direction \a dir (0, 1 ou 2)
  virtual CellDirectionMng cellDirection(Integer idir) = 0;

  //! Liste des faces dans la direction \a dir
  virtual FaceDirectionMng faceDirection(eMeshDirection dir) = 0;

  //! Liste des faces dans la direction \a dir (0, 1 ou 2)
  virtual FaceDirectionMng faceDirection(Integer idir) = 0;

  //! Liste des noeuds dans la direction \a dir
  virtual NodeDirectionMng nodeDirection(eMeshDirection dir) = 0;

  //! Liste des noeuds dans la direction \a dir (0, 1 ou 2)
  virtual NodeDirectionMng nodeDirection(Integer idir) = 0;

  /*!
   * \brief Calcule les infos pour les accès par direction.
   *
   * Actuellement, les restrictions suivantes existent:
   * - calcule uniquement les infos sur les entités mailles.
   * - suppose que la maille 0 est dans un coin (ne fonctionne que
   * pour le meshgenerator).
   * - les informations de direction sont invalidées si le maillage évolue.
   */
  virtual void computeDirections() = 0;

  /*!
   * \brief Recalcule les informations de cartésiennes après une reprise.
   *
   * Cette méthode doit être appelée à la place de computeDirections()
   * lors d'une reprise.
   */
  virtual void recreateFromDump() = 0;

  //! Informations sur la connectivité
  virtual CartesianConnectivity connectivity() = 0;

  /*!
   * \brief Nombre de patchs du maillage.
   *
   * Il y a toujours au moins un patch qui représente la maillage cartésien
   */
  virtual Integer nbPatch() const = 0;

  /*!
   * \brief Retourne la \a index-ième patch du maillage.
   *
   * Si le maillage est cartésien, il n'y a qu'un seul patch.
   *
   * L'instance retournée reste valide tant que cette instance n'est pas détruite.
   */
  virtual ICartesianMeshPatch* patch(Integer index) const = 0;

  /*!
   * \brief Raffine en 2D un bloc du maillage cartésien.
   *
   * Cette méthode ne peut être appelée que si le maillage est un maillage
   * AMR (IMesh::isAmrActivated()==true).
   *
   * Les mailles dont les positions des centres sont comprises entre
   * \a position et \a (position+length) sont raffinées et les informations
   * de connectivité correspondantes sont mises à jour.
   *
   * Cette opération est collective.
   */
  virtual void refinePatch2D(Real2 position, Real2 length) = 0;

  /*!
   * \brief Renumérote les uniqueId() des entités.
   *
   * Suivant les valeurs de \a v, on renumérote les uniqueId() des faces et/ou 
   * des entités des patches pour avoir la même numérotation
   * quel que soit le découpage.
   */
  virtual void renumberItemsUniqueId(const CartesianMeshRenumberingInfo& v) = 0;

  //! Effectue des vérifications sur la validité de l'instance.
  virtual void checkValid() const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_CARTESIANMESH_EXPORT ICartesianMesh*
arcaneCreateCartesianMesh(IMesh* mesh);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

