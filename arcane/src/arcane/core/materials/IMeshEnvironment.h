// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMeshEnvironment.h                                          (C) 2000-2023 */
/*                                                                           */
/* Interface d'un milieu d'un maillage.                                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_MATERIALS_IMESHENVIRONMENT_H
#define ARCANE_CORE_MATERIALS_IMESHENVIRONMENT_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/materials/IMeshComponent.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneMaterials
 * \brief Interface d'un mmilieu utilisateur.
 */
class ARCANE_CORE_EXPORT IUserMeshEnvironment
{
 public:

  virtual ~IUserMeshEnvironment(){}
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneMaterials
 * \brief Interface d'un milieu d'un maillage.
 * 
 * Les milieux sont créés via IMeshEnvironmentMng::createEnvironment().
 *
 * Les milieux ne peuvent pas être détruits et tous les milieux et leurs
 * matériaux doivent être créés lors de l'initialisation.
 *
 * Un milieu peut éventuellement être vide.
 */
class ARCANE_CORE_EXPORT IMeshEnvironment
: public IMeshComponent
{
 public:

  virtual ~IMeshEnvironment(){}

 public:

  //! Liste des matériaux de ce milieu
  virtual ConstArrayView<IMeshMaterial*> materials() =0;

  //! Nombre de matériaux dans le milieu
  virtual Integer nbMaterial() const =0;

  /*!
   * \brief Identifiant du milieu.
   * Il s'agit aussi de l'indice (en commencant par 0) de ce milieu
   * dans la liste des milieux.
   */
  //virtual Int32 id() const =0;

  //! Milieu utilisateur associé
  virtual IUserMeshEnvironment* userEnvironment() const =0;

  //! Positionne le milieu utilisateur associé
  virtual void setUserEnvironment(IUserMeshEnvironment* umm) =0;

  /*!
   * \brief Maille de ce milieu pour la maille \a c.
   *
   * Si ce milieu n'est pas présent dans la présent dans la maille,
   * la maille milieu nulle est retournée.
   *
   * Le coût de cette fonction est proportionnel au nombre de matériaux
   * présents dans la maille.
   */   
  virtual EnvCell findEnvCell(AllEnvCell c) const =0;

  //! Vue associée à ce milieu
  virtual EnvItemVectorView envView() const =0;

  //! Vue sur la liste des entités pures (associées à la maille globale) du milieu
  virtual EnvPurePartItemVectorView pureEnvItems() const =0;

  //! Vue sur la liste des entités impures (partielles) partielles du milieu
  virtual EnvImpurePartItemVectorView impureEnvItems() const =0;

  //! Vue sur la partie pure ou impure des entités du milieu
  virtual EnvPartItemVectorView partEnvItems(eMatPart part) const =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

