// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMeshComponent.h                                            (C) 2000-2025 */
/*                                                                           */
/* Interface d'un composant (matériau ou milieu) d'un maillage.              */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MATERIALS_CORE_IMESHCOMPONENT_H
#define ARCANE_MATERIALS_CORE_IMESHCOMPONENT_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/accelerator/core/AcceleratorCoreGlobal.h"

#include "arcane/core/ItemTypes.h"
#include "arcane/core/materials/MaterialsCoreGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{
class IMeshComponentInternal;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneMaterials
 * \brief Interface d'un composant (matériau ou milieu) d'un maillage.
 */
class ARCANE_CORE_EXPORT IMeshComponent
{
 public:

  virtual ~IMeshComponent() = default;

 public:

  //! Gestionnaire associé.
  virtual IMeshMaterialMng* materialMng() =0;

  //! Gestionnaire de trace associé.
  virtual ITraceMng* traceMng() =0;

  //! Nom du composant
  virtual String name() const =0;

  /*!
   * \brief Groupe des mailles de ce matériau.
   *
   * \warning Ce groupe ne doit pas être modifié. Pour changer
   * le nombre d'éléments d'un matériau, il faut passer
   * par le materialMng().
   */
  virtual CellGroup cells() const =0;

  /*!
   * \brief Identifiant du composant.
   *
   * Il s'agit aussi de l'indice (en commençant par 0) de ce composant
   * dans la liste des composants de ce type.
   * Il existe une liste spécifique pour les matériaux et les milieux
   * et donc un composant qui représente un matériau peut avoir le
   * même id qu'un composant représentant un milieu.
   */
  virtual Int32 id() const =0;

  /*!
   * \brief Maille de ce composant pour la maille \a c.
   *
   * Si le composant n'est pas présent dans la présent dans la maille,
   * la maille nulle est retournée.
   *
   * Le coût de cette fonction est proportionnel au nombre de composants
   * présents dans la maille.
   */   
  virtual ComponentCell findComponentCell(AllEnvCell c) const =0;

  //! Vue associée à ce composant
  virtual ComponentItemVectorView view() const =0;

  //! Vérifie que le composant est valide.
  virtual void checkValid() =0;

  //! Vrai si le composant est un matériau
  virtual bool isMaterial() const =0;

  //! Vrai si le composant est un milieu
  virtual bool isEnvironment() const =0;

  //! Indique si le composant est défini pour l'espace \a space
  virtual bool hasSpace(MatVarSpace space) const =0;

  //! Vue sur la liste des entités pures (associées à la maille globale) du composant
  virtual ComponentPurePartItemVectorView pureItems() const =0;

  //! Vue sur la liste des entités impures (partielles) partielles du composant
  virtual ComponentImpurePartItemVectorView impureItems() const =0;

  //! Vue sur la partie pure ou impure des entités du composant
  virtual ComponentPartItemVectorView partItems(eMatPart part) const =0;

  /*!
   * \brief Retourne le composant sous la forme d'un IMeshMaterial.
   *
   * Si isMaterial()==false, retourne \a nullptr
   */
  virtual IMeshMaterial* asMaterial() =0;

  /*!
   * \brief Retourne le composant sous la forme d'un IMeshMaterial.
   *
   * Si isEnvironment()==false, retourne \a nullptr
   */
  virtual IMeshEnvironment* asEnvironment() =0;

  /*!
   * \brief Positionne une politique d'exécution pour ce constituant
   *
   * \warning Cette méthode est expérimentale. A ne pas utiliser en dehors d'Arcane.
   *
   * La politique d'exécution sélectionnée sera sera utilisée pour
   * les opérations de création ou de modification de EnvCellVector,
   * MatCellVector ou ComponentItemVector.
   *
   * Si \a policy vaut Accelerator::eExecutionPolicy::None (le défaut), c'est la politique du
   * IMeshMaterialMng associé qui est utilisée. Si elle vaut Accelerator::eExecutionPolicy::Sequential
   * ou Accelerator::eExecutionPolicy::Thread, alors l'exécution aura lieu sur l'hôte en séquentiel
   * ou en multi-thread. Les autres valeurs sont invalides.
   *
   * \note Le changement de politique d'exécute s'applique pour toute modification
   * qui a lieu ensuite, même pour les instances de ComponentItemVector déjà créées.
   */
  virtual void setSpecificExecutionPolicy(Accelerator::eExecutionPolicy policy) = 0;

  /*!
   * \brief Politique d'exécution spécifique.
   *
   * \sa setSpecificExecutionPolicy().
   */
  virtual Accelerator::eExecutionPolicy specificExecutionPolicy() const = 0;

 public:

  //! API interne
  virtual IMeshComponentInternal* _internalApi() =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
