// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MaterialGlobal.h                                            (C) 2000-2019 */
/*                                                                           */
/* Déclarations générales des matériaux de Arcane.                           */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MATERIALS_MATERIALGLOBAL_H
#define ARCANE_MATERIALS_MATERIALGLOBAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"

#include <iosfwd>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#define MATERIALS_BEGIN_NAMESPACE  namespace Materials {
#define MATERIALS_END_NAMESPACE    }

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifdef ARCANE_COMPONENT_arcane_materials
#define ARCANE_MATERIALS_EXPORT ARCANE_EXPORT
#else
#define ARCANE_MATERIALS_EXPORT ARCANE_IMPORT
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IMeshBlock;
class IMeshComponent;
class IMeshMaterial;
class IMeshEnvironment;
class IMeshMaterialVariable;
class IMeshMaterialMng;
class ComponentItemInternal;
class ComponentItem;
class ComponentItemVector;
class ComponentItemVectorView;
class ComponentPartItemVectorView;
class ComponentPurePartItemVectorView;
class ComponentImpurePartItemVectorView;
class EnvPartItemVectorView;
class EnvPurePartItemVectorView;
class EnvImpurePartItemVectorView;
class EnvItemVectorView;
class MatPartItemVectorView;
class MatPurePartItemVectorView;
class MatImpurePartItemVectorView;
class MatItemVectorView;
class MaterialVariableBuildInfo;
class MeshMaterialVariableSynchronizerList;
class MatVarIndex;

template <typename DataType> class ItemMaterialVariableScalar;
template <typename ItemType,typename DataType> class MeshMaterialVariableScalar;
template <typename ItemType,typename DataType> class MeshMaterialVariableArray;
template<typename DataType> class CellMaterialVariableScalarRef;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

typedef ComponentItemInternal* ComponentItemInternalPtr;
typedef IMeshComponent* IMeshComponentPtr;
typedef IMeshMaterial* IMeshMaterialPtr;
typedef IMeshEnvironment* IMeshEnvironmentPtr;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Liste de composants multi-matériaux du maillage.
typedef ConstArrayView<IMeshComponent*> MeshComponentList;

//! Liste de composants multi-matériaux du maillage.
typedef ConstArrayView<IMeshComponent*> MeshComponentListView;

//! Liste de milieux du maillage.
typedef ConstArrayView<IMeshEnvironment*> MeshEnvironmentListView;

//! Liste de matériaux du maillage.
typedef ConstArrayView<IMeshMaterial*> MeshMaterialListView;

//! Liste de ComponentCell issues d'un IMeshComponent.
typedef ComponentItemVector ComponentCellVector;

//! Type de la vue sur un EnvCellVector
typedef EnvItemVectorView EnvCellVectorView;

//! Type de la vue sur un MatCellVector
typedef MatItemVectorView MatCellVectorView;

//! Type de la vue sur un ComponentCellVector
typedef ComponentItemVectorView ComponentCellVectorView;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

static const Int32 LEVEL_MATERIAL = 1;
static const Int32 LEVEL_ENVIRONMENT = 2;
static const Int32 LEVEL_ALLENVIRONMENT = 0;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief flags pour paramétrer le comportement lors d'une modification
 * de matériaux ou milieux.
 */
enum class eModificationFlags
{
  //! Active les optimisations génériques
  GenericOptimize = 1,
  //! Active les optimisations pour les ajouts/supressions multiples
  OptimizeMultiAddRemove = 2,
  /*!
   * \brief Active les optimisations lorsque plusieurs matériaux sont présents dans
   * un milieu.
   */
  OptimizeMultiMaterialPerEnvironment = 4
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Opération de mise à jour des milieux/matériaux.
 */
enum class eOperation
{
  //! Ajoute des entités
  Add,
  //! Supprime des entités
  Remove
};

//! Opérateur de sortie sur un flot
extern "C++" ostream&
operator<< (ostream& ostr,eOperation operation);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Espace de définition d'une variable matériau.
 */
enum class MatVarSpace
{
  // TODO: renommer Material en MaterialAndEnvironment
  //! Variable ayant des valeurs sur les milieux et matériaux
  MaterialAndEnvironment = 1,
  //! Variable ayant des valeurs uniquement sur les milieux
  Environment
};

//! Partie d'un composant
enum class eMatPart
{
  Pure = 0,
  Impure =1
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Materials

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
