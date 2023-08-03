// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MaterialsGlobal.h                                           (C) 2000-2023 */
/*                                                                           */
/* Déclarations générales des matériaux de Arcane.                           */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MATERIALS_MATERIALSGLOBAL_H
#define ARCANE_MATERIALS_MATERIALSGLOBAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/materials/MaterialsCoreGlobal.h"

#include <iosfwd>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifdef ARCANE_COMPONENT_arcane_materials
#define ARCANE_MATERIALS_EXPORT ARCANE_EXPORT
#else
#define ARCANE_MATERIALS_EXPORT ARCANE_IMPORT
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
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
class MeshMaterialVariable;
class MatVarIndex;
class IMeshMaterialSynchronizeBuffer;
class ComponentConnectivityList;
class MeshMaterialMng;

template <typename DataType> class ItemMaterialVariableScalar;
template <typename ItemType,typename DataType> class MeshMaterialVariableScalar;
template <typename ItemType,typename DataType> class MeshMaterialVariableArray;

using MeshMaterialVariableFactoryVariableRefCreateFunc = IMeshMaterialVariable* (*)(const MaterialVariableBuildInfo& vb);

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
  OptimizeMultiMaterialPerEnvironment = 4,
  /*!
   * \brief Indique qu'on utilise la version incrémentale de AllEnvData::forceRecompute().
   *
   * Ce mode n'est actif que si GenericOptimize, OptimizeMultiAddRemove et
   * OptimizeMultiMaterialPerEnvironment sont actifs.
   */
  IncrementalRecompute = 8
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
