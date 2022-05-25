// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MaterialsCoreGlobal.h                                       (C) 2000-2022 */
/*                                                                           */
/* Déclarations générales des matériaux de Arcane.                           */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MATERIALS_MATERIALSCOREGLOBAL_H
#define ARCANE_MATERIALS_MATERIALSCOREGLOBAL_H
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

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class AllEnvCellVectorView;
class ComponentCell;
class IMeshBlock;
class IMeshComponent;
class IMeshMaterial;
class IMeshEnvironment;
class MatVarIndex;
class MeshMaterialVariableIndexer;
class MeshMaterialInfo;
class MatCell;
class EnvCell;
class MatItemVectorView;
class EnvItemVectorView;
class MatPurePartItemVectorView;
class MatImpurePartItemVectorView;
class MatPartItemVectorView;
class EnvPurePartItemVectorView;
class EnvImpurePartItemVectorView;
class EnvPartItemVectorView;

class IMeshMaterialMng;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Liste de composants multi-matériaux du maillage.
typedef ConstArrayView<IMeshComponent*> MeshComponentList;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Espace de définition d'une variable matériau.
 */
enum class MatVarSpace
{
  //! Variable ayant des valeurs sur les milieux et matériaux
  MaterialAndEnvironment = 1,
  //! Variable ayant des valeurs uniquement sur les milieux
  Environment
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Partie d'un composant
enum class eMatPart
{
  Pure = 0,
  Impure =1
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
extern "C++" std::ostream&
operator<< (std::ostream& ostr,eOperation operation);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
