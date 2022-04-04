// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMeshBlock.h                                                (C) 2000-2013 */
/*                                                                           */
/* Interface d'un bloc d'un maillage.                                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MATERIALS_IMESHBLOCK_H
#define ARCANE_MATERIALS_IMESHBLOCK_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ItemTypes.h"
#include "arcane/materials/MaterialsGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MATERIALS_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class AllEnvCellVectorView;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneMaterials
 * \brief Interface d'un bloc d'un maillage.
 * 
 * Les blocs sont créés via IMeshMaterialMng::createBlock().
 *
 * Les blocs ne peuvent pas être détruits et doivent être créés lors
 * de l'initialisation.
 *
 * La notion de bloc est optionnelle et il n'est pas nécessaire d'avoir
 * des blocs pour utiliser les milieux et les matériaux.
 *
 * Un bloc se caractérise par un nom (name()), un groupe de mailles (cells())
 * et une liste de milieux (environments()).
 *
 * A noter qu'en théorie le groupe de mailles (cells())
 * est indépendant de la liste des milieux mais que pour des raisons de
 * cohérence, il est préférable que ce groupe corresponde à l'union des
 * milieux du bloc. Cependant, aucune vérification de cette cohérence n'est effectuée.
 *
 * Il est possible d'utiliser une instance ce cette classe comme argument à
 * ENUMERATE_ENV ou à ENUMERATE_ALLENVCELL.
 */
class ARCANE_MATERIALS_EXPORT IMeshBlock
{
 public:

  virtual ~IMeshBlock(){}

 public:

  //! Gestionnaire associé.
  virtual IMeshMaterialMng* materialMng() =0;

  //! Nom du bloc
  virtual const String& name() const =0;

  /*!
   * \brief Groupe des mailles de ce bloc.
   */
  virtual const CellGroup& cells() const =0;

  //! Liste des milieux de ce bloc
  virtual ConstArrayView<IMeshEnvironment*> environments() =0;

  //! Nombre de milieux dans le bloc
  virtual Integer nbEnvironment() const =0;

  /*!
   * \brief Identifiant du bloc.
   * Il s'agit aussi de l'indice (en commencant par 0) de ce bloc
   * dans la liste des blocs.
   */
  virtual Int32 id() const =0;

  //! Vue sur les mailles milieux correspondant à ce bloc.
  virtual AllEnvCellVectorView view() =0;
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

