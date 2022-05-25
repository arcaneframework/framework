// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MatVarIndex.h                                               (C) 2000-2022 */
/*                                                                           */
/* Index sur les variables matériaux.                                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_MATERIALS_MATVARINDEX_H
#define ARCANE_CORE_MATERIALS_MATVARINDEX_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/materials/MaterialsCoreGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Représente un index sur les variables matériaux et milieux.
 *
 * L'index comprend 2 valeurs:
 * - la première (arrayIndex()) est le numéro dans la liste des tableaux de la variable.
 * - le deuxième (valueIndex()) est l'indice dans le tableau des valeurs de cette variable.
 *
 * \note Pour des raisons de performance, le constructeur par défaut
 * n'initialise par les membres de cette classe. Il faut donc appeler
 * reset() pour initialiser à une valeur invalide.
 */
class ARCANE_CORE_EXPORT MatVarIndex
{
 public:

  MatVarIndex(Int32 array_index,Int32 value_index)
  : m_array_index(array_index), m_value_index(value_index)
  {
  }
  MatVarIndex(){}
 
 public:

  //! Retourne l'indice du tableau de valeur dans la liste des variables.
  Int32 arrayIndex() const { return m_array_index; }

  //! Retourne l'indice dans le tableau de valeur
  Int32 valueIndex() const { return m_value_index; }

  //! Positionne l'index
  void setIndex(Int32 array_index,Int32 value_index)
  {
    m_array_index = array_index;
    m_value_index = value_index;
  }

  void reset()
  {
    m_array_index = (-1);
    m_value_index = (-1);
  }

  bool null() const
  {
    return m_value_index==(-1);
  }

 private:
  
  Int32 m_array_index;
  Int32 m_value_index;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_CORE_EXPORT std::ostream&
operator<<(std::ostream& o,const MatVarIndex& mvi);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline bool
operator==(MatVarIndex mv1,MatVarIndex mv2)
{
  if (mv1.arrayIndex() != mv2.arrayIndex())
    return false;
  return mv1.valueIndex() == mv2.valueIndex();
}

inline bool operator!=(MatVarIndex mv1,MatVarIndex mv2)
{
  return !(operator==(mv1,mv2));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneMaterials
 * \brief Index d'un Item matériaux dans une variable.
 */
class ARCANE_CORE_EXPORT ComponentItemLocalId
{
 public:
  ComponentItemLocalId() : m_local_id(-1,-1){}
  explicit ComponentItemLocalId(MatVarIndex mvi) : m_local_id(mvi){}
 public:
  MatVarIndex localId() const { return m_local_id; }
 private:
  MatVarIndex m_local_id;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneMaterials
 * \brief Index d'un Item matériaux pure dans une variable.
 */
class ARCANE_CORE_EXPORT PureMatVarIndex
{
 public:
  explicit PureMatVarIndex(Int32 idx) : m_index(idx){}
 public:
  Int32 valueIndex() const { return m_index; }
 private:
  Int32 m_index;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_CORE_EXPORT std::ostream&
operator<<(std::ostream& o,const ComponentItemLocalId& mvi);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

