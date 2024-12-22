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

  constexpr ARCCORE_HOST_DEVICE MatVarIndex(Int32 array_index,Int32 value_index)
  : m_array_index(array_index), m_value_index(value_index)
  {
  }
  ARCCORE_HOST_DEVICE MatVarIndex(){}
 
 public:

  //! Retourne l'indice du tableau de valeur dans la liste des variables.
  constexpr ARCCORE_HOST_DEVICE Int32 arrayIndex() const { return m_array_index; }

  //! Retourne l'indice dans le tableau de valeur
  constexpr ARCCORE_HOST_DEVICE Int32 valueIndex() const { return m_value_index; }

  //! Positionne l'index
  constexpr ARCCORE_HOST_DEVICE void setIndex(Int32 array_index,Int32 value_index)
  {
    m_array_index = array_index;
    m_value_index = value_index;
  }

  constexpr ARCCORE_HOST_DEVICE void reset()
  {
    m_array_index = (-1);
    m_value_index = (-1);
  }

  constexpr ARCCORE_HOST_DEVICE bool null() const
  {
    return m_value_index==(-1);
  }

 private:
  
  Int32 m_array_index;
  Int32 m_value_index;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_CORE_EXPORT ARCCORE_HOST_DEVICE std::ostream&
operator<<(std::ostream& o,const MatVarIndex& mvi);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline ARCCORE_HOST_DEVICE bool
operator==(MatVarIndex mv1,MatVarIndex mv2)
{
  if (mv1.arrayIndex() != mv2.arrayIndex())
    return false;
  return mv1.valueIndex() == mv2.valueIndex();
}

inline bool ARCCORE_HOST_DEVICE operator!=(MatVarIndex mv1,MatVarIndex mv2)
{
  return !(operator==(mv1,mv2));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneMaterials
 * \brief Index d'un Item matériaux pure dans une variable.
 */
class ARCANE_CORE_EXPORT PureMatVarIndex
{
 public:

  explicit ARCCORE_HOST_DEVICE PureMatVarIndex(Int32 idx)
  : m_index(idx)
  {}

 public:

  Int32 ARCCORE_HOST_DEVICE valueIndex() const { return m_index; }

 private:

  Int32 m_index;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneMaterials
 * \brief Index d'un ComponentItem dans une variable.
 */
class ARCANE_CORE_EXPORT ConstituentItemLocalId
{
 public:

  constexpr ARCCORE_HOST_DEVICE ConstituentItemLocalId()
  : m_local_id(-1, -1)
  {}
  constexpr ARCCORE_HOST_DEVICE explicit ConstituentItemLocalId(MatVarIndex mvi)
  : m_local_id(mvi)
  {}

 public:

  //! Index générique pour accéder aux valeurs d'une variable.
  MatVarIndex ARCCORE_HOST_DEVICE localId() const { return m_local_id; }

 private:

  MatVarIndex m_local_id;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_CORE_EXPORT ARCCORE_HOST_DEVICE std::ostream&
operator<<(std::ostream& o,const ComponentItemLocalId& mvi);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneMaterials
 * \brief Index d'un MatItem dans une variable.
 */
class MatItemLocalId
: public ConstituentItemLocalId
{
 public:

  MatItemLocalId() = default;
  constexpr ARCCORE_HOST_DEVICE explicit MatItemLocalId(MatVarIndex mvi)
  : ConstituentItemLocalId(mvi)
  {}
  constexpr ARCCORE_HOST_DEVICE MatItemLocalId(ComponentItemLocalId lid)
  : ConstituentItemLocalId(lid)
  {}
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneMaterials
 * \brief Index d'un EnvItem dans une variable.
 */
class EnvItemLocalId
: public ConstituentItemLocalId
{
 public:

  EnvItemLocalId() = default;
  constexpr ARCCORE_HOST_DEVICE explicit EnvItemLocalId(MatVarIndex mvi)
  : ConstituentItemLocalId(mvi)
  {}
  constexpr ARCCORE_HOST_DEVICE EnvItemLocalId(ComponentItemLocalId lid)
  : ConstituentItemLocalId(lid)
  {}
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

